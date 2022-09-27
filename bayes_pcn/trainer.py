import pandas as pd
import torch
from torch.utils.data import DataLoader
from typing import Any, Dict, Tuple
import wandb

from bayes_pcn.pcnet.util import fixed_indices_exists

from .pcnet import PCNetEnsemble, DataBatch
from .util import *


def score_data_batch(data_batch: DataBatch, model: PCNetEnsemble, acc_thresh: float,
                     n_repeat: int, prefix: str = None) -> Tuple[Dict[str, float], DataBatch]:
    """Compute the model's recall MSE and accuracy on train and test batches in data_batch.

    Args:
        data_batch (DataBatch): _description_
        model (PCNetEnsemble): _description_
        acc_thresh (float): _description_
        n_repeat (int): _description_
        prefix (str, optional): _description_. Defaults to None.

    Returns:
        Dict[str, float]: _description_
    """
    result = {}
    batch_info = {}
    all_mses = {}
    X_truth = data_batch.train[0]
    prefix = f"/{prefix}" if prefix is not None else ""

    train_result = model.infer(X_obs=X_truth, n_repeat=n_repeat, fixed_indices=None)
    mse_train, acc_train, mses_train = score(X_pred=train_result.data, X_truth=X_truth,
                                             acc_thresh=acc_thresh, fixed_indices=None)
    result.update({f"train{prefix}_mse": mse_train, f"train{prefix}_acc": acc_train})
    batch_info["train"] = train_result.info
    all_mses["train"] = mses_train

    tests_pred = {}
    for name, (X, fixed_indices) in data_batch.tests.items():
        test_result = model.infer(X_obs=X, n_repeat=n_repeat, fixed_indices=fixed_indices)
        batch_info[name] = test_result.info
        mse_base, _, _ = score(X_pred=X, X_truth=X_truth,
                               acc_thresh=acc_thresh, fixed_indices=fixed_indices)
        mse_test, acc_test, mses_test = score(X_pred=test_result.data, X_truth=X_truth,
                                              acc_thresh=acc_thresh, fixed_indices=fixed_indices)
        result.update({f"{name}{prefix}_base": mse_base,
                       f"{name}{prefix}_mse": mse_test,
                       f"{name}{prefix}_acc": acc_test})
        tests_pred[name] = (test_result.data, fixed_indices)
        all_mses[name] = mses_test

    batch_info["all_mses"] = all_mses
    result_data_batch = DataBatch(train=data_batch.train, tests=data_batch.tests,
                                  train_pred=(train_result.data, None), tests_pred=tests_pred,
                                  original_shape=data_batch.original_shape, info=batch_info)
    return result, result_data_batch


def score(X_pred: torch.Tensor, X_truth: torch.Tensor, acc_thresh: float,
          fixed_indices: torch.Tensor = None) -> Tuple[float, float, torch.Tensor]:
    """_summary_

    Args:
        X_pred (torch.Tensor): Model prediction matrix of shape <d_batch x x_dim>.
        X_truth (torch.Tensor): Ground truth matrix of shape <d_batch x x_dim>.
        acc_thresh (float): MSE has to be lower than this for a datapoint to be considered
            correctly retrieved.
        fixed_indices (torch.Tensor, Optional): Matrix of shape <d_batch x x_dim> that denotes
            which data-specific indices to prevent modification when predicting.

    Returns:
        Tuple[float, float, torch.Tensor]: Mean MSE and accuracy of the model's prediction on
            the current batch + individual datapoint scores.
    """
    scaling = X_pred.shape[-1]
    if fixed_indices_exists(fixed_indices=fixed_indices):
        # Only compare non-fixed data indices for scoring.
        X_truth = X_truth * fixed_indices.logical_not()
        X_pred = X_pred * fixed_indices.logical_not()
        scaling = scaling * fixed_indices.logical_not().sum(dim=-1) / fixed_indices.shape[-1]
    mse = ((X_truth - X_pred)**2).sum(dim=-1) / scaling
    acc = mse[mse < acc_thresh].shape[0] / mse.shape[0]
    return mse.mean().item(), acc, mse


def train_epoch(train_loader: DataLoader, test_loaders: Dict[str, DataLoader], model: PCNetEnsemble,
                epoch: int, n_repeat: int, log_every: int = 1, save_every: int = None,
                acc_thresh: float = 0.005, fast_mode: bool = False, args: DotDict = None,
                forget_every: int = None) -> PCNetEnsemble:
    """Update model on all datapoint once. Assess model performance on unnoised and noised data.

    Args:
        train_loader (DataLoader): _description_
        test_loaders (Dict[str, DataLoader]): _description_
        model (PCNetEnsemble): _description_
        epoch (int): _description_
        n_repeat (int): _description_
        log_every (int, optional): _description_. Defaults to 1.
        save_every (int, optional): _description_. Defaults to None.
        acc_thresh (float, optional): _description_. Defaults to 0.005.
        fast_mode (bool, optional): _description_. Defaults to False.
        args (DotDict, optional): _description_. Defaults to None.

    Returns:
        PCNetEnsemble: _description_
    """
    def should_log(index):
        if log_every is None:
            return index in [2**i for i in range(1, 16)]
        else:
            return index % log_every == 0

    def should_save(index):
        if save_every is None:
            return index in [2**i for i in range(1, 16)]
        else:
            return index % save_every == 0

    def should_forget(index):
        return forget_every is not None and index > 1 and (index - 1) % forget_every == 0

    train_loader = iter(train_loader)
    test_loaders = {name: iter(test_loader) for name, test_loader in test_loaders.items()}

    for i in range(1, len(train_loader)+1):
        # Prepare train and test data batches for learning and evaluation
        curr_batch = get_next_data_batch(train_loader=train_loader, test_loaders=test_loaders)
        X_shape = curr_batch.original_shape
        wandb_dict = {"step": (epoch - 1) * len(train_loader) + i - (1 if epoch > 1 else 0)}
        init_img, unseen_img, curr_img, gen_img = None, None, None, None
        update_img, layer_update_imgs = None, []
        if i == 1:
            first_batch = curr_batch

        if should_forget(index=i):
            model.forget()

        if not fast_mode and should_log(index=i):
            # Evaluate model performance on the current data batch before update
            result, pred_batch = score_data_batch(data_batch=curr_batch, model=model,
                                                  acc_thresh=acc_thresh, n_repeat=n_repeat,
                                                  prefix='unseen')
            wandb_dict.update(result)
            unseen_img = plot_data_batch(data_batch=pred_batch)

        # Update model
        X_train = curr_batch.train[0]
        update_result = model.learn(X_obs=X_train)

        if not fast_mode and should_log(index=i):
            # Evaluate model performance on the first data batch
            result, pred_batch = score_data_batch(data_batch=first_batch, model=model,
                                                  acc_thresh=acc_thresh, n_repeat=n_repeat,
                                                  prefix='initial')
            wandb_dict.update(result)
            init_img = plot_data_batch(data_batch=pred_batch)

        if should_log(index=i):
            # Evaluate model performance on the current data batches after update
            result, pred_batch = score_data_batch(data_batch=curr_batch, model=model,
                                                  acc_thresh=acc_thresh, n_repeat=n_repeat,
                                                  prefix='current')
            wandb_dict.update(result)
            curr_img = plot_data_batch(data_batch=pred_batch)
            # Generate data samples and plot log loss trajectory
            gen_img = generate_samples(model=model, X_shape=X_shape, d_batch=8,
                                       caption="Model samples via ancestral sampling.")
            update_img = plot_update_energy(update_result=update_result,
                                            caption="Activation update energy curve (avg/min/max).")
            caption_prefix = "Activation update energy curve for layer: "
            layer_update_imgs = plot_layerwise_update_energy(update_result=update_result,
                                                             caption=caption_prefix)
            # Plot mean L1 norms of the first PCNet parameters
            norm_info = dict()
            for i_layer, layer in enumerate(model._pcnets[0].layers):
                norm_info[f"layer{i_layer+1}_R_avg_norm"] = layer._R.abs().mean().item()
                norm_info[f"layer{i_layer+1}_U_avg_norm"] = layer._U.abs().mean().item()
                norm_info[f"layer{i_layer+1}_U_avg_diag_norm"] = layer._U.diag().abs().mean().item()
            wandb_dict.update(norm_info)

            # Log to wandb
            wandb_dict = {f"iteration/{k}": v for k, v in wandb_dict.items()}
            wandb_dict["Current Image"] = curr_img
            wandb_dict["Generated Image"] = gen_img
            wandb_dict["Energy/Free Energy Plot"] = update_img
            for i_layer, layer_update_img in enumerate(layer_update_imgs):
                wandb_dict[f"Energy/Negative Log Joint Plot (Layer {i_layer})"] = layer_update_img

            if not fast_mode:
                wandb_dict["Initial Image"] = init_img
                wandb_dict["Unseen Image"] = unseen_img
            wandb.log(wandb_dict)

        if should_save(index=i):
            save_config((model, args), f"{args.path}/model_{epoch}_{i}.pt")
    return model


def score_epoch(train_loader: DataLoader, test_loaders: Dict[str, DataLoader], model: PCNetEnsemble,
                acc_thresh: float, epoch: int, n_repeat: int) -> Dict[str, float]:
    train_loader = iter(train_loader)
    test_loaders = {name: iter(test_loader) for name, test_loader in test_loaders.items()}

    scores = []
    for _ in range(len(train_loader)):
        batch = get_next_data_batch(train_loader=train_loader, test_loaders=test_loaders)
        score, pred_batch = score_data_batch(data_batch=batch, model=model,
                                             acc_thresh=acc_thresh, n_repeat=n_repeat)
        scores.append(score)
    score_df = pd.DataFrame(scores)

    wandb_dict = {"step": epoch}
    for key, val in score_df.mean(axis=0).iteritems():
        wandb_dict[f"{key}/avg"] = val
    for key, val in score_df.min(axis=0).iteritems():
        wandb_dict[f"{key}/min"] = val
    for key, val in score_df.max(axis=0).iteritems():
        wandb_dict[f"{key}/max"] = val
    wandb.log({f"epoch/{k}": v for k, v in wandb_dict.items()})
    return wandb_dict


def model_dispatcher(args: Dict[str, Any], dataset_info: Dict[str, Any]) -> PCNetEnsemble:
    return PCNetEnsemble(n_models=args.n_models, n_layers=args.n_layers, h_dim=args.h_dim,
                         x_dim=dataset_info.get('x_dim'), act_fn=args.act_fn, infer_T=args.T_infer,
                         infer_lr=args.activation_lr, sigma_prior=args.sigma_prior,
                         sigma_obs=args.sigma_obs, sigma_data=args.sigma_data,
                         n_proposal_samples=args.n_proposal_samples,
                         activation_optim=args.activation_optim,
                         activation_init_strat=args.activation_init_strat,
                         weight_init_strat=args.weight_init_strat,
                         layer_log_prob_strat=args.layer_log_prob_strat,
                         layer_sample_strat=args.layer_sample_strat,
                         layer_update_strat=args.layer_update_strat,
                         ensemble_log_joint_strat=args.ensemble_log_joint_strat,
                         ensemble_proposal_strat=args.ensemble_proposal_strat,
                         scale_layer=args.scale_layer, resample=args.resample,
                         weight_lr=args.weight_lr, beta_forget=args.beta_forget,
                         beta_noise=args.beta_noise, mhn_metric=args.mhn_metric, bias=args.bias,
                         n_elbo_particles=args.n_elbo_particles, kernel_type=args.kernel_type)
