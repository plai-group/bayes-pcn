import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader
from typing import Dict, NamedTuple, Tuple
import wandb

from .pcnet import PCNetEnsemble
from .util import unnormalize


class DataBatch(NamedTuple):
    train: Tuple[torch.Tensor, torch.Tensor]
    tests: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
    original_shape: torch.Size
    train_pred: Tuple[torch.Tensor, torch.Tensor] = None
    tests_pred: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = None


def get_next_data_batch(train_loader: DataLoader, test_loaders: Dict[str, DataLoader]) -> DataBatch:
    """Given train and test loaders that are iterators, return a DataBatch object.
    NOTE: Do not call next() on dataloaders outside this method!

    Args:
        train_loader (DataLoader): Single train dataloader.
        test_loaders (Dict[str, DataLoader]): A dictionary of test dataloaders.

    Returns:
        DataBatch: A DataBatch object with reshaped versions of the next iterator datapoints.
    """
    X, _ = next(train_loader)
    X_shape = X.shape  # original data shape
    X = X.reshape(X_shape[0], -1)
    test_batch = {}
    for name, test_loader in test_loaders.items():
        X_test, fixed_indices = next(test_loader)
        test_batch[name] = (X_test.reshape(X_test.shape[0], -1),
                            fixed_indices.reshape(fixed_indices.shape[0], -1))
    return DataBatch(train=(X, None), tests=test_batch, original_shape=X_shape)


def score_data_batch(data_batch: DataBatch, model: PCNetEnsemble, acc_thresh: float,
                     prefix: str = None) -> Tuple[Dict[str, float], DataBatch]:
    """Compute the model's recall MSE and accuracy on train and test batches in data_batch.

    Args:
        data_batch (DataBatch): _description_
        model (PCNetEnsemble): _description_
        acc_thresh (float): _description_
        prefix (str, optional): _description_. Defaults to None.

    Returns:
        Dict[str, float]: _description_
    """
    result = {}
    X_truth = data_batch.train[0]
    prefix = f"/{prefix}" if prefix is not None else ""
    train_pred = model.infer(X_obs=X_truth, fixed_indices=None).data
    mse_train, acc_train = score(X_pred=train_pred, X_truth=X_truth, acc_thresh=acc_thresh,
                                 fixed_indices=None)
    result.update({f"train{prefix}_mse": mse_train, f"train{prefix}_acc": acc_train})
    tests_pred = {}
    for name, (X, fixed_indices) in data_batch.tests.items():
        X_pred = model.infer(X_obs=X, fixed_indices=fixed_indices).data
        mse_test, acc_test = score(X_pred=X_pred, X_truth=X_truth, acc_thresh=acc_thresh,
                                   fixed_indices=fixed_indices)
        result.update({f"{name}{prefix}_mse": mse_test, f"{name}{prefix}_acc": acc_test})
        tests_pred[name] = (X_pred, fixed_indices)
    result_data_batch = DataBatch(train=data_batch.train, tests=data_batch.tests,
                                  original_shape=data_batch.original_shape,
                                  train_pred=(train_pred, None), tests_pred=tests_pred)
    return result, result_data_batch


def plot_data_batch(data_batch: DataBatch, caption: str = None) -> wandb.Image:
    """Plot the first image in the batch under various transformations.

    Args:
        data_batch (DataBatch): _description_
        caption (str, optional): _description_. Defaults to None.

    Returns:
        wandb.Image: _description_
    """
    rows = [None] * (1 + len(data_batch.tests)) * 2
    X_shape = data_batch.original_shape
    rows[0] = data_batch.train[0][:1].reshape(X_shape[1:])
    rows[len(rows)//2] = data_batch.train_pred[0][:1].reshape(X_shape[1:])
    for i, (X, X_pred) in enumerate(zip(data_batch.tests.values(), data_batch.tests_pred.values())):
        rows[i+1] = X[0][:1].reshape(X_shape[1:])
        rows[len(rows)//2+i+1] = X_pred[0][:1].reshape(X_shape[1:])
    img = unnormalize(torchvision.utils.make_grid(rows, nrow=len(rows)//2))
    return wandb.Image(img, caption=caption)


def generate_samples(model: PCNetEnsemble, X_shape: torch.Size, d_batch: int,
                     caption: str = None) -> wandb.Image:
    sample_obj = model.sample(d_batch=d_batch)
    X_gen, log_joint = sample_obj.data, [round(lj, 3) for lj in sample_obj.log_joint.tolist()]
    img = torchvision.utils.make_grid(X_gen.reshape(-1, *X_shape[1:]), nrow=min(d_batch, 4))
    log_joint_msg = f"Sample Weights: {log_joint}"
    caption = log_joint_msg if caption is None else caption + "\n" + log_joint_msg
    return wandb.Image(unnormalize(img), caption=caption)


def score(X_pred: torch.Tensor, X_truth: torch.Tensor, acc_thresh: float,
          fixed_indices: torch.Tensor = None) -> Tuple[float, float]:
    """_summary_

    Args:
        X_pred (torch.Tensor): Model prediction matrix of shape <d_batch x x_dim>.
        X_truth (torch.Tensor): Ground truth matrix of shape <d_batch x x_dim>.
        acc_thresh (float): MSE has to be lower than this for a datapoint to be considered
            correctly retrieved.
        fixed_indices (torch.Tensor, Optional): Matrix of shape <d_batch x x_dim> that denotes
            which data-specific indices to prevent modification when predicting.

    Returns:
        Tuple[float, float]: Mean MSE and accuracy of the model's prediction on current batch.
    """
    scaling = X_pred.shape[-1]
    if fixed_indices is not None:
        # Only compare non-fixed data indices for scoring.
        X_truth = X_truth * fixed_indices.logical_not()
        X_pred = X_pred * fixed_indices.logical_not()
        scaling = scaling * fixed_indices.logical_not().sum(dim=-1) / fixed_indices.shape[-1]
    mse = ((X_truth - X_pred)**2).sum(dim=-1) / scaling
    acc = mse[mse < acc_thresh].shape[0] / mse.shape[0]
    return mse.mean().item(), acc


def train_epoch(train_loader: DataLoader, test_loaders: Dict[str, DataLoader], model: PCNetEnsemble,
                epoch: int, log_every: int = 1, plot_every: int = 1, acc_thresh: float = 0.005,
                fast_mode: bool = False) -> PCNetEnsemble:
    """Update model on all datapoint once. Assess model performance on unnoised and noised data.

    Args:
        train_loader (DataLoader): _description_
        test_loaders (Dict[str, DataLoader]): _description_
        model (PCNet): _description_
        log_every (int, optional): _description_. Defaults to 1.
        plot_every (int, optional): _description_. Defaults to 1.
        acc_thresh (float, optional): _description_. Defaults to 0.005.

    Returns:
        PCNet: _description_
    """
    assert (log_every % plot_every) == 0
    train_loader = iter(train_loader)
    test_loaders = {name: iter(test_loader) for name, test_loader in test_loaders.items()}

    for i in range(1, len(train_loader)+1):
        # Prepare train and test data batches for learning and evaluation
        curr_batch = get_next_data_batch(train_loader=train_loader, test_loaders=test_loaders)
        X_shape = curr_batch.original_shape
        if i == 1:
            first_batch = curr_batch
        wandb_dict = {"step": (epoch - 1) * len(train_loader) + i}
        init_img, unseen_img, curr_img = None, None, None

        # Evaluate model performance on the current data batch before update
        if not fast_mode and (i % log_every) == 0:
            result, pred_batch = score_data_batch(data_batch=curr_batch, model=model,
                                                  acc_thresh=acc_thresh, prefix='unseen')
            wandb_dict.update(result)
            if (i % plot_every) == 0:
                unseen_img = plot_data_batch(data_batch=pred_batch)

        # Update model
        X_train = curr_batch.train[0]
        model.learn(X_obs=X_train)

        # Evaluate model performance on the first data batch
        if not fast_mode and (i % log_every) == 0:
            result, pred_batch = score_data_batch(data_batch=first_batch, model=model,
                                                  acc_thresh=acc_thresh, prefix='initial')
            wandb_dict.update(result)
            if (i % plot_every) == 0:
                init_img = plot_data_batch(data_batch=pred_batch)

        # Evaluate model performance on the current data batches after update
        if (i % log_every) == 0:
            result, pred_batch = score_data_batch(data_batch=curr_batch, model=model,
                                                  acc_thresh=acc_thresh, prefix='current')
            wandb_dict.update(result)
            if (i % plot_every) == 0:
                curr_img = plot_data_batch(data_batch=pred_batch)

        # Generate data samples
        if (i % plot_every) == 0:
            gen_img = generate_samples(model=model, X_shape=X_shape, d_batch=8,
                                       caption="Model samples via ancestral sampling.")

        # Log to wandb
        if (i % log_every) == 0:
            wandb_dict = {f"iteration/{k}": v for k, v in wandb_dict.items()}
            wandb_dict["Initial Image"] = init_img
            wandb_dict["Unseen Image"] = unseen_img
            wandb_dict["Current Image"] = curr_img
            wandb_dict["Generated Image"] = gen_img
            wandb.log(wandb_dict)
    return model


def score_epoch(train_loader: DataLoader, test_loaders: Dict[str, DataLoader], model: PCNetEnsemble,
                acc_thresh: float, epoch: int) -> Dict[str, float]:
    train_loader = iter(train_loader)
    test_loaders = {name: iter(test_loader) for name, test_loader in test_loaders.items()}

    scores = []
    for _ in range(1, len(train_loader)+1):
        batch = get_next_data_batch(train_loader=train_loader, test_loaders=test_loaders)
        scores.append(score_data_batch(data_batch=batch, model=model, acc_thresh=acc_thresh)[0])
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
