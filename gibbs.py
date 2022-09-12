import multiprocessing
from pyro.infer import MCMC, NUTS
from pyro.infer.mcmc.util import initialize_model, summary
import torch
from torch.utils.data import DataLoader
import torchvision
from typing import Dict, Any, List, Tuple
import wandb
from bayes_pcn.const import LayerLogProbStrat, LayerSampleStrat
from bayes_pcn.dataset import dataset_dispatcher, separate_train_test

from bayes_pcn.pcnet.ensemble import PCNetPosterior, PCNetEnsemble
from bayes_pcn.__main__ import get_parser, model_dispatcher, setup
from bayes_pcn.pcnet.util import ActivationGroup, DataBatch, UpdateResult
from bayes_pcn.trainer import get_next_data_batch, plot_data_batch, plot_update_energy,\
                              generate_samples, score_data_batch
from bayes_pcn.util import *


def parse_args():
    parser = get_parser()
    parser.add_argument("--T-gibbs", type=int, default=1000)
    parser.add_argument("--T-mh", type=int, default=100)
    parser.add_argument("--mh-step-size", type=float, default=0.0001)
    parser.add_argument("--gibbs-burnin", type=int, default=1)
    return parser.parse_args()


def sample_activations_posterior_data(args) -> List[List[torch.Tensor]]:
    model, obs, a_group, T_mh, mh_step_size = args
    # If applicable, warm start MCMC with previous iteration sample
    a_group.clamp(obs=True, hidden=False)
    a_group.device = model.device

    # Run NUTS and take a single sample from the chain
    # adapt_step_size=False, step_size=mh_step_size
    if mh_step_size > 0.:
        nuts_args = dict(model=model.sample, full_mass=True, max_tree_depth=5,
                         adapt_step_size=False, step_size=mh_step_size)
    else:
        nuts_args = dict(model=model.sample, full_mass=True, max_tree_depth=5)
    nuts_kernel = NUTS(**nuts_args)
    mcmc = MCMC(
        nuts_kernel,
        num_samples=T_mh//10 + 1,
        warmup_steps=9 * (T_mh//10),
        # num_samples=1,
        # warmup_steps=T_mh,
        disable_progbar=False
    )
    mcmc.run(a_group.d_batch, a_group)
    samples = mcmc.get_samples()

    # Format data into ActivationGroup and UpdateResult.
    hidden_kvs = sorted(list(samples.items()), key=lambda n: int(n[0].split('_')[-1]))
    n_h_layers, n_samples, d_batch = len(hidden_kvs), len(hidden_kvs[-1][-1]), a_group.d_batch

    # Collect layer wise activations per sample
    data_activations = []
    for i_sample in range(n_samples):
        activations = []
        for i_h_layer in reversed(range(0, n_h_layers)):
            layer_errs = hidden_kvs[i_h_layer][-1][i_sample]
            if i_h_layer == n_h_layers-1:
                pred_args = {"d_batch": d_batch}
            else:
                pred_args = {"X_in": activations[-1]}
            layer_pred_mean = model._pcnets[0].layers[i_h_layer+1].predict(**pred_args)
            layer_acts = layer_errs + layer_pred_mean
            activations.append(layer_acts)
        activations.append(obs)
        data_activations.append(activations[::-1])
    return data_activations


def sample_activations_posterior_mp(model: PCNetEnsemble, data_batch: DataBatch,
                                    prev_a_group: ActivationGroup, T_mh: int,
                                    mh_step_size: float,
                                    ) -> Tuple[ActivationGroup, UpdateResult]:
    """Sample all layer activations for current data batch. Activations include
    the observed layer activations.

    Args:
        model_fn (PCNet): _description_
        X (torch.Tensor): _description_

    Returns:
        Tuple[ActivationGroup, UpdateResult]: Single sample of fully initialized model activations
            and UpdateResult object with negative log joint plot obtained during NUTS.
    """
    # Warm start NUTS on previous chain sample
    X_obs = data_batch.train[0]
    n_data = len(X_obs)
    all_activations = []

    inputs = []
    for i_data in range(n_data):
        obs = X_obs[i_data:i_data+1]
        if prev_a_group is None:
            a_group = model.initialize_activation_group(X_obs=obs)
        else:
            a_group = ActivationGroup(activations=prev_a_group.get_datapoint(data_index=i_data))
        inputs.append((model, obs, a_group, T_mh, mh_step_size))
    all_activations = [sample_activations_posterior_data(args=args) for args in inputs]
    # import time
    # start = time.time()
    # pool_obj = multiprocessing.Pool(processes=2)
    # all_activations = pool_obj.map(sample_activations_posterior_data, inputs)
    # elapsed = time.time() - start
    # print(elapsed)
    # pool_obj.terminate()

    n_samples, n_layers = len(all_activations[0]), len(all_activations[0][0])
    formatted = [[[] for _ in range(n_layers)] for _ in range(n_samples)]
    for data_activations in all_activations:
        for i_sample in range(n_samples):
            for i_layer in range(n_layers):
                formatted[i_sample][i_layer].append(data_activations[i_sample][i_layer])
    formatted = [[torch.cat(layer_acts, dim=0) for layer_acts in acts] for acts in formatted]

    a_groups = [ActivationGroup(activations=activations) for activations in formatted]
    for a_group in a_groups:
        a_group.clamp(obs=True, hidden=True)
        a_group.device = model.device
    posterior_sample = a_groups[-1]
    ljs = [model.log_joint(a_group=a_group).log_prob.sum().item() for a_group in a_groups]
    mcmc_info = dict(model_0=dict(mean_losses=ljs, min_losses=ljs, max_losses=ljs))
    return posterior_sample, UpdateResult(pcnets=None, log_weights=None, info=mcmc_info)


def sample_parameters_posterior(model: PCNetEnsemble, a_groups: List[ActivationGroup]
                                ) -> Tuple[List[torch.Tensor], float]:
    """Update the new model on all activation groups then sample model parameters.

    Args:
        model (PCNetEnsemble): PCNetEnsemble with a single PCNet.
        a_groups (List[ActivationGroup]): A list of activation groups.

    Returns:
        List[torch.Tensor]: A list of sampled layer weights.
        float: Log probability of the sampled layer weights.
    """
    pcnet = model._pcnets[0]
    for a_group in a_groups:
        pcnet.update_weights(a_group=a_group)
    parameters_sample = model.sample_parameters()
    parameters_log_prob = model.parameters_log_prob(parameters_sample=parameters_sample)
    return parameters_sample, parameters_log_prob


def train_gibbs(train_loader: DataLoader, test_loaders: Dict[str, DataLoader], model: PCNetEnsemble,
                T_gibbs: int, T_mh: int, mh_step_size: float, gibbs_burnin: int, log_every: int = 1,
                acc_thresh: float = 0.005, n_repeat: int = 1) -> PCNetEnsemble:
    """Update model on all datapoint once. Assess model performance on unnoised and noised data.
    Is essentially NUTS within Gibbs.

    Args:
        train_loader (DataLoader): _description_
        test_loaders (Dict[str, DataLoader]): _description_
        model (PCNet): _description_
        log_every (int, optional): _description_. Defaults to 1.
        acc_thresh (float, optional): _description_. Defaults to 0.005.

    Returns:
        PCNet: _description_
    """
    assert T_gibbs > gibbs_burnin
    models = []
    orig_train_loader = train_loader
    orig_test_loaders = test_loaders
    model_template = deepcopy(model)

    prev_a_groups = []
    for epoch in range(T_gibbs):
        # Prepare dataloaders
        train_loader = iter(orig_train_loader)
        test_loaders = {name: iter(test_loader) for name, test_loader in orig_test_loaders.items()}
        a_groups, update_results = [], []

        # Sample hidden activations via NUTS
        for i in range(0, len(train_loader)):
            data_batch = get_next_data_batch(train_loader=train_loader, test_loaders=test_loaders)
            prev_a_group = prev_a_groups[i] if len(prev_a_groups) > i else None
            a_group, update_result = sample_activations_posterior_mp(model=model,
                                                                     data_batch=data_batch,
                                                                     prev_a_group=prev_a_group,
                                                                     T_mh=T_mh,
                                                                     mh_step_size=mh_step_size)
            a_groups.append(a_group)
            update_results.append(update_result)
        prev_a_groups = a_groups

        # Sample network weights using conjugate Bayesian updates
        params_sample, params_log_prob = sample_parameters_posterior(model=model, a_groups=a_groups)
        model = deepcopy(model_template)
        model.fix_parameters(parameters=params_sample)
        models.append(model)

        # Evaluate model recall and log to wandb
        X_shape = data_batch.original_shape
        wandb_dict = {"step": epoch}
        if (epoch % log_every) == 0:
            a_group_ljs = [ur.info["model_0"]["mean_losses"][-1] for ur in update_results]
            wandb_dict["Log Joint"] = sum(a_group_ljs) + params_log_prob
            result, pred_batch = score_data_batch(data_batch=data_batch, model=model,
                                                  acc_thresh=acc_thresh, n_repeat=n_repeat,
                                                  prefix='current')
            wandb_dict.update(result)
            sample_img = plot_data_batch(data_batch=pred_batch)
            recall_info_dict = dict(model_1=pred_batch.info["train"]["repeat_1"]["hidden"])
            recall_result = UpdateResult(pcnets=None, log_weights=None, info=recall_info_dict)
            recall_energy_img = plot_update_energy(update_result=recall_result,
                                                   caption="Recall energy curve.")
            gen_img = generate_samples(model=model, X_shape=X_shape, d_batch=8,
                                       caption="Model samples via ancestral sampling.")
            log_joint_img = plot_update_energy(update_result=update_result,
                                               caption="NUTS Log Joint (last batch).")
            wandb_dict = {f"iteration/{k}": v for k, v in wandb_dict.items()}
            wandb_dict["Sample Image Denoising"] = sample_img
            wandb_dict["Generated Image"] = gen_img
            wandb_dict["NUTS Log Joint Plot"] = log_joint_img
            wandb_dict["Recall Energy Plot"] = recall_energy_img
            wandb.log(wandb_dict)
    return PCNetPosterior(ensembles=models[gibbs_burnin:])


def generate_samples_ensemble(model: PCNetPosterior, X_shape: torch.Size, d_batch: int,
                              caption: str = None) -> wandb.Image:
    Xs = [model.sample().data for _ in range(d_batch)]
    X_total = torch.cat(Xs, dim=0)
    nrow = d_batch // 4
    img = unnormalize(torchvision.utils.make_grid(X_total.reshape(-1, *X_shape[1:]), nrow=nrow))
    return wandb.Image(img, caption=caption)


def score_gibbs(train_loader: DataLoader, test_loaders: Dict[str, DataLoader],
                model: PCNetPosterior, acc_thresh: float, epoch: int, n_repeat: int
                ) -> Dict[str, float]:
    # HACK: Merge models in PCNetPosterior into a single PCNetEnsemble for recall
    pcnets = [ensemble._pcnets[0] for ensemble in model.ensembles]
    merged_model = deepcopy(model.ensembles[-1])
    merged_model._pcnets = pcnets
    merged_model._log_weights = (torch.ones(len(pcnets)) / len(pcnets)).log()
    merged_model.device = merged_model.device

    wandb_dict = {"step": epoch}
    train_loader = iter(train_loader)
    test_loaders = {name: iter(test_loader) for name, test_loader in test_loaders.items()}
    denoised_imgs = []
    for i in range(1, min(4, len(train_loader))+1):
        curr_batch = get_next_data_batch(train_loader=train_loader, test_loaders=test_loaders)
        result, pred_batch = score_data_batch(data_batch=curr_batch, model=merged_model,
                                              acc_thresh=acc_thresh, n_repeat=n_repeat,
                                              prefix=f'batch_{i}')
        denoised_img = plot_data_batch(data_batch=pred_batch)
        denoised_imgs.append(denoised_img)
        wandb_dict.update(result)

    wandb_dict = {f"epoch/{k}": v for k, v in wandb_dict.items()}
    wandb_dict["Final Generated Images"] = generate_samples_ensemble(model=model, d_batch=32,
                                                                     X_shape=(32, 3, 32, 32),
                                                                     caption="Final samples.")
    for i, denoised_img in enumerate(denoised_imgs):
        wandb_dict[f"Final Denoised Images #{i}"] = denoised_img
    wandb.log(wandb_dict)

    result_dict = dict()
    for k, v in wandb_dict.items():
        if not isinstance(v, wandb.Image):
            result_dict[k] = [v]
    return result_dict


def run_gibbs(learn_loaders: Dict[str, DataLoader], score_loaders: Dict[str, DataLoader],
              config: Tuple[PCNetEnsemble, Dict[str, Any]]) -> Dict[str, Any]:
    model, args = config
    results_dict = {'name': [args.run_name], 'seed': [args.seed], 'epoch': [1]}
    acc_thresh = args.recall_threshold
    learn_train_loader, learn_test_loaders = separate_train_test(loaders=learn_loaders)
    score_train_loader, score_test_loaders = separate_train_test(loaders=score_loaders)

    ensemble = train_gibbs(train_loader=learn_train_loader, test_loaders=learn_test_loaders,
                           model=model, log_every=args.log_every, T_gibbs=args.T_gibbs,
                           T_mh=args.T_mh, mh_step_size=args.mh_step_size,
                           gibbs_burnin=args.gibbs_burnin)
    result_dict = score_gibbs(train_loader=score_train_loader, test_loaders=score_test_loaders,
                              model=ensemble, acc_thresh=acc_thresh, epoch=1, n_repeat=1)
    results_dict.update(result_dict)
    save_config(config, f'{args.path}/{args.run_name}/latest.pt')
    return results_dict


def main_gibbs():
    args = parse_args()
    os.environ["WANDB_MODE"] = args.wandb_mode
    wandb.init(project="bayes_pcn", entity="jasonyoo", config=args)
    wandb.define_metric("iteration/step")
    wandb.define_metric("iteration/*", step_metric="iteration/step")
    wandb.define_metric("epoch/step")
    wandb.define_metric("epoch/*", step_metric="epoch/step")
    args = DotDict(wandb.config)
    if args.run_name is None:
        name = f"t{args.T_mh}l{args.n_layers}s{args.sigma_prior}{args.act_fn}_{wandb.run.id}"
        args.run_name = name
    if args.log_every is None:
        args.log_every = 1
    wandb.run.name = args.run_name
    args.path = f'runs/{args.run_group}/{args.run_name}'
    print(f"Saving models to directory: {args.path}")

    args.n_batch = args.n_data  # FIXME: Experimental Change!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    setup(args)
    learn_loaders, score_loaders, dataset_info = dataset_dispatcher(args)
    model = model_dispatcher(args=args, dataset_info=dataset_info)
    model.layer_log_prob_strat = LayerLogProbStrat.MAP
    model.layer_sample_strat = LayerSampleStrat.MAP

    # If load_path is selected, use the current args but on a saved model
    if args.load_path is None:
        config = (model, args)
    else:
        config = (load_config(args.load_path)[0], args)
    if args.cuda and torch.cuda.device_count() > 0:
        model.device = torch.device('cuda')

    result = run_gibbs(learn_loaders=learn_loaders, score_loaders=score_loaders, config=config)
    save_result(result=result, path=f"{args.path}/result.csv")


if __name__ == "__main__":
    try:
        main_gibbs()
    finally:
        wandb.finish()
