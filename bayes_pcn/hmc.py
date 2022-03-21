import pandas as pd
from pyro.infer import MCMC, NUTS
from pyro.infer.mcmc.util import summary
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Tuple
import wandb
from .main import parse_args, create_batch_dict, plot_batch_dict,\
                  generate_samples, save_config, load_config, save_result, setup
from .model import PCNet, PCNetEnsemble
from .util import *


def sample_activations_posterior(model: PCNet, X_batch: torch.Tensor) -> torch.Tensor:
    """Sample all layer activations for current data batch. Activations include
    the observed layer activations.

    Args:
        model_fn (PCNet): _description_
        X (torch.Tensor): _description_

    Returns:
        _type_: _description_
    """
    X_batch = X_batch.to(model.device)
    nuts_kernel = NUTS(model=model.sample_data)  # , step_size=0.0001, adapt_step_size=False)
    mcmc = MCMC(
        nuts_kernel,
        num_samples=1,
        warmup_steps=model.T_infer,
    )
    mcmc.run(X_batch)
    samples = mcmc.get_samples()
    hidden_kvs = sorted(list(samples.items()), key=lambda n: int(n[0].split('_')[-1]))
    # TODO: Inspect log joint
    return [X_batch] + [v.squeeze(0) for _, v in hidden_kvs]


def sample_weight_posterior(model: PCNet, all_activations: List[List[torch.Tensor]]
                            ) -> List[torch.Tensor]:
    """Update a new model on all data from all_activations then sample model parameters

    Args:
        model (PCNet): _description_
        R (torch.Tensor): _description_
        U (torch.Tensor): _description_
        all_activations (List[List[torch.Tensor]]): Outer dim represents loop over
            minibatches, inner dim represents loop over layer activations , and tensor
            contains minibatch activations at a particular layer

    Returns:
        List[torch.Tensor]: A list of layer weights
    """
    for activations in all_activations:
        model._update(activations=activations)
    return model.sample_weights()


def train_hmc(train_loader: DataLoader, test_loaders: Dict[str, DataLoader], model: PCNet,
              n_epoch: int, log_every: int = 1, plot_every: int = 1) -> PCNetEnsemble:
    """Update model on all datapoint once. Assess model performance on unnoised and noised data.
    Is essentially NUTS within Gibbs.

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

    pcnets = []
    orig_train_loader = train_loader
    orig_test_loaders = test_loaders
    for epoch in range(n_epoch):
        all_activations = []
        train_loader = iter(orig_train_loader)
        test_loaders = {name: iter(test_loader) for name, test_loader in orig_test_loaders.items()}

        for i in range(1, len(train_loader)+1):
            # Prepare train and test data batches for learning and evaluation
            curr_batch, X_shape = create_batch_dict(train_loader=train_loader,
                                                    test_loaders=test_loaders)
            activations = sample_activations_posterior(model=model, X_batch=curr_batch['train'][0])
            all_activations.append(activations)

        # Update model
        W = sample_weight_posterior(model=model, all_activations=all_activations)
        model.reset()
        model.fix_weights(layer_weights=W)

        pcnet = deepcopy(model)
        pcnets.append(pcnet)

        # TODO: Inspect log joint
        wandb_dict = {"step": epoch}
        if (i % log_every) == 0:
            min_weight = min([W_layer.min() for W_layer in W])
            max_weight = max([W_layer.max() for W_layer in W])
            wandb_dict["max_weight_norm"] = max(max_weight, -min_weight)
            min_activation = min([min([a_layer.min() for a_layer in batch])
                                  for batch in all_activations])
            max_activation = max([max([a_layer.max() for a_layer in batch])
                                  for batch in all_activations])
            wandb_dict["max_activation_norm"] = max(max_activation, -min_activation)
        if (i % plot_every) == 0:
            # Plot first images of first and current data batches
            curr_img = plot_batch_dict(batch_dict=curr_batch, model=pcnet, X_shape=X_shape)
            gen_img = generate_samples(model=pcnet, X_shape=X_shape, d_batch=6,
                                       caption="Model samples via ancestral & inference sampling.")
        if (i % log_every) == 0:
            # Log to wandb
            wandb_dict = {f"iteration/{k}": v for k, v in wandb_dict.items()}
            wandb_dict["Sample Image Denoising"] = curr_img
            wandb_dict["Generated Image"] = gen_img
            wandb.log(wandb_dict)
    return PCNetEnsemble(pcnets=pcnets)


def generate_samples_ensemble(model: PCNetEnsemble, X_shape: torch.Size, d_batch: int,
                              caption: str = None,) -> wandb.Image:
    Xs = [model.sample() for _ in range(d_batch)]
    X_total = torch.cat([Xs], dim=0)
    nrow = d_batch // 4
    img = unnormalize(torchvision.utils.make_grid(X_total.reshape(-1, *X_shape[1:]), nrow=nrow))
    return wandb.Image(img, caption=caption)


def score_hmc(train_loader: DataLoader, test_loaders: Dict[str, DataLoader], model: PCNetEnsemble,
              acc_thresh: float, epoch: int) -> Dict[str, float]:
    wandb_dict = {"step": epoch}
    wandb_dict["Final Samples"] = generate_samples_ensemble(model=model, X_shape=(16, 3, 32, 32),
                                                            d_batch=16, caption="Final samples.")
    wandb.log({f"epoch/{k}": v for k, v in wandb_dict.items()})
    return wandb_dict


def run_hmc(train_loader: DataLoader, test_loaders: Dict[str, DataLoader],
            config: Tuple[PCNet, Dict[str, Any]]):
    model, args = config
    results_dict = {'name': args.run_name, 'seed': args.seed, 'epoch': []}
    acc_thresh = args.recall_threshold

    ensemble = train_hmc(train_loader=train_loader, test_loaders=test_loaders, model=model,
                         n_epoch=args.n_epoch, log_every=args.log_every, plot_every=args.plot_every)
    model = ensemble.get_member(index=0)
    result_dict = score_hmc(train_loader=train_loader, test_loaders=test_loaders,
                            model=ensemble, acc_thresh=acc_thresh, epoch=1)
    results_dict.update(result_dict)
    save_config(config, f'{args.path}/latest.pt')
    return results_dict


def main_hmc():
    args = parse_args()
    os.environ["WANDB_MODE"] = args.wandb_mode
    wandb.init(project="bayes_pcn", entity="jasonyoo", config=args)
    wandb.define_metric("iteration/step")
    wandb.define_metric("iteration/*", step_metric="iteration/step")
    wandb.define_metric("epoch/step")
    wandb.define_metric("epoch/*", step_metric="epoch/step")
    args = DotDict(wandb.config)
    args.run_name = wandb.run.name

    setup(args)
    train_loader, test_loaders, dataset_info = dataset_dispatcher(args)
    pcnet = PCNet(mode=args.pcn_mode, n_layers=args.n_layers, d_out=dataset_info.get('x_dim'),
                  d_h=args.h_dim, weight_lr=args.weight_lr, activation_lr=args.activation_lr,
                  T_infer=args.T_infer, n_repeat=args.n_repeat)

    # If load_path is selected, use the current args but on a saved model
    if args.load_path is None:
        config = (pcnet, args)
    else:
        config = (load_config(args.load_path)[0], args)
    # if torch.cuda.device_count() > 0:
    #     config[0].to(torch.device('cuda'))

    result = run_hmc(train_loader=train_loader, test_loaders=test_loaders, config=config)
    save_result(result=result, path=f"{args.path}/result.csv")


if __name__ == "__main__":
    main_hmc()
