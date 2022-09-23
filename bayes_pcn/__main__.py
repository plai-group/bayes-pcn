import argparse
import os
import torch
from torch.utils.data import DataLoader
from typing import Any, Dict, Tuple
import wandb

from bayes_pcn.const import *

from .dataset import dataset_dispatcher, separate_train_test
from .util import DotDict, save_config, load_config, save_result, setup
from .pcnet import PCNetEnsemble
from .trainer import train_epoch, score_epoch


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # general configs
    parser.add_argument('--run-name', type=str, default=None,
                        help='Name of this run. If not specified set to wandb run ID.')
    parser.add_argument('--run-group', type=str, default='default', help='Group this run is in.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--load-path', type=str, default=None)
    parser.add_argument('--wandb-account', type=str, default="jasonyoo",
                        help='Weights and Biases account ID.')
    parser.add_argument('--wandb-mode', type=str, choices=['online', 'offline'], default='offline')
    parser.add_argument('--dtype', type=str, choices=['float32', 'float64'], default='float32',
                        help='Use float64 if deletion is needed for numerical stability.')
    parser.add_argument('--cuda', action='store_true', help='Use GPU if available.')

    # model configs
    parser.add_argument('--n-models', type=int, default=1)
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--h-dim', type=int, default=256)
    parser.add_argument('--sigma-prior', type=float, default=1.)
    parser.add_argument('--sigma-obs', type=float, default=0.01)
    parser.add_argument('--sigma-data', type=float, default=None)
    parser.add_argument('--beta-forget', type=float, default=0., help='between 0-1. 0 = no forget.')
    parser.add_argument('--beta-noise', type=float, default=0.1,
                        help='diffusion rate when using --layer-update-strat=noising.')
    parser.add_argument('--scale-layer', action='store_true', help='normalize layer activations.')
    parser.add_argument('--bias', action='store_true', help='Use bias alongside linear transform.')
    parser.add_argument('--n-elbo-particles', type=int, default=1,
                        help='# particles for ELBO estimation when --ensemble-proposal-strat=diag.')
    parser.add_argument('--act-fn', type=ActFn,
                        default=ActFn.RELU, choices=list(ActFn))
    parser.add_argument('--activation-init-strat', type=ActInitStrat,
                        default=ActInitStrat.RANDN, choices=list(ActInitStrat))
    parser.add_argument('--weight-init-strat', type=WeightInitStrat,
                        default=WeightInitStrat.RANDN, choices=list(WeightInitStrat))
    parser.add_argument('--layer-log-prob-strat', type=LayerLogProbStrat,
                        default=LayerLogProbStrat.P_PRED, choices=list(LayerLogProbStrat))
    parser.add_argument('--layer-sample-strat', type=LayerSampleStrat,
                        default=LayerSampleStrat.MAP, choices=list(LayerSampleStrat))
    parser.add_argument('--layer-update-strat', type=LayerUpdateStrat,
                        default=LayerUpdateStrat.BAYES, choices=list(LayerUpdateStrat))
    parser.add_argument('--ensemble-log-joint-strat', type=EnsembleLogJointStrat,
                        default=EnsembleLogJointStrat.SHARED,
                        choices=list(EnsembleLogJointStrat))
    parser.add_argument('--ensemble-proposal-strat', type=EnsembleProposalStrat,
                        default=EnsembleProposalStrat.MODE,
                        choices=list(EnsembleProposalStrat))
    parser.add_argument('--mhn-metric', type=MHNMetric,
                        default=MHNMetric.DOT, choices=list(MHNMetric))
    parser.add_argument('--kernel-type', type=Kernel, default=Kernel.RBF, choices=list(Kernel))

    # data configs
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'tinyimagenet', 'flickr30k'],
                        default='cifar10')
    parser.add_argument('--dataset-mode', type=str, default='fast',
                        choices=['fast', 'mix', 'white', 'drop', 'mask', 'all'],
                        help='Specifies test dataset configuration.')
    parser.add_argument('--n-data', type=int, default=4)
    parser.add_argument('--n-batch', type=int, default=None, help='None sets this to --n-data.')
    parser.add_argument('--n-batch-score', type=int, default=None)
    parser.add_argument('--data-start-index', type=int, default=0,
                        help='Which datapoint to start from.')

    # training configs
    parser.add_argument('--n-epoch', type=int, default=1)
    parser.add_argument('--weight-lr', type=float, default=0.0001)
    parser.add_argument('--activation-lr', type=float, default=0.01)
    parser.add_argument('--activation-optim', choices=['adam', 'sgd'], default='adam')
    parser.add_argument('--T-infer', type=int, default=500)
    parser.add_argument('--n-proposal-samples', type=int, default=1)
    parser.add_argument('--n-repeat', type=int, default=1,
                        help='how much ICM iteration to perform at inference.')
    parser.add_argument('--resample', action='store_true', help='resample if using n-models > 1.')
    parser.add_argument('--forget-every', type=int, default=None,
                        help="Apply forget with strength --beta-forget every this # of iterations.")

    # eval configs
    parser.add_argument('--recall-threshold', type=float, default=0.005)
    parser.add_argument('--log-every', type=int, default=None,
                        help="Log every this # iterations. None means log every 2^x iterations.")
    parser.add_argument('--save-every', type=int, default=None,
                        help="Save every this # iterations. None means save every 2^x iterations.")
    parser.add_argument('--log-every-epoch', type=int, default=1,
                        help="Log the entire dataset score and save the model every this # epochs. "
                             "Used if n-epoch>1.")
    return parser


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


def run(learn_loaders: Dict[str, DataLoader], score_loaders: Dict[str, DataLoader],
        config: Tuple[PCNetEnsemble, Dict[str, Any]]) -> Dict[str, Any]:
    model, args = config
    results_dict = {'name': args.run_name, 'seed': args.seed, 'epoch': [], 'n_data': args.n_data}
    acc_thresh = args.recall_threshold
    fast_mode = args.dataset_mode == 'fast'
    learn_train_loader, learn_test_loaders = separate_train_test(loaders=learn_loaders)
    score_train_loader, score_test_loaders = separate_train_test(loaders=score_loaders)

    best_scores_dict = {}
    for e in range(1, args.n_epoch+1):
        train_epoch(train_loader=learn_train_loader, test_loaders=learn_test_loaders, model=model,
                    epoch=e, n_repeat=args.n_repeat, log_every=args.log_every, fast_mode=fast_mode,
                    save_every=args.save_every, acc_thresh=acc_thresh, args=args,
                    forget_every=args.forget_every)
        if e % args.log_every_epoch == 0:
            result_dict = score_epoch(train_loader=score_train_loader,
                                      test_loaders=score_test_loaders, epoch=e, model=model,
                                      acc_thresh=acc_thresh, n_repeat=args.n_repeat)
            results_dict['epoch'].append(e)

            # Save results and models if best average scores were obtained
            for key, score in result_dict.items():
                # Only care about average stats, not min and max
                if not key.endswith("/avg"):
                    continue
                key = key[:-4]
                name = key.replace("/", "_")
                # Add score to history
                if key not in results_dict:
                    results_dict[key] = []
                results_dict[key].append(score)
                # Save model if it achieves best score
                best_score = best_scores_dict.get(key, float('inf'))
                if "_mse" in key and score < best_score:
                    best_scores_dict[key] = score
                    save_config(config, f"{args.path}/{name}.pt")
                if "_acc" in key and score > best_score:
                    best_scores_dict[key] = score
                    save_config(config, f"{args.path}/{name}.pt")
        save_config(config, f'{args.path}/latest.pt')
    return results_dict


def setup_env(args: Any) -> DotDict:
    """
    Setup wandb environment, create directories, set seeds, etc.
    """
    os.environ["WANDB_MODE"] = args.wandb_mode
    wandb.init(project="bayes_pcn", entity=args.wandb_account, config=args)
    wandb.define_metric("iteration/step")
    wandb.define_metric("iteration/*", step_metric="iteration/step")
    wandb.define_metric("epoch/step")
    wandb.define_metric("epoch/*", step_metric="epoch/step")

    args = DotDict(wandb.config)
    if args.run_name is None:
        args.run_name = f"{args.seed}h{args.h_dim}l{args.n_layers}m{args.n_models}_{wandb.run.id}"
    wandb.run.name = args.run_name
    args.path = f'runs/{args.run_group}/{args.run_name}'
    print(f"Saving models to directory: {args.path}")

    setup(args=args)
    return args


def main():
    args = get_parser().parse_args()
    args = setup_env(args=args)

    learn_loaders, score_loaders, dataset_info = dataset_dispatcher(args=args)
    model = model_dispatcher(args=args, dataset_info=dataset_info)

    # If load_path is selected, use the current args but on a saved model
    if args.load_path is None:
        config = (model, args)
    else:
        config = (load_config(args.load_path)[0], args)
    if args.cuda and torch.cuda.device_count() > 0:
        model.device = torch.device('cuda')

    result = run(learn_loaders=learn_loaders, score_loaders=score_loaders, config=config)
    save_result(result=result, path=f"{args.path}/train.csv")


if __name__ == "__main__":
    try:
        main()
    finally:
        wandb.finish()
