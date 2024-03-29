import argparse
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from typing import Dict
import wandb

from bayes_pcn.const import LayerUpdateStrat
from bayes_pcn.dataset import dataset_dispatcher, separate_train_test
from bayes_pcn.pcnet import PCNetEnsemble
from bayes_pcn.trainer import score_data_batch
from bayes_pcn.util import *


"""
TLDR: Used for evaluating trained model performance.

Load a trained model and log its performance on al train/test batches to wandb.
Test dataloaders don't have to be the same as one used while training.

Usage: `python score.py --model-path=...`
"""


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb-account', type=str, default="jasonyoo",
                        help='Weights and Biases account ID.')
    parser.add_argument('--model-path', type=str, default='runs/debug/default/latest.pt')
    parser.add_argument('--dataset-mode', type=str, default='all',
                        choices=['fast', 'mix', 'mix_high', 'white', 'drop', 'mask', 'all'],
                        help='Specifies test dataset configuration.')
    parser.add_argument('--n-data', type=int, default=None)
    parser.add_argument('--n-batch', type=int, default=None)
    parser.add_argument('--n-repeat', type=int, default=1)
    parser.add_argument('--acc-thresh', type=float, default=0.01)
    parser.add_argument('--cuda', action='store_true', help='Use GPU if available.')
    parser.add_argument('--fast-mode', action='store_true', help='Only log epoch stats.')
    parser.add_argument('--wandb-mode', type=str, choices=['online', 'offline'], default='online')
    return parser


def score_epoch(train_loader: DataLoader, test_loaders: Dict[str, DataLoader], model: PCNetEnsemble,
                acc_thresh: float, epoch: int, n_repeat: int, fast_mode: bool, save_dir: str
                ) -> Dict[str, float]:
    train_loader = iter(train_loader)
    test_loaders = {name: iter(test_loader) for name, test_loader in test_loaders.items()}

    scores = []
    individual_scores = []
    for i in range(1, len(train_loader)+1):
        batch = get_next_data_batch(train_loader=train_loader, test_loaders=test_loaders)
        X_shape = batch.original_shape
        result, pred_batch = score_data_batch(data_batch=batch, model=model,
                                              acc_thresh=acc_thresh, n_repeat=n_repeat)
        individual_scores.append(pred_batch.info["all_mses"])
        if not fast_mode:
            recall_img = plot_data_batch(data_batch=pred_batch)
            wandb_dict = {"z_step": i, **result}
            wandb_dict = {f"iteration/{k}": v for k, v in wandb_dict.items()}
            wandb_dict["Recalled Image"] = recall_img

            # Log test dataset free energy trajectories
            trajectory_img_dict = plot_test_trajectories(data_batch=pred_batch)
            wandb_dict.update({f"Energy/{k}": v for k, v in trajectory_img_dict.items()})
            wandb.log(wandb_dict)
        scores.append(result)

    score_df = pd.DataFrame(scores)
    score_df.to_csv(f"{save_dir}/iteration_scores.csv", index=False)
    tmp_df = pd.DataFrame(individual_scores)
    individual_scores = {col: torch.cat(tmp_df[col].tolist(), dim=0) for col in tmp_df.columns}
    individual_score_df = pd.DataFrame(individual_scores)
    individual_score_df.to_csv(f"{save_dir}/iteration_individual_scores.csv", index=False)
    wandb_dict = {"z_step": epoch}
    for key, val in score_df.mean(axis=0).iteritems():
        wandb_dict[key] = val
    wandb_dict = {f"epoch/{k}": v for k, v in wandb_dict.items()}
    gen_img = generate_samples(model=model, X_shape=X_shape, d_batch=8,
                               caption="Model samples via ancestral sampling.")
    wandb_dict["Generated Image"] = gen_img
    wandb.log(wandb_dict)

    del wandb_dict["Generated Image"]
    del wandb_dict["epoch/z_step"]
    wandb_dict = {k.split('/')[-1]: [v] for k, v in wandb_dict.items()}
    return wandb_dict


def main():
    args = get_parser().parse_args()
    os.environ["WANDB_MODE"] = args.wandb_mode
    model, loaded_args = load_config(args.model_path)
    save_dir = os.path.dirname(args.model_path)

    # Override some of the loaded arguments
    try:
        loaded_args.n_data = int(os.path.basename(args.model_path).split('_')[-1][:-3])
    except Exception as e:
        print("Not overloading number of data to evaluate.")
    loaded_args.cuda = args.cuda
    loaded_args.dataset_mode = args.dataset_mode
    loaded_args.n_repeat = args.n_repeat
    loaded_args.acc_thresh = args.acc_thresh
    loaded_n_data = loaded_args.n_data
    loaded_n_batch = loaded_args.n_batch
    loaded_n_batch_score = loaded_args.n_batch_score

    if args.n_data is not None:
        loaded_args.n_data = args.n_data
        loaded_args.n_batch = min(loaded_args.n_batch, args.n_data)
    if args.n_batch is not None:
        loaded_args.n_batch = args.n_batch
        loaded_args.n_batch_score = args.n_batch
    if loaded_args.cuda and torch.cuda.device_count() > 0:
        model.device = torch.device('cuda')
    # HACK: For ML runs, the only allowed data size is the one it was trained on
    if loaded_args.layer_update_strat == LayerUpdateStrat.ML.value:
        loaded_args.n_data = loaded_n_data
        loaded_args.n_batch = loaded_n_batch
        loaded_args.n_batch_score = loaded_n_batch_score

    wandb.init(project="bayes-pcn-score", entity=args.wandb_account, config=loaded_args)
    wandb.define_metric("iteration/z_step")
    wandb.define_metric("iteration/*", step_metric="iteration/z_step")
    wandb.define_metric("epoch/z_step")
    wandb.define_metric("epoch/*", step_metric="epoch/z_step")
    wandb.run.name = f"{loaded_args.run_name}_d{loaded_args.n_data}"
    setup(args=loaded_args)
    _, score_loaders, _ = dataset_dispatcher(args=loaded_args)
    loaders = score_loaders  # if args.fast_mode else learn_loaders
    train_loader, test_loaders = separate_train_test(loaders=loaders)

    result_dict = score_epoch(train_loader=train_loader, test_loaders=test_loaders,
                              epoch=1, model=model, acc_thresh=loaded_args.acc_thresh,
                              n_repeat=loaded_args.n_repeat, fast_mode=args.fast_mode,
                              save_dir=save_dir)
    save_dict = {**loaded_args, **result_dict}
    save_result(result=save_dict, path=f"{save_dir}/score.csv", overwrite=False)


if __name__ == "__main__":
    try:
        main()
    finally:
        wandb.finish()
