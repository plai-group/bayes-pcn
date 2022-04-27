"""
TODO
== update ==
Load trained model
Evaluate performance on all datapoints
Remove a datapoint
Evaluate performance on all datapoints
Restore a datapoint
Evaluate performance on all datapoints

== Rebind ==
Load trained model
Evaluate performance on all datapoints
Create a "hybrid image" based on first two images
Rebind first image
Evaluate performance on all datapoints
"""

import argparse
import json
import pandas as pd
import os
import torch
from torch.utils.data import DataLoader
from typing import Dict
import wandb

from bayes_pcn.dataset import dataset_dispatcher, separate_train_test
from bayes_pcn.pcnet import PCNetEnsemble
from bayes_pcn.pcnet.util import DataBatch
from bayes_pcn.trainer import get_next_data_batch, plot_data_batch,\
                              score_data_batch, generate_samples
from bayes_pcn.util import load_config, save_result, setup


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--update-index', type=int, default=0,
                        help='denotes which batch index to update (starts at 0).')
    parser.add_argument('--dataset-mode', type=str, default='fast',
                        choices=['fast', 'mix', 'white', 'drop', 'mask', 'all'],
                        help='Specifies test dataset configuration.')
    parser.add_argument('--n-repeat', type=int, default=1)
    parser.add_argument('--cuda', action='store_true', help='Use GPU if available.')
    parser.add_argument('--fast-mode', action='store_true', help='Only log epoch stats.')
    parser.add_argument('--wandb-mode', type=str, choices=['online', 'offline'], default='online')
    return parser


def get_update_batch(train_loader: DataLoader, test_loaders: Dict[str, DataLoader],
                     update_index: int) -> DataBatch:
    train_loader = iter(train_loader)
    test_loaders = {name: iter(test_loader) for name, test_loader in test_loaders.items()}
    for _ in range(update_index+1):
        data_batch = get_next_data_batch(train_loader=train_loader, test_loaders=test_loaders)
    return data_batch


def score_epoch(train_loader: DataLoader, test_loaders: Dict[str, DataLoader], model: PCNetEnsemble,
                acc_thresh: float, epoch: int, n_repeat: int, save_dir: str, omit_index: int,
                caption: str) -> Dict[str, float]:
    train_loader = iter(train_loader)
    test_loaders = {name: iter(test_loader) for name, test_loader in test_loaders.items()}

    scores = []
    for i in range(1, len(train_loader)+1):
        batch = get_next_data_batch(train_loader=train_loader, test_loaders=test_loaders)
        if i == omit_index + 1:
            continue
        X_shape = batch.original_shape
        result, pred_batch = score_data_batch(data_batch=batch, model=model,
                                              acc_thresh=acc_thresh, n_repeat=n_repeat)
        recall_img = plot_data_batch(data_batch=pred_batch)
        wandb_dict = {"step": (epoch - 1) * (len(train_loader) - 1) + i - (1 if epoch > 1 else 0),
                      **result}
        wandb_dict = {f"iteration/{caption}_{k}": v for k, v in wandb_dict.items()}
        wandb_dict["Recalled Image"] = recall_img
        wandb.log(wandb_dict)
        scores.append(result)

    score_df = pd.DataFrame(scores)
    score_df.to_csv(f"{save_dir}/iteration_scores.csv", index=False)
    wandb_dict = {"z_step": epoch}
    for key, val in score_df.mean(axis=0).iteritems():
        wandb_dict[key] = val
    wandb_dict = {f"epoch/{caption}_{k}": v for k, v in wandb_dict.items()}
    gen_img = generate_samples(model=model, X_shape=X_shape, d_batch=8,
                               caption="Model samples via ancestral sampling.")
    # wandb_dict["Generated Image"] = gen_img
    wandb.log(wandb_dict)

    # del wandb_dict["Generated Image"]
    wandb_dict = {k.split('/')[-1]: v for k, v in wandb_dict.items()}
    wandb_dict["z_step"] = [wandb_dict[f"{caption}_z_step"]]  # Need this to save with pandas
    return wandb_dict


if __name__ == "__main__":
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
    if loaded_args.cuda and torch.cuda.device_count() > 0:
        model.device = torch.device('cuda')

    wandb.init(project="bayes-pcn-update", entity="jasonyoo", config=loaded_args)
    wandb.define_metric("iteration/z_step")
    wandb.define_metric("iteration/*", step_metric="iteration/z_step")
    wandb.define_metric("epoch/z_step")
    wandb.define_metric("epoch/*", step_metric="epoch/z_step")
    setup(args=loaded_args)
    learn_loaders, score_loaders, dataset_info = dataset_dispatcher(args=loaded_args)
    train_loader, test_loaders = separate_train_test(loaders=learn_loaders)

    # Score all but update batch
    result_dict_1 = score_epoch(train_loader=train_loader, test_loaders=test_loaders,
                                epoch=1, model=model, acc_thresh=loaded_args.recall_threshold,
                                n_repeat=loaded_args.n_repeat, omit_index=args.update_index,
                                save_dir=save_dir, caption='prev')

    # Score update batch, update data, and score it again
    update_batch = get_update_batch(train_loader=train_loader, test_loaders=test_loaders,
                                    update_index=args.update_index)
    # TODO: CREATE A NEW BATCH HERE THAT COMBINES UPDATEBATCH DATA WITH THE FIRST FROG IMAGE
    # AND MASKS OUT APPROPRIATE COLUMNS BY LOOKING AT fixed_indices!!!!!!!!!!!!!!!!!!!!!!!!!
    new_batch = get_update_batch(train_loader=train_loader, test_loaders=test_loaders,
                                 update_index=args.update_index)
    result_1, pred_batch_1 = score_data_batch(data_batch=update_batch, model=model,
                                              acc_thresh=loaded_args.recall_threshold,
                                              n_repeat=args.n_repeat)
    recall_img_1 = plot_data_batch(data_batch=pred_batch_1)
    result_1 = {k: round(v, 5) for k, v in result_1.items() if 'mse' in k}
    model.update(X_obs=update_batch.train[0])
    result_2, pred_batch_2 = score_data_batch(data_batch=new_batch, model=model,
                                              acc_thresh=loaded_args.recall_threshold,
                                              n_repeat=args.n_repeat)
    recall_img_2 = plot_data_batch(data_batch=pred_batch_2)
    result_2 = {k: round(v, 5) for k, v in result_2.items() if 'mse' in k}

    # Score all but update batch
    result_dict_2 = score_epoch(train_loader=train_loader, test_loaders=test_loaders,
                                epoch=2, model=model, acc_thresh=loaded_args.recall_threshold,
                                n_repeat=loaded_args.n_repeat, omit_index=args.update_index,
                                save_dir=save_dir, caption='post')

    wandb.log({"Prev Image": recall_img_1, "Post Image": recall_img_2,
               "Prev Stats": result_1, "Post Stats": result_2})
    print("Prev update:")
    print(json.dumps(result_1, sort_keys=True, indent=4))
    print("Post update:")
    print(json.dumps(result_2, sort_keys=True, indent=4))
    recall_img_1.image.show()
    recall_img_2.image.show()
    wandb.finish()
