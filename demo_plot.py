import argparse
import json
import os

from bayes_pcn.dataset import dataset_dispatcher, separate_train_test
from bayes_pcn.trainer import get_next_data_batch, plot_data_batch,\
                              score_data_batch, generate_samples
from bayes_pcn.util import load_config


"""
TLDR: Used for debugging or quick visualization.

Load a trained model and print its performance on a specific train/test batch.
Test dataloaders don't have to be the same as one used while training.

Usage: `python examine.py --path=...`
"""


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--save-dir', type=str, default="demo_dir")
    parser.add_argument('--index', type=int, default=0,
                        help='denotes which batch index to look at.')
    parser.add_argument('--n-repeat', type=int, default=1)
    parser.add_argument('--dataset-mode', type=str, default='all',
                        choices=['fast', 'mix', 'white', 'drop', 'mask', 'all'],
                        help='Specifies test dataset configuration.')
    return parser


def visualize_index(path, index, n_repeat, dataset_mode, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    model, args = load_config(path)
    args.dataset_mode = dataset_mode
    learn_loaders, _, _ = dataset_dispatcher(args)
    train_loader, test_loaders = separate_train_test(learn_loaders)
    train_loader, test_loaders = iter(train_loader), {k: iter(v) for k, v in test_loaders.items()}
    for _ in range(index+1):
        data_batch = get_next_data_batch(train_loader=train_loader, test_loaders=test_loaders)
    result, pred_batch = score_data_batch(data_batch=data_batch, model=model,
                                          acc_thresh=0.005, n_repeat=n_repeat)
    result = {k: round(v, 5) for k, v in result.items()}
    batch_img = plot_data_batch(data_batch=pred_batch)
    print(json.dumps(result, sort_keys=True, indent=4))
    batch_img.image.save(os.path.join(save_dir, os.path.basename(path)[:-3]+".png"), "PNG")


def visualize_unseen(path, n_repeat, dataset_mode, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    model, args = load_config(path)
    args.dataset_mode = dataset_mode
    index = args.n_data
    args.n_data, args.n_batch, args.n_batch_score = args.n_data + 1, 1, 1
    learn_loaders, _, _ = dataset_dispatcher(args)
    train_loader, test_loaders = separate_train_test(learn_loaders)
    train_loader, test_loaders = iter(train_loader), {k: iter(v) for k, v in test_loaders.items()}
    for _ in range(index+1):
        data_batch = get_next_data_batch(train_loader=train_loader, test_loaders=test_loaders)
    result, pred_batch = score_data_batch(data_batch=data_batch, model=model,
                                          acc_thresh=0.005, n_repeat=n_repeat)
    result = {k: round(v, 5) for k, v in result.items()}
    batch_img = plot_data_batch(data_batch=pred_batch)
    print(json.dumps(result, sort_keys=True, indent=4))
    batch_img.image.save(os.path.join(save_dir, os.path.basename(path)[:-3]+".png"), "PNG")


if __name__ == "__main__":
    args = get_parser().parse_args()
    visualize_index(path=args.model_path, index=args.index, n_repeat=args.n_repeat,
                    dataset_mode=args.dataset_mode, save_dir=args.save_dir+"/index")
    visualize_unseen(path=args.model_path, n_repeat=args.n_repeat,
                     dataset_mode=args.dataset_mode, save_dir=args.save_dir+"/unseen")
