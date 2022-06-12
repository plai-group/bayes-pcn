import argparse
import json
import numpy as np
import os
import torch

from bayes_pcn.dataset import dataset_dispatcher, separate_train_test
from bayes_pcn.pcnet.util import DataBatch
from bayes_pcn.trainer import get_next_data_batch, plot_data_batch,\
                              score_data_batch
from bayes_pcn.util import load_config


"""
TLDR: Used for debugging or quick visualization.

Load a trained model and print its performance on a specific train/test batch.
Test dataloaders don't have to be the same as one used while training.

Usage: `python examine.py --path=...`


Goal
- Left side of the image has the training images flipping by 3 frames per second
for up to 60 images
- Right side of the image has 60 randomly corrupted images in an 6x10 grid
- Every time an image is flipped past, the corrupted image on the right side is
replaced with the recalled image


# TODO
1. Randomly select some high level corruption mechanism and save BayesPCN recall result.
2. Have a script that separate training, corrupted, and recalled images into different images.
3. Have a script that stitches together the output image iteration by iteration
- per iteration:
    - mix and match images needed for RHS panel, convert them to vectors
    - join them into a single RHS image using the torchvision method
    - 2x magnify the image being currently written and place it on the LHS of the image
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


"""
def visualize_index(path, index, n_repeat, dataset_mode, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    model, args = load_config(path)
    args.dataset_mode = dataset_mode
    learn_loaders, _, _ = dataset_dispatcher(args)
    train_loader, test_loaders = separate_train_test(learn_loaders)
    train_loader, test_loaders = iter(train_loader), {k: iter(v) for k, v in test_loaders.items()}

    test_loader_name = np.random.choice([k for k in test_loaders.keys()])
    test_loaders = {test_loader_name: test_loaders[test_loader_name]}
    for _ in range(index+1):
        data_batch = get_next_data_batch(train_loader=train_loader, test_loaders=test_loaders)
    result, pred_batch = score_data_batch(data_batch=data_batch, model=model,
                                          acc_thresh=0.005, n_repeat=n_repeat)
    result = {k: round(v, 5) for k, v in result.items()}
    batch_img = plot_data_batch(data_batch=pred_batch)
    print(json.dumps(result, sort_keys=True, indent=4))
    batch_img.image.save(os.path.join(save_dir, f"{index}.png"), "PNG")
"""


def extract_batch(pred_batch, data_index):
    def extract_category(cat_tuple):
        first = cat_tuple[0][data_index:data_index+1, :]
        second = None if cat_tuple[1] is None else cat_tuple[1][data_index:data_index+1, :]
        return (first, second)
    train = extract_category(pred_batch.train)
    tests = {key: extract_category(val) for key, val in pred_batch.tests.items()}
    train_pred = extract_category(pred_batch.train_pred)
    tests_pred = {key: extract_category(val) for key, val in pred_batch.tests_pred.items()}
    new_batch = DataBatch(train=train, tests=tests, train_pred=train_pred, tests_pred=tests_pred,
                          original_shape=pred_batch.original_shape)
    return new_batch


# def get_test_merged_batch(data_batch, index):
#     # Create a DataBatch with the test images with different noises combined
#     data, fixed_indices = [], []
#     for i in range(0, index+1):
#         np.random.seed(i)
#         noise_name = np.random.choice([k for k in data_batch.tests.keys()])
#         curr_data, curr_fixed_indices = data_batch.tests[noise_name]
#         data.append(curr_data[i])
#         fixed_indices.append(curr_fixed_indices[i])
#     tests = dict(custom=(torch.stack(data, dim=0), torch.stack(fixed_indices, dim=0)))
#     return DataBatch(train=data_batch.train, tests=tests, train_pred=data_batch.train_pred,
#                      tests_pred=data_batch.tests_pred, original_shape=data_batch.original_shape)


# def visualize_index(path, index, n_repeat, dataset_mode, save_dir):
#     os.makedirs(save_dir, exist_ok=True)
#     model, args = load_config(path)
#     args.dataset_mode = dataset_mode
#     args.n_data, args.n_batch, args.n_batch_score = index+1, index+1, index+1
#     learn_loaders, _, _ = dataset_dispatcher(args)
#     train_loader, test_loaders = separate_train_test(learn_loaders)
#     train_loader, test_loaders = iter(train_loader), {k: iter(v) for k, v in test_loaders.items()}
#     data_batch = get_next_data_batch(train_loader=train_loader, test_loaders=test_loaders)
#     data_batch = get_test_merged_batch(data_batch, index)
#     result, pred_batch = score_data_batch(data_batch=data_batch, model=model,
#                                           acc_thresh=0.005, n_repeat=n_repeat)
#     result = {k: round(v, 5) for k, v in result.items()}
#     print(json.dumps(result, sort_keys=True, indent=4))
#     for i in range(args.n_batch):
#         os.makedirs(os.path.join(save_dir, f"{index}"), exist_ok=True)
#         curr_batch = extract_batch(pred_batch=pred_batch, data_index=i)
#         batch_img = plot_data_batch(data_batch=curr_batch)
#         batch_img.image.save(os.path.join(save_dir, f"{index}/{i}.png"), "PNG")


def get_test_merged_batch(data_batch, index):
    # Create a DataBatch with the test images with different noises combined
    auto_indices, hetero_indices = [], []
    auto_train, hetero_train = [], []
    auto_tests, hetero_tests = [], []
    auto_fixed_indices, hetero_fixed_indices = [], []
    train_pred, tests_pred = data_batch.train_pred, data_batch.tests_pred
    original_shape = data_batch.original_shape

    for i in range(0, index+1):
        np.random.seed(i)
        noise_name = np.random.choice([k for k in data_batch.tests.keys()])
        curr_data, curr_fixed_indices = data_batch.tests[noise_name]
        if 'white' in noise_name:
            auto_indices.append(i)
            auto_train.append(data_batch.train[0][i])
            auto_tests.append(curr_data[i])
            auto_fixed_indices.append(curr_fixed_indices[i])
        else:
            hetero_indices.append(i)
            hetero_train.append(data_batch.train[0][i])
            hetero_tests.append(curr_data[i])
            hetero_fixed_indices.append(curr_fixed_indices[i])
    auto_train = (torch.stack(auto_train, dim=0) if len(auto_train) > 0 else None, None)
    hetero_train = (torch.stack(hetero_train, dim=0) if len(hetero_train) > 0 else None, None)
    auto_tests = dict(custom=(torch.stack(auto_tests, dim=0),
                              torch.stack(auto_fixed_indices, dim=0))
                      if len(auto_tests) > 0 else (None, None))
    hetero_tests = dict(custom=(torch.stack(hetero_tests, dim=0),
                                torch.stack(hetero_fixed_indices, dim=0))
                        if len(hetero_tests) > 0 else (None, None))
    auto_batch = DataBatch(train=auto_train, tests=auto_tests, train_pred=train_pred,
                           tests_pred=tests_pred, original_shape=original_shape)
    hetero_batch = DataBatch(train=hetero_train, tests=hetero_tests, train_pred=train_pred,
                             tests_pred=tests_pred, original_shape=original_shape)
    return (auto_batch, auto_indices, hetero_batch, hetero_indices)


def visualize_index(path, index, n_repeat, dataset_mode, save_dir):
    os.makedirs(os.path.join(save_dir, f"{index}"), exist_ok=True)
    model, args = load_config(path)
    args.dataset_mode = dataset_mode
    args.n_data, args.n_batch, args.n_batch_score = index+1, index+1, index+1
    learn_loaders, _, _ = dataset_dispatcher(args)
    train_loader, test_loaders = separate_train_test(learn_loaders)
    train_loader, test_loaders = iter(train_loader), {k: iter(v) for k, v in test_loaders.items()}
    data_batch = get_next_data_batch(train_loader=train_loader, test_loaders=test_loaders)
    auto_batch, a_indices, hetero_batch, h_indices = get_test_merged_batch(data_batch, index)
    if len(a_indices) > 0:
        auto_result, auto_pred_batch = score_data_batch(data_batch=auto_batch, model=model,
                                                        acc_thresh=0.005, n_repeat=n_repeat)
        auto_result = {k: round(v, 5) for k, v in auto_result.items()}
        print(json.dumps(auto_result, sort_keys=True, indent=4))
        for i, a_index in enumerate(a_indices):
            curr_batch = extract_batch(pred_batch=auto_pred_batch, data_index=i)
            batch_img = plot_data_batch(data_batch=curr_batch)
            batch_img.image.save(os.path.join(save_dir, f"{index}/{a_index}.png"), "PNG")
    if len(h_indices) > 0:
        hetero_result, hetero_pred_batch = score_data_batch(data_batch=hetero_batch, model=model,
                                                            acc_thresh=0.005, n_repeat=n_repeat)
        hetero_result = {k: round(v, 5) for k, v in hetero_result.items()}
        print(json.dumps(hetero_result, sort_keys=True, indent=4))
        for i, h_index in enumerate(h_indices):
            curr_batch = extract_batch(pred_batch=hetero_pred_batch, data_index=i)
            batch_img = plot_data_batch(data_batch=curr_batch)
            batch_img.image.save(os.path.join(save_dir, f"{index}/{h_index}.png"), "PNG")


if __name__ == "__main__":
    args = get_parser().parse_args()
    np.random.seed(args.index)
    torch.manual_seed(0)
    visualize_index(path=args.model_path, index=args.index, n_repeat=args.n_repeat,
                    dataset_mode=args.dataset_mode, save_dir=args.save_dir+"/first")
