from copy import deepcopy
import io
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
from PIL import Image
import torch
from torch.utils.data import DataLoader
import torchvision
from typing import Dict, List
import wandb

from bayes_pcn.const import DataType
from bayes_pcn.pcnet.structs import DataBatch, UpdateResult
from bayes_pcn.pcnet.ensemble import PCNetEnsemble


class DotDict(dict):
    """
    https://stackoverflow.com/questions/13520421/recursive-dotdict/13520518
    a dictionary that supports dot notation
    as well as dictionary access notation
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct=None):
        if dct:
            for key, value in dct.items():
                if hasattr(value, 'keys'):
                    value = DotDict(value)
                elif isinstance(value, list):
                    value = [x for x in value]
                self[key] = value

    def get_dict(self):
        ret = {}
        for key, value in self.items():
            if hasattr(value, 'keys'):
                value = value.getDict()
            ret[key] = value
        return ret


def setup(args: DotDict):
    os.makedirs(args.path, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float32 if args.dtype == 'float32' else torch.float64)


def load_config(path):
    with open(path, 'rb') as f:
        config = pickle.load(f)
        config = (config[0], DotDict(config[1]))
        return config


def save_config(config, path):
    config = (deepcopy(config[0]), dict(config[1]))
    config[0].device = torch.device('cpu')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        os.remove(path)
    with open(path, 'wb') as f:
        pickle.dump(config, f)


def unnormalize(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.cpu().numpy()
    return np.transpose(npimg, (1, 2, 0)).clip(0, 1)


def fig2img(fig, caption: str) -> wandb.Image:
    """Convert a Matplotlib figure to a wandb Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    pil_img = Image.open(buf)
    img = wandb.Image(pil_img, caption=caption)
    pil_img.close()
    return img


def save_result(result, path, overwrite=False):
    if not overwrite and os.path.exists(path):
        prev_df = pd.read_csv(path)
        print("Will merge this previous result csv with the new result:")
        print(prev_df)
        df = pd.concat((prev_df, pd.DataFrame(result)))
    else:
        df = pd.DataFrame(result)
    print("Result csv:")
    print(df)
    df.to_csv(path, index=False)


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


def plot_data_batch(data_batch: DataBatch, data_type: str, caption: str = None) -> wandb.Image:
    """Plot the first image in the batch under various transformations.

    Args:
        data_batch (DataBatch): _description_
        caption (str, optional): _description_. Defaults to None.

    Returns:
        wandb.Image: _description_
    """
    if data_type == DataType.TOY.value:
        return plot_2d_batch(data_batch=data_batch, caption=caption)
    elif data_type == DataType.IMAGE.value:
        return plot_image_batch(data_batch=data_batch, caption=caption)
    else:
        raise NotImplementedError()


def plot_2d_batch(data_batch: DataBatch, caption: str = None) -> wandb.Image:
    labels = ['train'] + [k for k in data_batch.tests.keys()]
    queries = [data_batch.train[0]] + [v[0] for v in data_batch.tests.values()]
    preds = [data_batch.train_pred[0]] + [v[0] for v in data_batch.tests_pred.values()]
    truths = [data_batch.train[0]] * 4
    fig, ax = plt.subplots(ncols=len(labels), nrows=1, squeeze=True,
                           sharex=True, sharey=True, figsize=(12, 3))
    for i, (label, query, pred, truth) in enumerate(zip(labels, queries, preds, truths)):
        ax_obj = ax[i]
        ax_obj.set_title(f"{label}")
        ax_obj.scatter(query[:, 0], query[:, 1], marker="o", c="gray", label="query")
        ax_obj.scatter(pred[:, 0], pred[:, 1], marker="*", c="blue", label="pred")
        ax_obj.scatter(truth[:, 0], truth[:, 1], marker="x", c="red", label="truth")
    handles, labels = ax[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', ncol=len(labels))
    fig.suptitle = "Energy vs Gradient Descent Iteration"
    fig.tight_layout()
    img = fig2img(fig=fig, caption=caption)
    # plt.close(fig)
    return img


def plot_image_batch(data_batch: DataBatch, caption: str = None) -> wandb.Image:
    rows = [None] * (1 + len(data_batch.tests)) * 2
    X_shape = data_batch.original_shape
    rows[0] = data_batch.train[0][:1].reshape(X_shape[1:])
    rows[len(rows)//2] = data_batch.train_pred[0][:1].reshape(X_shape[1:])
    for i, (X, X_pred) in enumerate(zip(data_batch.tests.values(), data_batch.tests_pred.values())):
        rows[i+1] = X[0][:1].reshape(X_shape[1:])
        rows[len(rows)//2+i+1] = X_pred[0][:1].reshape(X_shape[1:])
    img = unnormalize(torchvision.utils.make_grid(rows, nrow=len(rows)//2))
    return wandb.Image(img, caption=caption)


def plot_energy_trajectory(all_mean_losses: List[List[float]], all_min_losses: List[List[float]],
                           all_max_losses: List[List[float]], caption: str = None) -> wandb.Image:
    """Given mean, min, and max losses trajectories, plot them in a single wandb.Image.
    The image will contain n_repeats plots, each of which contains time vs energy plot with
    n_grad_descent_iters datapoints.

    Args:
        all_mean_losses (List[List[float]]): Dim: <n_repeats x n_grad_descent_iters>
        all_min_losses (List[List[float]]): Dim: <n_repeats x n_grad_descent_iters>
        all_max_losses (List[List[float]]): Dim: <n_repeats x n_grad_descent_iters>
        caption (str): wandb.Image caption

    Returns:
        wandb.Image: An image with energy trajectory plots per repeat.
    """
    n_mean_losses = len(all_mean_losses)
    fig, ax = plt.subplots(ncols=n_mean_losses, nrows=1, squeeze=True, sharex=False, sharey=False)
    for i in range(n_mean_losses):
        if n_mean_losses == 1:
            ax_title, ax_obj = "first repeat", ax
        elif n_mean_losses == 2:
            ax_title, ax_obj = "first repeat" if i == 0 else "last repeat", ax[i]
            ax_obj = ax[i]
        else:
            ax_title, ax_obj = f"repeat {i}", ax[i]
        ax_obj.set_title(ax_title)
        x = [i for i in range(1, len(all_mean_losses[i]) + 1)]
        ax_obj.plot(x, all_mean_losses[i], 'b-')
        ax_obj.plot(x, all_min_losses[i], 'b--')
        ax_obj.plot(x, all_max_losses[i], 'b--')
    fig.suptitle = "Energy vs Gradient Descent Iteration"
    fig.tight_layout()
    img = fig2img(fig=fig, caption=caption)
    plt.close(fig)
    return img


def plot_test_trajectories(data_batch: DataBatch) -> Dict[str, wandb.Image]:
    # For each train and test data batches, plot the activation gradient descent energy trajectory.
    # NOTE: Only plots hidden layer energies unless no hidden layers exist.
    result = dict()
    for name, content in data_batch.info.items():
        if not ('train' in name or 'test' in name):
            continue
        repeat_keys = list(filter(lambda key: 'repeat' in key, content))
        sorted_keys = sorted(repeat_keys, key=lambda k: int(k.split('_')[-1]))
        first_repeat = content[sorted_keys[0]]['hidden']
        last_repeat = content[sorted_keys[-1]]['hidden']
        if first_repeat is None or last_repeat is None:
            first_repeat = content[sorted_keys[0]]['obs']
            last_repeat = content[sorted_keys[-1]]['obs']
        if len(sorted_keys) == 1:
            kwargs = dict(all_mean_losses=[first_repeat['mean_losses']],
                          all_min_losses=[first_repeat['min_losses']],
                          all_max_losses=[first_repeat['max_losses']])
        else:
            kwargs = dict(all_mean_losses=[first_repeat['mean_losses'], last_repeat['mean_losses']],
                          all_min_losses=[first_repeat['min_losses'], last_repeat['min_losses']],
                          all_max_losses=[first_repeat['max_losses'], last_repeat['max_losses']])
        result[name] = plot_energy_trajectory(**kwargs)
    return result


def plot_update_energy(update_result: UpdateResult, caption: str = None) -> wandb.Image:
    """Plot the min, max, and mean energy trajectory of a batch, averaged across models.

    Args:
        update_result (UpdateResult): Update result from model.learn.

    Returns:
        wandb.Image: Weights and Biases image detailing the plot.
    """
    update_info, model_names = update_result.info, []
    all_mean_losses, all_min_losses, all_max_losses = [], [], []
    for model_name, model_fit_info in update_info.items():
        model_names.append(model_name)
        all_mean_losses.append(model_fit_info.get('mean_losses', []))
        all_min_losses.append(model_fit_info.get('min_losses', []))
        all_max_losses.append(model_fit_info.get('max_losses', []))
    ncols = min(4, len(update_info))
    nrows = math.ceil(len(update_info)/ncols)
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, squeeze=False, sharex=True, sharey=True)
    for i in range(len(update_info)):
        ax_obj = ax[i // ncols, i % ncols]
        ax_obj.set_title(f"{model_names[i]}")
        x = [i for i in range(1, len(all_mean_losses[i]) + 1)]
        ax_obj.plot(x, all_mean_losses[i], 'b-')
        ax_obj.plot(x, all_min_losses[i], 'b--')
        ax_obj.plot(x, all_max_losses[i], 'b--')
    fig.tight_layout()
    img = fig2img(fig=fig, caption=caption)
    plt.close(fig)
    return img


def plot_layerwise_update_energy(update_result: UpdateResult,
                                 caption: str = None) -> List[wandb.Image]:
    """Plot the layerwise loss trajectory of a batch.

    Args:
        update_result (UpdateResult): Update result from model.learn.

    Returns:
        List[wandb.Image]: A list of Weights and Biases images detailing the plot.
    """
    update_info = update_result.info
    first_model_info = update_info[list(update_info.keys())[0]]
    layerwise_mean_losses = first_model_info.get('layerwise_mean_losses,', None)
    if layerwise_mean_losses is None:
        # If layerwise mean loss is not available, just return an empty plot
        fig, ax = plt.subplots(ncols=1, nrows=1, squeeze=False, sharex=True, sharey=True)
        fig.tight_layout()
        return [fig2img(fig=fig, caption="not available")]
    n_iter, n_layer = len(layerwise_mean_losses), len(layerwise_mean_losses[0])

    imgs = []
    for i_layer in range(n_layer):
        fig = plt.plot()
        fig, ax = plt.subplots(ncols=1, nrows=1, squeeze=False, sharex=True, sharey=True)
        ax_obj = ax[0, 0]
        ax_obj.set_title(f"Layer {i_layer}")
        x = [i for i in range(1, n_iter+1)]
        y = [layerwise_mean_losses[i_iter][i_layer].mean().item() for i_iter in range(n_iter)]
        ax_obj.plot(x, y, 'r-')
        fig.tight_layout()
        imgs.append(fig2img(fig=fig, caption=caption))
        plt.close(fig)
    return imgs


def generate_samples(model: PCNetEnsemble, X_shape: torch.Size, d_batch: int,
                     caption: str = None, X_top: torch.Tensor = None) -> wandb.Image:
    sample_obj = model.sample(d_batch=d_batch, X_top=X_top)
    X_gen, log_joint = sample_obj.data, [round(lj, 3) for lj in sample_obj.log_joint.tolist()]
    img = torchvision.utils.make_grid(X_gen.reshape(-1, *X_shape[1:]), nrow=min(d_batch, 4))
    log_joint_msg = f"Sample Weights: {log_joint}"
    caption = log_joint_msg if caption is None else caption + "\n" + log_joint_msg
    return wandb.Image(unnormalize(img), caption=caption)
