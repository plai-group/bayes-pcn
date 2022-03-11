"""
AM via PCN Paper

2 layer net: T (time) in {12, 16, 24, 32}
Multi layer net: T in {32, 48, 72}

Data size in {100, 250, 500, 750, 1000}
hidden layer width in {256, 512, 1024, 2048}
gamma (learning rate for activations) in {1, 0.5, 0.1, 0.05, 0.01}
alpha (learning rate for weights) in {0.0001, 0.00005}

Model Detail: PyTorch initialization + ReLU activations
Corrupt Image Retrieval
- Iteration function F 30 times, using T in {100, 250, 500}

Weird Good Behaviour:
- h: 256, data: 100, gamma: 0.00001, T: 10000
"""

import argparse
from copy import deepcopy
from typing import Any, Dict, List, Tuple
import pandas as pd
import pickle
import os
import torch
from torch.utils.data import DataLoader
import torchvision
import wandb

from dataset import dataset_dispatcher, imshow
from model import PCNet
from util import DotDict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='runs/default')
    parser.add_argument('--dataset', type=str, choices=['cifar10'], default='cifar10')
    parser.add_argument('--dataset-mode', type=str, default='fast',
                        choices=['fast', 'mix', 'white', 'drop', 'mask'],
                        help='Specifies test dataset configuration.')
    parser.add_argument('--pcn-mode', type=str, choices=['ml', 'bayes'], default='ml')
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--n-data', type=int, default=100)
    parser.add_argument('--h-dim', type=int, default=256)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n-batch', type=int, default=100)
    parser.add_argument('--n-epoch', type=int, default=10000)
    parser.add_argument('--weight-lr', type=float, default=0.0001)
    parser.add_argument('--activation-lr', type=float, default=0.01)
    parser.add_argument('--T-infer', type=int, default=500)
    parser.add_argument('--F-infer', type=int, default=1)
    parser.add_argument('--recall-threshold', type=float, default=0.005)
    parser.add_argument('--load-path', type=str, default=None)
    parser.add_argument('--log-every', type=int, default=1, help="Log every this # of iterations")
    parser.add_argument('--plot-every', type=int, default=1, help="Plot every this # of iterations")
    parser.add_argument('--save-every', type=int, default=1, help="Save every this # of iterations")
    parser.add_argument('--wandb-mode', type=str, choices=['online', 'offline'], default='online')
    args = parser.parse_args()
    return args


def setup(args):
    os.makedirs(args.path, exist_ok=True)
    torch.manual_seed(args.seed)


def save_config(config, path):
    config = (deepcopy(config[0]), dict(config[1]))
    config[0].to(torch.device('cpu'))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        os.remove(path)
    with open(path, 'wb') as f:
        pickle.dump(config, f)


def load_config(path):
    with open(path, 'rb') as f:
        config = pickle.load(f)
        config = (config[0], DotDict(config[1]))
        return config


def save_result(result, path):
    if os.path.exists(path):
        prev_df = pd.read_csv(path)
        print("Will merge this previous result csv with the new result:")
        print(prev_df)
        df = pd.concat((prev_df, pd.DataFrame(result)))
    else:
        df = pd.DataFrame(result)
    print("Result csv:")
    print(df)
    df.to_csv(path, index=False)


def compare_cifar(path, index):
    """Given the model path and index of the image in the first batch, plot images.

    Args:
        path (_type_): _description_
        index (_type_): _description_
    """
    def mse(truth: torch.Tensor, pred: torch.Tensor) -> float:
        return (truth - pred).norm(p=2) ** 2

    model, args = load_config(path)
    args.noise = 'mask'
    train, tests, _ = dataset_dispatcher(args)
    batch = next(iter(train))[0]
    train_image = batch[index]
    train_recon = model.infer(train_image.flatten().unsqueeze(0), n_repeat=1).reshape(3, 32, 32)
    inputs, recons = [train_image], [train_recon]
    mse_pairs = [(0., round(mse(train_image, train_recon), 4))]
    for test in tests:
        batch, fixed_indices = next(iter(test))
        test_image, fixed_indices = batch[index], fixed_indices[index]
        test_recon = model.infer(test_image.flatten().unsqueeze(0), n_repeat=100,
                                 fixed_indices=fixed_indices.flatten().unsqueeze(0)
                                 ).reshape(3, 32, 32)
        inputs.append(test_image)
        recons.append(test_recon)
        mse_pairs.append((round(mse(train_image, test_image), 4),
                          round(mse(train_image, test_recon), 4)))

    rows = inputs + recons
    imshow(torchvision.utils.make_grid(rows, nrow=len(rows)//2))
    print(f"Input MSE vs Reconstruction MSE In Order: {mse_pairs}")


def generate(path, d_batch):
    from dataset import imshow
    import torchvision
    model, _ = load_config(path)
    X_top_down, _ = model.generate(d_batch=d_batch, noise=0.05)
    X_gen, _ = model.generate_iterative(d_batch=d_batch, noise=0.05)
    X_total = torch.cat((X_top_down, X_gen), dim=0)
    imshow(torchvision.utils.make_grid(X_total.reshape(-1, 3, 32, 32)))


def score(model: PCNet, X: torch.Tensor, X_truth: torch.Tensor, acc_thresh: float,
          n_repeat: int = 1, fixed_indices: torch.Tensor = None) -> Tuple[float, float]:
    """_summary_

    Args:
        model (PCNet): Predictive coding network.
        X (torch.Tensor): Input matrix of shape <d_batch x x_dim>.
        X_truth (torch.Tensor): Ground truth matrix of shape <d_batch x x_dim>.
        acc_thresh (float): MSE has to be lower than this for a datapoint to be considered
            correctly retrieved.
        n_repeat (int): Number of times to repeat model._infer.
        fixed_indices (torch.Tensor, Optional): Matrix of shape <d_batch x x_dim> that denotes
            which data-specific indices to prevent modification when predicting.

    Returns:
        Tuple[float, float]: Mean MSE and accuracy of the model's prediction on current batch.
    """
    X_pred = model.infer(X_obs=X, fixed_indices=fixed_indices, n_repeat=n_repeat)
    mse = ((X_truth - X_pred)**2).sum(dim=-1)
    acc = len(mse[mse < acc_thresh]) / len(mse)
    return mse.mean().item(), acc


def create_batch_dict(train_loader: DataLoader, test_loaders: Dict[str, DataLoader]
                      ) -> Tuple[Dict[str, Tuple[torch.Tensor, torch.Tensor]], torch.Size]:
    """Given train and test loaders, return batch dict where keys are loader names and
    values are reshaped data batches along with original data shape. Assumes dataloaders
    are already iterators. NOTE: Do not call next() on dataloader outside this method!

    Args:
        train_loader (DataLoader): Single train dataloader.
        test_loaders (Dict[str, DataLoader]): A dictionary of test dataloaders.

    Returns:
        Tuple[Dict[str, Tuple[torch.Tensor, torch.Tensor]], torch.Size]: Dictionary with
            loader name and reshaped loader contents + original data shape.
    """
    X, _ = next(train_loader)
    X_shape = X.shape  # original data shape
    X = X.reshape(X.shape[0], -1)
    curr_batch = {'train': (X, None)}
    for name, test_loader in test_loaders.items():
        X_test, fixed_indices = next(test_loader)
        n_batch = X_test.shape[0]
        curr_batch[name] = (X_test.reshape(n_batch, -1), fixed_indices.reshape(n_batch, -1))
    return curr_batch, X_shape


def score_batch_dict(batch_dict: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
                     model: PCNet, acc_thresh: float, suffix: str = None,
                     n_repeat: int = 1) -> Dict[str, float]:
    """_summary_

    Args:
        batch_dict (Dict[str, Tuple[torch.Tensor, torch.Tensor]]): _description_
        model (PCNet): _description_
        acc_thresh (float): _description_
        n_repeat (int): Number of times to repeat model._infer.
        prefix (str, optional): _description_. Defaults to None.
        suffix (str, optional): _description_. Defaults to None.

    Returns:
        Dict[str, float]: _description_
    """
    result = {}
    X_truth = batch_dict["train"][0]
    suffix = "/" + suffix if suffix is not None else ""
    for name, (X, fixed_indices) in batch_dict.items():
        mse_train, acc_train = score(model=model, X=X, X_truth=X_truth, acc_thresh=acc_thresh,
                                     n_repeat=n_repeat, fixed_indices=fixed_indices)
        result[f"{name}{suffix}_mse"] = mse_train
        result[f"{name}{suffix}_acc"] = acc_train
    return result


def plot_batch_dict(batch_dict: Dict[str, Tuple[torch.Tensor, torch.Tensor]], model: PCNet,
                    X_shape: torch.Size, caption: str = None, n_repeat: int = 1) -> wandb.Image:
    """Plot the first image in the batch under various transformations

    Args:
        batch_dict (Dict[str, Tuple[torch.Tensor, torch.Tensor]]): _description_
        model (PCNet): _description_
        X_shape (torch.Size): _description_
        caption (str, optional): _description_. Defaults to None.
        n_repeat (int, optional): _description_. Defaults to 1.

    Returns:
        wandb.Image: _description_
    """
    row1, row2 = [], []
    for X, fixed_indices in batch_dict.values():
        X, fixed_indices = X[:1], (fixed_indices[:1] if fixed_indices is not None else None)
        row1.append(X.reshape(X_shape[1:]))
        X_pred = model.infer(X_obs=X, fixed_indices=fixed_indices, n_repeat=n_repeat)
        row2.append(X_pred.reshape(X_shape[1:]))
    rows = row1 + row2
    return wandb.Image(torchvision.utils.make_grid(rows, nrow=len(rows)//2), caption=caption)


def train_epoch(train_loader: DataLoader, test_loaders: Dict[str, DataLoader], model: PCNet,
                epoch: int, log_every: int = 1, plot_every: int = 1, acc_thresh: float = 0.005
                ) -> PCNet:
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
        curr_batch, X_shape = create_batch_dict(train_loader=train_loader,
                                                test_loaders=test_loaders)
        X_train = curr_batch["train"][0]
        if i == 1:
            first_batch = curr_batch

        wandb_dict = {"step": (epoch - 1) * len(train_loader) + i}
        init_img, unseen_img, curr_img = None, None, None
        # Evaluate model performance on the current data batch before update and optionally plot
        if (i % log_every) == 0:
            wandb_dict.update(score_batch_dict(batch_dict=curr_batch, model=model,
                                               acc_thresh=acc_thresh, suffix='unseen'))
        if (i % plot_every) == 0:
            unseen_img = plot_batch_dict(batch_dict=curr_batch, model=model, X_shape=X_shape)

        # Update model
        model.learn(X_obs=X_train)

        if (i % log_every) == 0:
            # Evaluate model performance on the first and current data batches after update
            wandb_dict.update(score_batch_dict(batch_dict=first_batch, model=model,
                                               acc_thresh=acc_thresh, suffix='initial'))
            wandb_dict.update(score_batch_dict(batch_dict=curr_batch, model=model,
                                               acc_thresh=acc_thresh, suffix='current'))

        if (i % plot_every) == 0:
            # Plot first images of first and current data batches
            init_img = plot_batch_dict(batch_dict=first_batch, model=model, X_shape=X_shape)
            curr_img = plot_batch_dict(batch_dict=curr_batch, model=model, X_shape=X_shape)
        if (i % log_every) == 0:
            # Log to wandb
            wandb_dict = {f"iteration/{k}": v for k, v in wandb_dict.items()}
            wandb_dict["Initial Image"] = init_img
            wandb_dict["Unseen Image"] = unseen_img
            wandb_dict["Current Image"] = curr_img
            wandb.log(wandb_dict)
    return model


def score_epoch(train_loader: DataLoader, test_loaders: Dict[str, DataLoader], model: PCNet,
                acc_thresh: float, epoch: int) -> Dict[str, float]:
    train_loader = iter(train_loader)
    test_loaders = {name: iter(test_loader) for name, test_loader in test_loaders.items()}

    scores = []
    for _ in range(1, len(train_loader)+1):
        curr_batch, _ = create_batch_dict(train_loader=train_loader, test_loaders=test_loaders)
        scores.append(score_batch_dict(batch_dict=curr_batch, model=model, acc_thresh=acc_thresh))
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


def run(train_loader: DataLoader, test_loaders: Dict[str, DataLoader], config: Dict[str, Any]):
    model, args = config
    results_dict = {'epoch': []}
    best_scores_dict = {}
    acc_thresh = args.recall_threshold

    for e in range(1, args.n_epoch+1):
        train_epoch(train_loader=train_loader, test_loaders=test_loaders, model=model, epoch=e,
                    log_every=args.log_every, plot_every=args.plot_every, acc_thresh=acc_thresh)
        result_dict = score_epoch(train_loader=train_loader, test_loaders=test_loaders,
                                  model=model, acc_thresh=acc_thresh, epoch=e)
        save_config(config, f'{args.path}/latest.pt')

        results_dict['epoch'].append(e)
        for key, score in result_dict.items():
            # Only care about average stats, not min and max
            if not key.endswith("/avg"):
                continue
            key = key[:-4]
            # Add score to history
            if key not in results_dict:
                results_dict[key] = []
            results_dict[key].append(score)
            # Save model if it achieves best score
            best_score = best_scores_dict.get(key, float('inf'))
            if "_mse" in key and score < best_score:
                best_scores_dict[key] = score
                name = key.replace("/", "_")
                save_config(config, f"{args.path}/{name}.pt")
            if "_acc" in key and score > best_score:
                best_scores_dict[key] = score
                name = key.replace("/", "_")
                save_config(config, f"{args.path}/{name}.pt")
    return results_dict


def main():
    args = parse_args()
    os.environ["WANDB_MODE"] = args.wandb_mode
    wandb.init(project="bayes_pcn", entity="jasonyoo", config=args)
    wandb.define_metric("iteration/step")
    wandb.define_metric("iteration/*", step_metric="iteration/step")
    wandb.define_metric("epoch/step")
    wandb.define_metric("epoch/*", step_metric="epoch/step")
    args = DotDict(wandb.config)

    setup(args)
    train_loader, test_loaders, dataset_info = dataset_dispatcher(args)
    pcnet = PCNet(mode=args.pcn_mode, n_layers=args.n_layers, d_out=dataset_info.get('x_dim'),
                  d_h=args.h_dim, weight_lr=args.weight_lr, activation_lr=args.activation_lr,
                  T_infer=args.T_infer)

    # If load_path is selected, use the current args but on a saved model
    if args.load_path is None:
        config = (pcnet, args)
    else:
        config = (load_config(args.load_path)[0], args)
    if torch.cuda.device_count() > 0:
        config[0].to(torch.device('cuda'))

    result = run(train_loader=train_loader, test_loaders=test_loaders, config=config)
    save_result(result=result, path=f"{args.path}/result.csv")


if __name__ == "__main__":
    main()
