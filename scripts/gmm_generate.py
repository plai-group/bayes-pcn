import argparse
import os
from typing import Any, Dict
import torch
from pycave.bayes import GaussianMixture
import sys
sys.path.append('.')  # noqa

from bayes_pcn.dataset import dataset_dispatcher, separate_train_test
from bayes_pcn.pcnet.ensemble import PCNetEnsemble
from bayes_pcn.pcnet.util import DataBatch
from bayes_pcn.trainer import get_next_data_batch, generate_samples
from bayes_pcn.util import load_config


"""
NOTE: Use from the project directory
1. Fit a GMM using expectation maximization on top level activations
2. Generate some samples and save model
"""

N_DATA = 64  # How many datapoints to search over


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-components', type=int, required=True)
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--gmm-path', type=str, required=True)
    return parser


def get_data_batch(model_args: Dict[str, Any]) -> DataBatch:
    model_args.n_data = N_DATA
    model_args.n_batch, model_args.n_batch_score = model_args.n_data, model_args.n_data
    learn_loaders, _, _ = dataset_dispatcher(model_args)
    train_loader, test_loaders = separate_train_test(learn_loaders)
    return get_next_data_batch(train_loader=iter(train_loader),
                               test_loaders={k: iter(v) for k, v in test_loaders.items()})


def get_top_layer_activations(model: PCNetEnsemble, data_batch: DataBatch) -> torch.Tensor:
    """Get the top layer activations corresponding to all training datapoints.

    Args:
        model (PCNetEnsemble): Trained BayesPCN model
        data_batch (DataBatch): DataBatch object including the entire dataset

    Returns:
        torch.Tensor: Tensor of shape <# datapoints x top layer dim>
    """
    pred_obj = model.infer(X_obs=data_batch.train[0], fixed_indices=None, n_repeat=1)
    return pred_obj.a_group.get_acts(layer_index=-1, detach=True)


def fit_gmm(activations: torch.Tensor, n_components: int) -> GaussianMixture:
    gmm = GaussianMixture(num_components=n_components)
    return gmm.fit(activations)


def sample(model: PCNetEnsemble, gmm: GaussianMixture, X_shape: tuple, n_samples: int):
    activations = gmm.sample(num_datapoints=n_samples)
    return generate_samples(model=model, X_shape=X_shape, d_batch=n_samples, X_top=activations)


def generate_samples_gmm(n_components: int, model_path: str, gmm_path: str):
    model, args = load_config(model_path)
    data_batch = get_data_batch(args)
    if os.path.exists(gmm_path):
        gmm = GaussianMixture.load(path=gmm_path)
    else:
        activations = get_top_layer_activations(model=model, data_batch=data_batch)
        gmm = fit_gmm(activations=activations, n_components=n_components)
        gmm.save(path=gmm_path)
    gen_images = sample(model=model, gmm=gmm, X_shape=data_batch.original_shape, n_samples=32)
    gen_images.image.show()


if __name__ == "__main__":
    args = get_parser().parse_args()
    generate_samples_gmm(n_components=args.n_components,
                         model_path=args.model_path,
                         gmm_path=args.gmm_path)
