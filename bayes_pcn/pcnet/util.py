from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np
from numpy import linalg as la
import torch
import torch.distributions as dists
import torch.nn.functional as F

from typing import Any, Callable, Dict, List, NamedTuple, Tuple
from .a_group import ActivationGroup


class DataBatch(NamedTuple):
    train: Tuple[torch.Tensor, torch.Tensor]
    tests: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
    original_shape: torch.Size
    train_pred: Tuple[torch.Tensor, torch.Tensor] = None
    tests_pred: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = None
    info: Dict[str, Any] = None


class Prediction(NamedTuple):
    data: torch.Tensor
    a_group: ActivationGroup
    info: Dict[str, Any] = None


class Sample(NamedTuple):
    data: torch.Tensor
    log_joint: float
    info: Dict[str, Any] = None


class UpdateResult(NamedTuple):
    pcnets: List[Any]
    log_weights: torch.Tensor
    info: Dict[str, Any] = None


class BaseDistribution(ABC):
    @abstractmethod
    def sample(self) -> ActivationGroup:
        raise NotImplementedError()

    @abstractmethod
    def log_prob(self, a_group: ActivationGroup) -> torch.Tensor:
        raise NotImplementedError()


class Dirac(BaseDistribution):
    def __init__(self, mean_vectors: torch.Tensor, X_obs: torch.Tensor, dims: List[int]) -> None:
        self.mean_vectors = mean_vectors.clone()
        self.X_obs = X_obs.clone()
        self.dims = dims
        self.device = X_obs.device

    def sample(self) -> ActivationGroup:
        activations = torch.cat((self.X_obs, self.mean_vectors), dim=-1)
        a_group = ActivationGroup.from_concatenated(activations=activations, dims=self.dims)
        a_group.device = self.device
        return a_group

    def log_prob(self, a_group: ActivationGroup) -> torch.Tensor:
        return torch.zeros(a_group.d_batch).to(self.device)


def safe_mvn(mean_vectors: torch.Tensor, precision_matrices: torch.Tensor
             ) -> dists.MultivariateNormal:
    try:
        return dists.MultivariateNormal(loc=mean_vectors, precision_matrix=precision_matrices)
    except ValueError:
        safe_precision_matrices = []
        for precision_matrix in precision_matrices:
            if not is_PD(A=precision_matrix):
                precision_matrix = nearest_PD(A=precision_matrix)
            safe_precision_matrices.append(precision_matrix)
        safe_precision_matrices = torch.stack(safe_precision_matrices, dim=0)
        return dists.MultivariateNormal(loc=mean_vectors, precision_matrix=safe_precision_matrices)


class MVN(BaseDistribution):
    def __init__(self, mean_vectors: torch.Tensor, precision_matrices: torch.Tensor,
                 X_obs: torch.Tensor, dims: List[int]) -> None:
        self.dist = safe_mvn(mean_vectors=mean_vectors, precision_matrices=precision_matrices)
        self.X_obs = X_obs.clone()
        self.dims = dims
        self.device = X_obs.device

    def sample(self) -> ActivationGroup:
        sample = self.dist.sample()
        activations = torch.cat((self.X_obs, sample), dim=-1)
        a_group = ActivationGroup.from_concatenated(activations=activations, dims=self.dims)
        a_group.device = self.device
        return a_group

    def log_prob(self, a_group: ActivationGroup) -> torch.Tensor:
        data = a_group.get_data(flatten=True, no_obs=True)
        return self.dist.log_prob(data)


def maximize_log_joint(log_joint_fn: Callable[[ActivationGroup], torch.Tensor],
                       a_group: ActivationGroup, infer_T: int, infer_lr: float,
                       activation_optim: str, fixed_indices: torch.Tensor = None,
                       **kwargs) -> Dict[str, List[float]]:
    """Move in the space of activation vectors to minimize log joint under the model.
    a_group is modified in place. To clarify, the model is defined by its log_joint_fn.
    Depending on what part of a_group is 'clamped' or not updated by gradient descent,
    this function can be used to only update hidden layer neurons, observation neurons,
    or both. Clamps all layers on method exit.

    NOTE: Autoassociative and heteroassociative recall cannot occur in the same batch.

    Args:
        log_joint_fn (Callable[[ActivationGroup], torch.Tensor]): A function that accepts
            an ActivationGroup object and returns log probability vector of shape <d_batch>.
        a_group (ActivationGroup): Initial coordinate at the activation space.
        infer_T (int): Maximum number of gradient descent iterations.
        infer_lr (float): Gradient descent learning rate.
        activation_optim (str): Which optimizer to use for gradient descent.
        fixed_indices (torch.Tensor, optional): Boolean matrix of shape <d_batch x d_out> that
            denotes which observation neuron indices to prevent modification. Defaults to None.

    Returns:
        List[float]: A dictionary with mean, min, and max batch loss over time.
    """
    mean_losses = []
    min_losses = []
    max_losses = []
    prev_log_joint = None
    has_fixed_indices = fixed_indices_exists(fixed_indices=fixed_indices)
    if has_fixed_indices:
        a_group.clamp(obs=False, hidden=False)

    if activation_optim == 'adam':
        optim_cls = torch.optim.Adam
    elif activation_optim == 'sgd':
        optim_cls = torch.optim.SGD
    else:
        raise NotImplementedError()
    optimizer = optim_cls(a_group.data, lr=infer_lr)
    optimizer.zero_grad()

    for _ in range(infer_T):
        log_joint = log_joint_fn(a_group)
        loss = -log_joint.sum(dim=0)
        if loss.grad_fn is not None:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if has_fixed_indices:
                orig_obs = a_group.original_obs
                pred_obs = a_group.get_acts(layer_index=0, detach=True)
                corrected_obs = orig_obs * fixed_indices + pred_obs * fixed_indices.logical_not()
                a_group.set_acts(layer_index=0, value=corrected_obs)

        mean_losses.append(loss.item() / a_group.d_batch)
        min_losses.append(-log_joint.min().item())
        max_losses.append(-log_joint.max().item())
        early_stop = early_stop_infer(log_joint=log_joint, prev_log_joint=prev_log_joint)
        prev_log_joint = log_joint.detach()
        if early_stop:
            break

    mean_losses.extend([mean_losses[-1]] * (infer_T - len(mean_losses)))
    min_losses.extend([min_losses[-1]] * (infer_T - len(min_losses)))
    max_losses.extend([max_losses[-1]] * (infer_T - len(max_losses)))
    a_group.clamp(obs=True, hidden=True)
    return {'mean_losses': mean_losses, 'min_losses': min_losses, 'max_losses': max_losses}


def early_stop_infer(log_joint: torch.Tensor, prev_log_joint: torch.Tensor) -> bool:
    """Signal that inference iteration should stop if all differences between current and
    past log joint scores are less than 0.001.

    Args:
        log_joint (torch.Tensor): Current iteration log joint vector of shape <d_batch>.
        prev_log_joint (torch.Tensor): Past iteration log joint vector of shape <d_batch>.

    Returns:
        bool: Whether to stop inference iteration or not.
    """
    if prev_log_joint is None:
        return False
    return ((log_joint - prev_log_joint).abs() > 1e-3).sum() == 0


def ess_resample(objs: List[Any], log_weights: torch.Tensor, ess_thresh: float,
                 n_selected: int) -> Tuple[List[Any], torch.Tensor]:
    n_objects = len(objs)
    weights = log_weights.exp()
    ess = 1 / n_objects * (weights.sum().square() / weights.square().sum())
    if ess >= ess_thresh:
        return objs, log_weights
    log_weights = (torch.ones(n_objects) / n_objects).log().to(weights.device)
    indices = torch.multinomial(input=weights, num_samples=n_selected, replacement=True)
    objs = [deepcopy(objs[index]) for index in indices]
    return objs, log_weights


def is_PD(A: torch.Tensor):
    if A.numel() > 0:
        # If I check only one of these conditions PyTorch can complain
        pd_check = dists.constraints._PositiveDefinite().check(A)
        cholesky_check = torch.linalg.cholesky_ex(A.flip(-2, -1)).info == 0
        return pd_check and cholesky_check
    return True


def nearest_PD(A: torch.Tensor):
    """Find the nearest positive-definite matrix to input

    Source: https://stackoverflow.com/a/43244194
    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if is_PD(A3):
        return A3.to(A.dtype)

    spacing = np.spacing(la.norm(A))
    eye = np.eye(A.shape[0])
    k = 1
    while not is_PD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += eye * (-mineig * k**2 + spacing)
        k += 1

    return A3.to(A.dtype)


def fixed_indices_exists(fixed_indices: torch.Tensor) -> bool:
    return fixed_indices is not None and fixed_indices.max().item() == 1


def local_wta(X_in: torch.Tensor, block_size: int, hard: bool = True) -> torch.Tensor:
    d_batch, d_orig = X_in.shape
    assert d_orig % block_size == 0
    num_blocks = d_orig // block_size
    X_in = X_in.reshape(d_batch, num_blocks, block_size)
    wta_fn = torch.argmax if hard else F.softmax
    mask_matrix = F.one_hot(wta_fn(X_in, dim=-1), num_classes=block_size)
    return (X_in * mask_matrix).reshape(d_batch, d_orig)
