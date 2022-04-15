from abc import ABC, abstractmethod
from copy import deepcopy
import torch
import torch.distributions as dists

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
            try:
                torch.linalg.cholesky(precision_matrix)
            except RuntimeError:
                precision_matrix = precision_matrix.diag().max(torch.tensor(1.)).diag()
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
        prev_log_joint = log_joint
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


def assert_positive_definite(matrix: torch.Tensor):
    torch.linalg.cholesky(matrix)


def fixed_indices_exists(fixed_indices: torch.Tensor) -> bool:
    return fixed_indices is not None and fixed_indices.max().item() == 1
