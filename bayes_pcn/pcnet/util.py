from abc import ABC, abstractmethod
from copy import deepcopy
import torch
import torch.distributions as dists
import torch.nn as nn
from typing import Any, Callable, Dict, List, NamedTuple, Tuple, Union


class ActivationGroup:
    def __init__(self, activations: List[torch.Tensor], from_concatenated: bool = False) -> None:
        """Contains all layer-wise activations of PCNets within PCNetEnsemble. Makes things
        easier to work with PyTorch optimizers. Does not modify bottom-most activations.
        NOTE: This method modifies contents of activations in place.
        NOTE: Layer activations are not parameters if from_concatenated is not true to
            have the method work with torch.autograd.hessian.

        Args:
            activations (List[torch.Tensor]): A list of PCNetEnsemble activations.
                The list enumerates over layers PCNetEnsemble layers in ascending order.
                The element tensor shapes should be <d_batch x d_layer>.
        """
        self._device: torch.device = torch.device('cpu')
        self._original_obs: torch.Tensor = None if from_concatenated else deepcopy(activations[0])
        layer_acts = activations[0] if from_concatenated else nn.Parameter(activations[0])
        self._data: List[torch.Tensor] = [layer_acts]
        self._dims: List[int] = [activations[0].shape[-1]]
        for layer_acts in activations[1:]:
            layer_acts = layer_acts if from_concatenated else nn.Parameter(layer_acts)
            self._data.append(layer_acts)
            self._dims.append(layer_acts.shape[-1])
        self._d_batch: int = layer_acts.shape[0]

    def get_acts(self, layer_index: int, detach: bool = True) -> torch.Tensor:
        if layer_index >= len(self._data):
            return None
        result = self._data[layer_index]
        return result.detach() if detach else result

    def set_acts(self, layer_index: int, value: torch.Tensor):
        if layer_index >= len(self._data):
            return None
        with torch.no_grad():
            self._data[layer_index] += value - self._data[layer_index]

    def get_data(self, flatten: bool = False, no_obs: bool = False
                 ) -> Union[List[torch.Tensor], torch.Tensor]:
        result = self._data
        if no_obs:
            result = result[1:]
        return torch.cat(result, dim=-1) if flatten else result

    def get_datapoint(self, data_index: int, flatten: bool = False
                      ) -> Union[List[torch.Tensor], torch.Tensor]:
        result = [layer_acts[data_index:data_index+1] for layer_acts in self._data]
        return torch.cat(result, dim=-1) if flatten else result

    def clamp(self, obs: bool = None, hidden: bool = None) -> None:
        assert (obs is not None) or (hidden is not None)
        if obs is not None:
            self._data[0].requires_grad = not obs
            # self._data[0] = self._data[0].detach() if obs else self._data[0]
        if hidden is not None:
            for i in range(1, len(self._data)):
                self._data[i].requires_grad = not hidden
                # self._data[i] = self._data[i].detach() if hidden else self._data[i]

    @classmethod
    def from_concatenated(cls, activations: torch.Tensor, dims: List[int]) -> 'ActivationGroup':
        """Given activation matrix, return an ActivationGroup object that chunks them according
        to dimensions in dims. Preserves gradients if they exist in activations.

        Args:
            activations (torch.Tensor): Activation matrix of shape <d_batch x |network neurons|>
            dims (List[int]): A list of activation dimensions from bottom to top layers.

        Returns:
            ActivationGroup: ActivationGroup object created from activations.
        """
        separated = []
        curr_loc = 0
        for dim in dims:
            layer_acts = activations[:, curr_loc:curr_loc+dim]
            separated.append(layer_acts)
            curr_loc = curr_loc + dim
        return cls(activations=separated, from_concatenated=True)

    @classmethod
    def merge(cls, a_groups: List['ActivationGroup']) -> 'ActivationGroup':
        """Merge a list of ActivationGroup into a single ActivationGroup by stacking them
        along the batch dimension.

        Args:
            a_groups (List[ActivationGroup]): A list of valid activation groups.

        Returns:
            ActivationGroup: A new activation group that combines elements of a_groups.
        """
        activations = []
        for i in range(len(a_groups[0].dims)):
            layer_acts = torch.cat([a_group.get_acts(layer_index=i, detach=True)
                                    for a_group in a_groups], dim=0)
            activations.append(layer_acts)
        return ActivationGroup(activations)

    @property
    def data(self):
        return self._data

    @property
    def dims(self):
        return self._dims

    @property
    def original_obs(self):
        return self._original_obs

    @property
    def d_batch(self):
        return self._d_batch

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device: torch.device):
        self._device = device
        if self._original_obs is not None:
            self._original_obs = self._original_obs.to(device)
        for i in range(len(self._data)):
            self._data[i] = self._data[i].to(device)


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
