from abc import ABC, abstractmethod
import torch
import torch.distributions as dists
from typing import List

from .activations import ActivationGroup
from .util import is_PD, nearest_PD


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


class DiagMVN(BaseDistribution):
    def __init__(self, mean_vectors: torch.Tensor, stdev_vectors: torch.Tensor,
                 X_obs: torch.Tensor, dims: List[int]) -> None:
        self.mean_vectors = mean_vectors.clone()
        self.stdev_vectors = stdev_vectors.clone()
        self.X_obs = X_obs.clone()
        self.dims = dims
        self.device = X_obs.device

    def sample(self) -> ActivationGroup:
        h_activations = self.mean_vectors + self.stdev_vectors*torch.randn_like(self.stdev_vectors)
        activations = torch.cat((self.X_obs, h_activations), dim=-1)
        a_group = ActivationGroup.from_concatenated(activations=activations, dims=self.dims)
        a_group.device = self.device
        return a_group

    def log_prob(self, a_group: ActivationGroup) -> torch.Tensor:
        data = a_group.get_data(flatten=True, no_obs=True)
        log_prob = dists.Normal(loc=self.mean_vectors, scale=self.stdev_vectors).log_prob(data)
        return log_prob.sum(dim=-1).to(self.device)


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
