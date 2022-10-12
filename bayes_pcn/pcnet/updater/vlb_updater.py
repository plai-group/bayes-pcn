import torch
from typing import Callable, List

from bayes_pcn.const import LayerLogProbStrat
from ..dists import *
from ..structs import *
from ..util import *
from . import AbstractVLBUpdater


class VLBModeUpdater(AbstractVLBUpdater):
    def _build_proposal(self, log_joint_fn: Callable[[ActivationGroup, LayerLogProbStrat],
                        LogProbResult], a_group: ActivationGroup) -> BaseDistribution:
        # NOTE: Sampling from the mode is roughly sampling from normal with very high precision.
        mean_vectors = []
        for i in range(a_group.d_batch):
            mean_vector = a_group.get_datapoint(data_index=i, flatten=True)[:, a_group.dims[0]:]
            mean_vectors.append(mean_vector)
        mean_vectors = torch.cat(mean_vectors, dim=0)
        X_obs = a_group.get_acts(layer_index=0)
        dims = a_group.dims
        return Dirac(mean_vectors=mean_vectors, X_obs=X_obs, dims=dims)


class VLBFullUpdater(AbstractVLBUpdater):
    def _fixed_input_log_joint_fn(self, log_joint_fn: Callable[[ActivationGroup, LayerLogProbStrat],
                                  LogProbResult], X_obs: torch.Tensor, original_dims: List[int]):
        def fn(hidden: torch.Tensor):
            # fix all non parameter values and only accept parameter as input
            assert hidden.shape[0] == 1 and X_obs.shape[0] == 1
            original_device = hidden.device
            activations = torch.cat((X_obs, hidden), dim=1)
            a_group = ActivationGroup.from_concatenated(activations=activations, dims=original_dims)
            a_group.device = original_device
            return log_joint_fn(a_group).log_prob
        return fn

    def _build_proposal(self, log_joint_fn: Callable[[ActivationGroup, LayerLogProbStrat],
                        LogProbResult], a_group: ActivationGroup) -> BaseDistribution:
        # FIXME: Right now, if Hessian is not invertible we just take the diagonal (refer to MVN)
        # NOTE: We must be close to convergence for the Hessian to be invertible
        mean_vectors = []
        precision_matrices = []
        X_obs = a_group.get_acts(layer_index=0, detach=True)
        dims = a_group.dims
        for i in range(a_group.d_batch):
            mean_vector = a_group.get_datapoint(data_index=i, flatten=True)[:, a_group.dims[0]:]
            mean_vectors.append(mean_vector)
            fn = self._fixed_input_log_joint_fn(log_joint_fn=log_joint_fn, X_obs=X_obs[i:i+1],
                                                original_dims=a_group.dims).log_prob
            precision_matrix = - torch.autograd.functional.hessian(fn, mean_vector)
            precision_matrices.append(precision_matrix.squeeze())
        mean_vectors = torch.cat(mean_vectors, dim=0)
        precision_matrices = torch.stack(precision_matrices, dim=0)
        return MVN(mean_vectors=mean_vectors, precision_matrices=precision_matrices,
                   X_obs=X_obs, dims=dims)
