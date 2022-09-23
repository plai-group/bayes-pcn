import torch
from typing import Callable

from bayes_pcn.const import LayerLogProbStrat
from . import AbstractVLBUpdater
from ..activations import ActivationGroup
from ..dists import BaseDistribution, DiagMVN
from ..structs import LogProbResult


class ReparamVIUpdater(AbstractVLBUpdater):
    def _build_proposal(self, log_joint_fn: Callable[[ActivationGroup, LayerLogProbStrat],
                        LogProbResult], a_group: ActivationGroup) -> BaseDistribution:
        mean_vectors = []
        stdev_vectors = []
        X_obs = a_group.get_acts(layer_index=0, detach=True)
        dims = a_group.dims
        for i in range(a_group.d_batch):
            mean_vector = a_group.get_datapoint(data_index=i, flatten=True)[:, a_group.dims[0]:]
            mean_vectors.append(mean_vector)
            stdev_vector = a_group.get_datapoint_stdevs(data_index=i, flatten=True)
            stdev_vectors.append(stdev_vector)
        mean_vectors = torch.cat(mean_vectors, dim=0)
        stdev_vectors = torch.cat(stdev_vectors, dim=0)
        return DiagMVN(mean_vectors=mean_vectors, stdev_vectors=stdev_vectors,
                       X_obs=X_obs, dims=dims)
