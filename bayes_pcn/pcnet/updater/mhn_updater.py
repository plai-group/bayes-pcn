from copy import deepcopy
import torch

from bayes_pcn.const import MHNMetric
from . import AbstractUpdater
from ..structs import *
from ..pcnet import PCNet


class MHNUpdater(AbstractUpdater):
    def __init__(self, pcnet_template: PCNet, metric: MHNMetric):
        self._pcnet_template = pcnet_template
        self._called = False
        self._metric = metric

    def __call__(self, X_obs: torch.Tensor, pcnets: List[PCNet], log_weights: torch.Tensor,
                 **kwargs) -> UpdateResult:
        """Perform a single gradient descent update on all PCNets parameters using the
        recovered mode of the (joint) log joint closest to X_obs.
        """
        assert sum([len(pcnet.layers) for pcnet in pcnets]) == len(pcnets)
        info = dict(model_0=dict(mean_losses=[], min_losses=[], max_losses=[]))

        # Do not make use of the default PCNet that is instantiated when PCNetEnsemble is created
        if not self._called:
            self._called = True
            new_pcnets = []
        else:
            new_pcnets = deepcopy(pcnets)

        # Add a new PCNet per fresh observation
        for datapoint in X_obs:
            new_pcnet = deepcopy(self._pcnet_template)
            new_pcnet.device = X_obs.device
            new_pcnet.layers[0].fix_parameters(parameters=deepcopy(datapoint))
            new_pcnets.append(new_pcnet)

        # Set component importance appropriately (should be unnormalized for MHN)
        if self._metric == MHNMetric.EUCLIDEAN:
            log_weights = torch.ones(len(new_pcnets)).to(X_obs.device)
        elif self._metric == MHNMetric.DOT:
            beta = new_pcnets[0].layers[0]._Sigma ** -0.5
            key_sq_norms = [beta*(pcnet.layers[0]._R.norm(p=2)**2).item() for pcnet in new_pcnets]
            log_weights = torch.tensor(key_sq_norms).to(X_obs.device)
        else:
            raise NotImplementedError()
        return UpdateResult(pcnets=new_pcnets, log_weights=log_weights, info=info)
