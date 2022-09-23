from copy import deepcopy
import torch
from typing import Callable

from bayes_pcn.const import EnsembleProposalStrat, LayerLogProbStrat
from . import AbstractUpdater
from ..structs import *
from ..pcnet import PCNet


def set_acts_to_noise(a_group: ActivationGroup, beta: float) -> ActivationGroup:
    last_layer = a_group.get_acts(layer_index=0, detach=True)
    for i_layer in reversed(range(1, len(a_group.dims))):
        curr_layer = ((1-beta)**0.5)*last_layer + beta*torch.randn_like(last_layer)
        assert curr_layer.shape == last_layer.shape
        a_group.set_acts(layer_index=i_layer, value=curr_layer)
        last_layer = curr_layer
    a_group.clamp(hidden=True)
    a_group.clamp(obs=True)
    # print(a_group.get_acts(layer_index=9).max())
    return a_group


class NoisingMLUpdater(AbstractUpdater):
    def __init__(self, activation_init_fn: Callable[[torch.Tensor], ActivationGroup],
                 infer_lr: float, infer_T: int, proposal_strat: EnsembleProposalStrat,
                 n_proposal_samples: int, activation_optim: str,
                 ensemble_log_joint: Callable[[ActivationGroup, LayerLogProbStrat],
                 torch.Tensor] = None, **kwargs) -> None:
        super().__init__(activation_init_fn, infer_lr, infer_T, proposal_strat, n_proposal_samples,
                         activation_optim, ensemble_log_joint, **kwargs)
        self._weight_lr = kwargs.get('weight_lr', None)
        self._beta = kwargs.get('beta_noise', 0.05)

    def __call__(self, X_obs: torch.Tensor, pcnets: List[PCNet], log_weights: torch.Tensor,
                 **kwargs) -> UpdateResult:
        """Perform a single gradient descent update on all PCNets parameters using the
        recovered mode of the (joint) log joint closest to X_obs.
        """
        assert len(pcnets) == 1
        info = {}
        new_pcnets = deepcopy(pcnets)
        for i, pcnet in enumerate(new_pcnets):
            a_group = self._activation_init_fn(X_obs=X_obs)
            # Update hidden layer while fixing obs layer
            a_group = set_acts_to_noise(a_group=a_group, beta=self._beta)
            pcnet.update_weights(a_group=a_group, lr=kwargs.get('weight_lr', self._weight_lr))
            info[f"model_{i}"] = {'mean_losses': [], 'min_losses': [],
                                  'max_losses': [], 'layerwise_mean_losses': []}
        return UpdateResult(pcnets=new_pcnets, log_weights=log_weights, info=info)
