from copy import deepcopy
import torch
from typing import Callable

from bayes_pcn.const import EnsembleProposalStrat, LayerLogProbStrat
from . import AbstractUpdater
from ..activations import maximize_log_joint
from ..structs import *
from ..pcnet import PCNet


class MLUpdater(AbstractUpdater):
    def __init__(self, activation_init_fn: Callable[[torch.Tensor], ActivationGroup],
                 infer_lr: float, infer_T: int, proposal_strat: EnsembleProposalStrat,
                 n_proposal_samples: int, activation_optim: str,
                 ensemble_log_joint: Callable[[ActivationGroup, LayerLogProbStrat],
                 torch.Tensor] = None, **kwargs) -> None:
        super().__init__(activation_init_fn, infer_lr, infer_T, proposal_strat, n_proposal_samples,
                         activation_optim, ensemble_log_joint, **kwargs)
        self._weight_lr = kwargs.get('weight_lr', None)

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
            a_group.clamp(obs=True, hidden=False)
            fit_info = maximize_log_joint(log_joint_fn=pcnet.log_joint, a_group=a_group,
                                          infer_T=self._infer_T, infer_lr=self._infer_lr,
                                          activation_optim=self._activation_optim, train_mode=True)
            pcnet.update_weights(a_group=a_group, lr=kwargs.get('weight_lr', self._weight_lr))
            info[f"model_{i}"] = fit_info
        return UpdateResult(pcnets=new_pcnets, log_weights=log_weights, info=info)
