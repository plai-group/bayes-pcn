from abc import ABC, abstractmethod
from copy import deepcopy
import torch
from typing import Any, Callable, Dict, List, Tuple

from ..activations import ActivationGroup, maximize_log_joint
from ..dists import BaseDistribution
from ..structs import *
from bayes_pcn.const import EnsembleProposalStrat, LayerLogProbStrat
from bayes_pcn.pcnet.pcnet import PCNet


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


class AbstractUpdater(ABC):
    def __init__(self, activation_init_fn: Callable[[torch.Tensor], ActivationGroup],
                 infer_lr: float, infer_T: int, proposal_strat: EnsembleProposalStrat,
                 n_proposal_samples: int, activation_optim: str,
                 ensemble_log_joint: Callable[[ActivationGroup, LayerLogProbStrat],
                 torch.Tensor] = None, n_elbo_particles: int = 1, **kwargs) -> None:
        self._activation_init_fn: Callable[[torch.Tensor], ActivationGroup] = activation_init_fn
        self._activation_optim = activation_optim
        self._infer_lr = infer_lr
        self._infer_T = infer_T
        self._proposal_strat = proposal_strat
        self._n_proposal_samples = n_proposal_samples
        self._shared_log_joint = ensemble_log_joint is not None
        self._ensemble_log_joint = ensemble_log_joint
        self._n_elbo_particles = n_elbo_particles

    @abstractmethod
    def __call__(self, X_obs: torch.Tensor, pcnets: List[PCNet], log_weights: torch.Tensor,
                 **kwargs) -> UpdateResult:
        raise NotImplementedError()

    @property
    def infer_lr(self):
        return self._infer_lr

    @property
    def infer_T(self):
        return self._infer_T


class AbstractVLBUpdater(AbstractUpdater):
    def __init__(self, activation_init_fn: Callable[[torch.Tensor], ActivationGroup],
                 infer_lr: float, infer_T: int, proposal_strat: EnsembleProposalStrat,
                 n_proposal_samples: int, activation_optim: str,
                 ensemble_log_joint: Callable[[ActivationGroup, LayerLogProbStrat],
                 torch.Tensor] = None, **kwargs) -> None:
        super().__init__(activation_init_fn, infer_lr, infer_T, proposal_strat, n_proposal_samples,
                         activation_optim, ensemble_log_joint, **kwargs)
        self._resample = kwargs.get('resample', False)

    def __call__(self, X_obs: torch.Tensor, pcnets: List[PCNet], log_weights: torch.Tensor,
                 **kwargs) -> UpdateResult:
        """Perform a conjugate Bayes update on all PCNet parameters using either the recovered mode
        of the weighted or per-model log joint closest to X_obs. Log weights are unchanged since we
        are taking a single 'sample' from the same variational distribution.
        """
        # Fit and sample proposal distribution(s).
        n_samples = self._n_proposal_samples
        n_layers = len(pcnets[0].layers)
        n_models = len(pcnets)
        fit_sample_args = dict(X_obs=X_obs, n_samples=n_samples,
                               n_layers=n_layers, n_models=n_models)
        if self._shared_log_joint:
            fit_sample_args['log_joint_fn'] = self._ensemble_log_joint
            fit_sample_args['sample_weight_log_joint_fns'] = [pcnet.log_joint for pcnet in pcnets]
            output = self._fit_sample_proposal(**fit_sample_args)
            a_groups_all = output[0] * len(pcnets)
            log_sample_weights = output[1]  # <(model x samples) x batch>
            stats = [output[2]] * len(pcnets)
        else:
            a_groups_all, log_sample_weights, stats = [], [], []
            for pcnet in pcnets:
                fit_sample_args['log_joint_fn'] = pcnet.log_joint
                output = self._fit_sample_proposal(**fit_sample_args)
                a_groups_all.extend(output[0])
                log_sample_weights.append(output[1])
                stats.append(output[2])
            log_sample_weights = torch.cat(log_sample_weights, dim=0)  # <(model x samples) x batch>

        # Update PCNets and log weights.
        new_pcnets = []
        new_log_weights = []
        fit_info = {}
        for i, pcnet in enumerate(pcnets):
            log_weight_model = log_weights[i]
            a_groups_model = a_groups_all[i:i+n_samples]
            log_sample_weights_model = log_sample_weights[i:i+n_samples]
            stat_model = stats[i]
            j = i * n_samples
            for a_group, log_sample_weight in zip(a_groups_model, log_sample_weights_model):
                new_pcnet = deepcopy(pcnet)
                new_pcnet.update_weights(a_group=a_group)
                new_pcnets.append(new_pcnet)
                new_log_weight = log_weight_model + log_sample_weight.sum()
                new_log_weights.append(new_log_weight)
                fit_info[f"model_{j}"] = stat_model
                j = j + 1
        # HACK: renormalize twice to correct numerical error
        new_log_weights = torch.tensor(new_log_weights).to(log_weights.device)
        new_log_weights = new_log_weights - torch.logsumexp(new_log_weights, dim=-1)
        new_log_weights = new_log_weights - torch.logsumexp(new_log_weights, dim=-1)

        # Optionally perform ESS resampling step here, modifying both pcnets and log_weights
        if self._resample:
            new_pcnets, new_log_weights = ess_resample(objs=new_pcnets, log_weights=new_log_weights,
                                                       ess_thresh=0.5, n_selected=len(pcnets))
        return UpdateResult(pcnets=new_pcnets, log_weights=new_log_weights, info=fit_info)

    @abstractmethod
    def _build_proposal(self, log_joint_fn: Callable[[ActivationGroup, LayerLogProbStrat],
                        LogProbResult], a_group: ActivationGroup) -> BaseDistribution:
        raise NotImplementedError()

    def _fit_sample_proposal(self, X_obs: torch.Tensor, n_samples: int, n_layers: int,
                             n_models: int, log_joint_fn:
                             Callable[[ActivationGroup, LayerLogProbStrat], LogProbResult],
                             **kwargs) -> Tuple[List[ActivationGroup], torch.Tensor, List[Any]]:
        if n_layers == 1:
            samples = [self._activation_init_fn(X_obs=X_obs) for _ in range(n_samples)]
            log_sample_weight = torch.zeros(n_models * n_samples, X_obs.shape[0])
            fit_info = dict()
        else:
            proposal, fit_info = self._fit_proposal(log_joint_fn=log_joint_fn, X_obs=X_obs)
            samples = [proposal.sample() for _ in range(n_samples)]
            log_sample_weight = self._log_sample_weight_unnorm(log_joint_fn=log_joint_fn,
                                                               proposal_fn=proposal.log_prob,
                                                               a_groups=samples, **kwargs)
        return samples, log_sample_weight, fit_info

    def _fit_proposal(self, log_joint_fn: Callable[[ActivationGroup, LayerLogProbStrat],
                      LogProbResult], X_obs: torch.Tensor
                      ) -> Tuple[BaseDistribution, Dict[str, Any]]:
        a_group = self._activation_init_fn(X_obs=X_obs)
        # Update hidden layer while fixing obs layer
        a_group.clamp(obs=True, hidden=False)
        fit_info = maximize_log_joint(log_joint_fn=log_joint_fn, a_group=a_group,
                                      infer_T=self._infer_T, infer_lr=self._infer_lr,
                                      activation_optim=self._activation_optim,
                                      n_particles=self._n_elbo_particles)
        proposal = self._build_proposal(log_joint_fn=log_joint_fn, a_group=a_group)
        return proposal, {'layer_dims': a_group.dims, **fit_info}

    def _log_sample_weight_unnorm(self, a_groups: List[ActivationGroup], log_joint_fn:
                                  Callable[[ActivationGroup, LayerLogProbStrat], LogProbResult],
                                  proposal_fn: Callable[[ActivationGroup], torch.Tensor],
                                  **kwargs) -> torch.Tensor:
        """Returns an unnormalized sample log weight matrix of shape <d_batch x n_samples>.

        Args:
            a_groups (List[ActivationGroup]): _description_
            proposal (BaseDistribution): _description_
            log_joint_fn (Callable[[ActivationGroup], torch.Tensor]): _description_

        Returns:
            torch.Tensor: _description_
        """
        log_joint_fns = kwargs.get('sample_weight_log_joint_fns', [log_joint_fn])
        with torch.no_grad():
            all_log_weights = []
            for a_group in a_groups:
                log_weights = []
                for log_joint_fn in log_joint_fns:
                    log_p = log_joint_fn(a_group, LayerLogProbStrat.P_PRED).log_prob
                    log_q = proposal_fn(a_group)
                    model_log_weights = log_p - log_q
                    log_weights.append(model_log_weights)
                if len(log_weights) == 1:
                    log_weights = log_weights[0].unsqueeze(0)
                else:
                    log_weights = torch.stack(log_weights, dim=0)
                all_log_weights.append(log_weights)  # <n_models x d_batch>
            if len(all_log_weights) == 1:
                log_weights = all_log_weights[0]
            else:
                log_weights = torch.cat(all_log_weights, dim=0)
            return log_weights  # <(n_models x n_samples) x d_batch>
