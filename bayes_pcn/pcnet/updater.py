from abc import ABC, abstractmethod
from copy import deepcopy
import torch
from typing import Callable, List, Tuple

from bayes_pcn.const import EnsembleProposalStrat, LayerLogProbStrat, MHNMetric

from .pcnet import PCNet
from .util import *


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
                                          activation_optim=self._activation_optim,
                                          n_particles=self._n_elbo_particles)
            if a_group.stochastic:
                # HACK: Sample from the variational distribution
                for i in range(1, len(a_group.data)):
                    noise = a_group.stdevs[i-1] * torch.randn_like(a_group.stdevs[i-1])
                    new_layer_acts = a_group.get_acts(layer_index=i, detach=True) + noise
                    a_group.set_acts(layer_index=i, value=new_layer_acts)
            pcnet.update_weights(a_group=a_group, lr=kwargs.get('weight_lr', self._weight_lr))
            info[f"model_{i}"] = fit_info
        return UpdateResult(pcnets=new_pcnets, log_weights=log_weights, info=info)


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
