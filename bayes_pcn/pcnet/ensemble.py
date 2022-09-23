from copy import deepcopy
import random
import torch
from typing import List

from ..const import *
from .activations import *
from .pcnet import PCNet
from .structs import *
from .updater import *
from .util import *


class PCNetEnsemble:
    def __init__(self, n_models: int, n_layers: int, x_dim: int, h_dim: int, act_fn: ActFn,
                 infer_T: int, infer_lr: float, sigma_prior: float, sigma_obs: float,
                 sigma_data: float, activation_optim: str, n_proposal_samples: int,
                 activation_init_strat: str, weight_init_strat: str, layer_log_prob_strat: str,
                 layer_sample_strat: str, layer_update_strat: str, ensemble_log_joint_strat: str,
                 ensemble_proposal_strat: str, scale_layer: bool, **kwargs) -> None:
        self._n_models: int = n_models
        self._n_layers: int = n_layers
        self._x_dim: int = x_dim
        self._h_dim: int = h_dim
        self._infer_lr = infer_lr
        self._infer_T = infer_T
        self._sigma_prior: float = sigma_prior
        self._sigma_obs: float = sigma_obs
        self._sigma_data: float = sigma_data
        self._scale_layer: bool = scale_layer
        self._activation_optim: str = activation_optim
        self._log_weights: torch.Tensor = (torch.ones(self._n_models) / self._n_models).log()
        self._beta_forget = kwargs.get('beta_forget')
        self._n_elbo_particles = kwargs.get('n_elbo_particles')
        act_fn = ActFn.get_enum_from_value(act_fn)
        kernel_type = Kernel.get_enum_from_value(kwargs.get('kernel_type'))
        self.weight_init_strat = WeightInitStrat.get_enum_from_value(weight_init_strat)

        base_models_init_args = dict(weight_init_strat=self.weight_init_strat, act_fn=act_fn,
                                     n_models=n_models, bias=kwargs.get('bias'),
                                     kernel_type=kernel_type)
        if LayerUpdateStrat.get_enum_from_value(layer_update_strat) == LayerUpdateStrat.MHN:
            # HACK: Makes MHN memory efficient
            base_models_init_args['economy_mode'] = True
        self._pcnets: List[PCNet] = self._initialize_base_models(**base_models_init_args)

        self.activation_init_strat = ActInitStrat.get_enum_from_value(activation_init_strat)
        self.weight_init_strat = WeightInitStrat.get_enum_from_value(weight_init_strat)
        self.layer_log_prob_strat = LayerLogProbStrat.get_enum_from_value(layer_log_prob_strat)
        self.layer_sample_strat = LayerSampleStrat.get_enum_from_value(layer_sample_strat)
        self.layer_update_strat = LayerUpdateStrat.get_enum_from_value(layer_update_strat)
        self.ensemble_log_joint_strat = EnsembleLogJointStrat.get_enum_from_value(
                                            ensemble_log_joint_strat)
        self.ensemble_proposal_strat = EnsembleProposalStrat.get_enum_from_value(
                                            ensemble_proposal_strat)
        self.device: torch.device = torch.device('cpu')

        update_fn_args = dict(activation_init_fn=self.initialize_activation_group,
                              proposal_strat=self.ensemble_proposal_strat,
                              infer_lr=infer_lr, infer_T=infer_T,
                              n_proposal_samples=n_proposal_samples,
                              activation_optim=activation_optim,
                              n_elbo_particles=self._n_elbo_particles)
        if self.ensemble_log_joint_strat == EnsembleLogJointStrat.SHARED:
            update_fn_args['ensemble_log_joint'] = self.log_joint

        if self.layer_update_strat == LayerUpdateStrat.ML:
            assert self.layer_log_prob_strat == LayerLogProbStrat.MAP
            update_fn_args['weight_lr'] = kwargs.get('weight_lr', None)
            self._updater = MLUpdater(**update_fn_args)
        elif self.layer_update_strat in [LayerUpdateStrat.BAYES, LayerUpdateStrat.KERNEL]:
            update_fn_args['resample'] = kwargs.get('resample', False)
            if self.ensemble_proposal_strat == EnsembleProposalStrat.MODE:
                self._updater = VLBModeUpdater(**update_fn_args)
            elif self.ensemble_proposal_strat == EnsembleProposalStrat.DIAG:
                self._updater = ReparamVIUpdater(**update_fn_args)
            elif self.ensemble_proposal_strat == EnsembleProposalStrat.FULL:
                self._updater = VLBFullUpdater(**update_fn_args)
            else:
                raise NotImplementedError()
        elif self.layer_update_strat == LayerUpdateStrat.MHN:
            metric = MHNMetric.get_enum_from_value(kwargs.get('mhn_metric', MHNMetric.DOT.value))
            update_fn_args = dict(pcnet_template=deepcopy(self._pcnets[0]), metric=metric)
            self._updater = MHNUpdater(**update_fn_args)
        elif self.layer_update_strat == LayerUpdateStrat.NOISING:
            assert self.layer_log_prob_strat == LayerLogProbStrat.MAP
            update_fn_args['weight_lr'] = kwargs.get('weight_lr', None)
            update_fn_args['beta_noise'] = kwargs.get('beta_noise', None)
            self._updater = NoisingMLUpdater(**update_fn_args)
        else:
            raise NotImplementedError()

    def delete(self, X_obs: torch.Tensor, fixed_indices: torch.Tensor = None) -> None:
        """Delete an observation from the memory. NOTE: Numerically unstable at the moment.

        Args:
            X_obs (torch.Tensor): Observation to delete.
            fixed_indices (torch.Tensor, optional): Matrix of shape <d_batch x x_dim> that denotes
                which data-specific indices to prevent modification when predicting.
        """
        X_obs = X_obs.to(self.device)
        if fixed_indices_exists(fixed_indices=fixed_indices):
            fixed_indices = fixed_indices.to(self.device)
        a_group = self.initialize_activation_group(X_obs=X_obs)

        # Update hidden layer while fixing obs layer
        a_group.clamp(obs=True, hidden=False)
        maximize_log_joint(log_joint_fn=self.log_joint, a_group=a_group,
                           infer_lr=self._infer_lr, infer_T=self._infer_T,
                           fixed_indices=fixed_indices, activation_optim=self._activation_optim)
        for pcnet in self._pcnets:
            pcnet.delete_from_weights(a_group=a_group)

    def forget(self, beta_forget: float = None) -> None:
        for pcnet in self._pcnets:
            pcnet.forget(beta_forget=self._beta_forget if beta_forget is None else beta_forget)

    def infer(self, X_obs: torch.Tensor, fixed_indices: torch.Tensor = None,
              n_repeat: int = 1) -> Prediction:
        original_device = X_obs.device
        X_obs = X_obs.to(self.device)
        if fixed_indices_exists(fixed_indices=fixed_indices):
            fixed_indices = fixed_indices.to(self.device)
        a_group = self.initialize_activation_group(X_obs=X_obs)

        infer_info = dict()
        for n in range(1, n_repeat+1):
            data_acts = a_group.get_acts(layer_index=0, detach=True)
            a_group = self.initialize_activation_group(X_obs=data_acts)
            hidden_info, obs_info = None, None
            # Update hidden layer while fixing obs layer
            if self._n_layers > 1:
                a_group.clamp(obs=True, hidden=False)
                hidden_info = maximize_log_joint(log_joint_fn=self.log_joint, a_group=a_group,
                                                 infer_lr=self._infer_lr,
                                                 infer_T=self._infer_T,
                                                 fixed_indices=fixed_indices,
                                                 activation_optim=self._activation_optim,
                                                 n_particles=self._n_elbo_particles)
            # Update obs layer while fixing hidden layers
            if self._n_layers == 1 or not fixed_indices_exists(fixed_indices=fixed_indices):
                a_group.clamp(obs=False, hidden=True)
                obs_info = maximize_log_joint(log_joint_fn=self.log_joint, a_group=a_group,
                                              infer_lr=self._infer_lr,
                                              infer_T=self._infer_T,
                                              fixed_indices=fixed_indices,
                                              activation_optim=self._activation_optim)
            infer_info[f"repeat_{n}"] = {'hidden': hidden_info, 'obs': obs_info}

        X_pred = a_group.get_acts(layer_index=0, detach=True).to(original_device)
        a_norms = [a_group.get_acts(layer_index=i).norm(p=1).item() for i in range(self._n_layers)]
        infer_info["act_norms"] = a_norms
        return Prediction(data=X_pred, a_group=a_group, info=infer_info)

    def initialize_activation_group(self, X_obs: torch.Tensor) -> ActivationGroup:
        d_batch = X_obs.shape[0]
        activations = [X_obs]
        if self.activation_init_strat == ActInitStrat.FIXED:
            for _ in range(self._n_layers-1):
                activation = self._h_dim**-0.5 * torch.ones(d_batch, self._h_dim)
                activations.append(activation.to(self.device))
        elif self.activation_init_strat == ActInitStrat.RANDN:
            for _ in range(self._n_layers-1):
                activation = self._h_dim**-0.5 * torch.randn(d_batch, self._h_dim)
                activations.append(activation.to(self.device))
        elif self.activation_init_strat == ActInitStrat.RANDNPLUS:
            for _ in range(self._n_layers-1):
                activation = self._h_dim**-0.5 * torch.randn(d_batch, self._h_dim).abs()
                activations.append(activation.to(self.device))
        elif self.activation_init_strat == ActInitStrat.SAMPLE:
            raise NotImplementedError()
        else:
            raise NotImplementedError()
        use_activations_stdevs = self.ensemble_proposal_strat == EnsembleProposalStrat.DIAG
        return ActivationGroup(activations=activations, stochastic=use_activations_stdevs)

    def learn(self, X_obs: torch.Tensor) -> UpdateResult:
        X_obs = X_obs.to(self.device)
        updater_args = dict(X_obs=X_obs, pcnets=self._pcnets, log_weights=self._log_weights)
        update_result = self._updater(**updater_args)
        self._pcnets = update_result.pcnets
        self._log_weights = update_result.log_weights
        return update_result

    def log_joint(self, a_group: ActivationGroup, log_prob_strat: LayerLogProbStrat = None,
                  batch_independence: LayerLogProbStrat = False) -> LogProbResult:
        """Return the joint log probability and the layerwise log probabilities of the
        input activation group. If there are multiple pcnets, only return the layerwise
        log probabilities of the last pcnet.

        Args:
            a_group (ActivationGroup): _description_
            log_prob_strat (LayerLogProbStrat, optional): _description_. Defaults to None.

        Returns:
            LogProbResult: _description_
        """
        lps = []
        for pcnet in self._pcnets:
            lp_result = pcnet.log_joint(a_group=a_group, log_prob_strat=log_prob_strat,
                                        batch_independence=batch_independence)
            lps.append(lp_result.log_prob)
        lps = torch.stack(lps, dim=1)
        weighted_ljs = lps + self._log_weights.unsqueeze(0)
        log_joint = torch.logsumexp(weighted_ljs, dim=-1)
        return LogProbResult(log_prob=log_joint, layer_log_probs=lp_result.layer_log_probs)

    def rebind(self, X_obs: torch.Tensor, fixed_indices: torch.Tensor) -> UpdateResult:
        delete_result = self.delete(X_obs=X_obs, fixed_indices=fixed_indices)
        learn_result = self.learn(X_obs=X_obs)
        rebind_info = dict(delete=delete_result, learn=learn_result)
        return UpdateResult(pcnets=None, log_weights=None, info=rebind_info)

    def sample(self, d_batch: int = 1, a_group: ActivationGroup = None,
               X_top: torch.Tensor = None) -> Sample:
        """Select a base model based on its importance weight and sample from that model.

        Args:
            d_batch (int, optional): Number of datapoints to sample. Defaults to 1.
            a_group (ActivationGroup, optional): Only used in Gibbs sampling. Defaults to None.
            X_top (torch.Tensor, optional): Only used for sampling with GMM prior on the topmost
                                            layer of a hierarchical network. Defaults to None.

        Returns:
            (Sample): Sample object with <d_batch x x_dim> data and <d_batch> log joint tensors.
        """
        info = []
        weights = (self._log_weights - self._log_weights.logsumexp(dim=0)).exp()
        model_indices = torch.multinomial(weights, num_samples=d_batch, replacement=True)
        indices, counts = model_indices.unique(return_counts=True)
        for index, count in zip(indices.tolist(), counts.tolist()):
            if count > 0:
                X_obs = None if a_group is None else a_group.get_acts(layer_index=0, detach=True)
                pcnet = self._pcnets[index]
                sample, sample_a_group = pcnet.sample(d_batch=count, X_obs=X_obs, X_top=X_top)
                info.append((sample, sample_a_group))

        random.shuffle(info)
        data = torch.cat([sample_info[0] for sample_info in info], dim=0)
        a_group = ActivationGroup.merge(a_groups=[sample_info[1] for sample_info in info])
        log_joint = self.log_joint(a_group=a_group).log_prob
        return Sample(data=data, log_joint=log_joint)

    def _initialize_base_models(self, n_models: int, act_fn: ActFn, **kwargs) -> List[PCNet]:
        return [PCNet(n_layers=self._n_layers, x_dim=self._x_dim, h_dim=self._h_dim,
                      sigma_prior=self._sigma_prior, sigma_obs=self._sigma_obs,
                      sigma_data=self._sigma_data, act_fn=act_fn, scale_layer=self._scale_layer,
                      **kwargs)
                for _ in range(n_models)]

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value: torch.device):
        self._device = value
        for pcnet in self._pcnets:
            pcnet.device = value
        for var in vars(self):
            if isinstance(self.__dict__[var], torch.Tensor):
                self.__dict__[var] = self.__dict__[var].to(value)

    @property
    def update_fn(self):
        return self._updater

    @property
    def layer_update_strat(self):
        return self._layer_update_strat

    @layer_update_strat.setter
    def layer_update_strat(self, value: LayerUpdateStrat):
        self._layer_update_strat = value
        for pcnet in self._pcnets:
            pcnet.layer_update_strat = value

    @property
    def layer_log_prob_strat(self):
        return self._layer_log_prob_strat

    @layer_log_prob_strat.setter
    def layer_log_prob_strat(self, value: LayerLogProbStrat):
        self._layer_log_prob_strat = value
        for pcnet in self._pcnets:
            pcnet.layer_log_prob_strat = value

    @property
    def layer_sample_strat(self):
        return self._layer_sample_strat

    @layer_sample_strat.setter
    def layer_sample_strat(self, value: LayerSampleStrat):
        self._layer_sample_strat = value
        for pcnet in self._pcnets:
            pcnet.layer_sample_strat = value

    def sample_parameters(self) -> List[torch.Tensor]:
        # For each layer in the PCNet, sample from normal / matrix normal
        assert len(self._pcnets) == 1
        layer_weights = []
        pcnet = self._pcnets[0]
        for layer in pcnet.layers:
            weights = layer.sample_parameters()
            layer_weights.append(weights)
        return layer_weights

    def parameters_log_prob(self, parameters_sample: List[torch.Tensor]) -> float:
        # For each layer in the PCNet, sample from normal / matrix normal
        assert len(self._pcnets) == 1
        log_prob = 0.
        pcnet = self._pcnets[0]
        for layer in pcnet.layers:
            layer.sample_parameters
        return log_prob

    def fix_parameters(self, parameters: List[torch.Tensor]) -> None:
        assert len(self._pcnets) == 1
        pcnet = self._pcnets[0]
        for layer, weights in zip(pcnet.layers, parameters):
            layer.fix_parameters(parameters=weights)


class PCNetPosterior:
    """Container class for batched Gibbs sampling learned PCNetEnsemble.
    All PCNetEnsemble must contain a single PCNet.
    """
    def __init__(self, ensembles: List[PCNetEnsemble]) -> None:
        for ensemble in ensembles:
            assert len(ensemble._pcnets) == 1
        self.ensembles = ensembles

    def get_member(self, index: int) -> PCNetEnsemble:
        return self.ensembles[index]

    def sample(self):
        pcnet_probs = torch.ones(len(self.ensembles)) / len(self.ensembles)
        pcnet = self.get_member(torch.multinomial(pcnet_probs, num_samples=1).item())
        return pcnet.sample()

    def to(self, device: torch.device):
        for pcnet in self.ensembles:
            pcnet.device = device
