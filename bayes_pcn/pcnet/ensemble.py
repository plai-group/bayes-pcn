import random
import torch
from typing import List

from bayes_pcn.const import *
from .pcnet import PCNet
from .updater import *
from .util import *


class PCNetEnsemble:
    # def __init__(self, args: Dict[str, Any]) -> None:
    def __init__(self, n_models: int, n_layers: int, x_dim: int, h_dim: int, act_fn: ActFn,
                 infer_T: int, infer_lr: float, sigma_prior: float, sigma_obs: float,
                 sigma_data: float, activation_optim: str, n_proposal_samples: int,
                 activation_init_strat: str, layer_log_prob_strat: str,
                 layer_sample_strat: str, layer_update_strat: str, ensemble_log_joint_strat: str,
                 ensemble_proposal_strat: str, scale_layer: bool, resample: bool, **kwargs) -> None:
        self._n_models: int = n_models
        self._n_layers: int = n_layers
        self._x_dim: int = x_dim
        self._h_dim: int = h_dim
        self._sigma_prior: float = sigma_prior
        self._sigma_obs: float = sigma_obs
        self._sigma_data: float = sigma_data
        self._scale_layer: bool = scale_layer
        self._activation_optim: str = activation_optim
        self._log_weights: torch.Tensor = (torch.ones(self._n_models) / self._n_models).log()
        act_fn = ActFn.get_enum_from_value(act_fn)

        self._pcnets: List[PCNet] = self._initialize_base_models(n_models=n_models, act_fn=act_fn)
        self.activation_init_strat = ActInitStrat.get_enum_from_value(activation_init_strat)
        self.layer_log_prob_strat = LayerLogProbStrat.get_enum_from_value(layer_log_prob_strat)
        self.layer_sample_strat = LayerSampleStrat.get_enum_from_value(layer_sample_strat)
        self.layer_update_strat = LayerUpdateStrat.get_enum_from_value(layer_update_strat)
        self.ensemble_log_joint_strat = EnsembleLogJointStrat.get_enum_from_value(
                                            ensemble_log_joint_strat)
        self.ensemble_proposal_strat = EnsembleProposalStrat.get_enum_from_value(
                                            ensemble_proposal_strat)
        self.device: torch.device = torch.device('cpu')

        update_fn_args = dict(activation_init_fn=self._initialize_activation_group,
                              proposal_strat=self.ensemble_proposal_strat,
                              infer_lr=infer_lr, infer_T=infer_T, resample=resample,
                              n_proposal_samples=n_proposal_samples,
                              activation_optim=activation_optim)
        if self.ensemble_log_joint_strat == EnsembleLogJointStrat.SHARED:
            update_fn_args['ensemble_log_joint'] = self.log_joint

        if self.layer_update_strat == LayerUpdateStrat.ML:
            self._updater = MLUpdater(**update_fn_args)
        elif self.layer_update_strat == LayerUpdateStrat.BAYES:
            if self.ensemble_proposal_strat == EnsembleProposalStrat.MODE:
                self._updater = VLBModeUpdater(**update_fn_args)
            elif self.ensemble_proposal_strat == EnsembleProposalStrat.FULL:
                self._updater = VLBFullUpdater(**update_fn_args)
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

    def infer(self, X_obs: torch.Tensor, fixed_indices: torch.Tensor = None,
              n_repeat: int = 1) -> Prediction:
        original_device = X_obs.device
        X_obs = X_obs.to(self.device)
        if fixed_indices is not None:
            fixed_indices = fixed_indices.to(self.device)
        a_group = self._initialize_activation_group(X_obs=X_obs)

        infer_info = dict()
        for n in range(n_repeat):
            # Update hidden layer while fixing obs layer
            a_group.clamp(obs=True, hidden=False)
            hidden_info = maximize_log_joint(log_joint_fn=self.log_joint, a_group=a_group,
                                             infer_lr=self._updater.infer_lr,
                                             infer_T=self._updater.infer_T,
                                             fixed_indices=fixed_indices,
                                             activation_optim=self._activation_optim)
            obs_info = None
            if fixed_indices is None:
                # Update obs layer while fixing hidden layers
                a_group.clamp(obs=False, hidden=True)
                obs_info = maximize_log_joint(log_joint_fn=self.log_joint, a_group=a_group,
                                              infer_lr=self._updater.infer_lr,
                                              infer_T=self._updater.infer_T,
                                              fixed_indices=fixed_indices,
                                              activation_optim=self._activation_optim)
            infer_info[f"repeat_{n}"] = {'hidden': hidden_info, 'obs': obs_info}

        X_pred = a_group.get_acts(layer_index=0, detach=True).to(original_device)
        return Prediction(data=X_pred, a_group=a_group, info=infer_info)

    def learn(self, X_obs: torch.Tensor) -> UpdateResult:
        X_obs = X_obs.to(self.device)
        update_result = self._updater(X_obs=X_obs, pcnets=self._pcnets,
                                      log_weights=self._log_weights)
        self._pcnets = update_result.pcnets
        self._log_weights = update_result.log_weights
        return update_result

    def log_joint(self, a_group: ActivationGroup) -> torch.Tensor:
        ljs = torch.stack([pcnet.log_joint(a_group=a_group) for pcnet in self._pcnets], dim=1)
        weighted_ljs = ljs + self._log_weights.unsqueeze(0)
        return torch.logsumexp(weighted_ljs, dim=-1)

    def sample(self, d_batch: int = 1) -> Sample:
        """Select a base model based on its importance weight and sample from that model.

        Args:
            d_batch (int, optional): Number of datapoints to sample. Defaults to 1.

        Returns:
            (Sample): Sample object with <d_batch x x_dim> data and <d_batch> log joint tensors.
        """
        info = []
        weights = self._log_weights.exp()
        model_indices = torch.multinomial(weights, num_samples=d_batch, replacement=True)
        indices, counts = model_indices.unique(return_counts=True)
        for index, count in zip(indices.tolist(), counts.tolist()):
            if count > 0:
                sample, a_group = self._pcnets[index].sample(d_batch=count)
                info.append((sample, a_group))

        random.shuffle(info)
        data = torch.cat([sample_info[0] for sample_info in info], dim=0)
        a_group = ActivationGroup.merge(a_groups=[sample_info[1] for sample_info in info])
        log_joint = self.log_joint(a_group=a_group)
        return Sample(data=data, log_joint=log_joint)

    def _initialize_base_models(self, n_models: int, act_fn: ActFn) -> List[PCNet]:
        return [PCNet(n_layers=self._n_layers, x_dim=self._x_dim, h_dim=self._h_dim,
                      sigma_prior=self._sigma_prior, sigma_obs=self._sigma_obs,
                      sigma_data=self._sigma_data, act_fn=act_fn, scale_layer=self._scale_layer)
                for _ in range(n_models)]

    def _initialize_activation_group(self, X_obs: torch.Tensor) -> ActivationGroup:
        d_batch = X_obs.shape[0]
        activations = [deepcopy(X_obs)]
        if self.activation_init_strat == ActInitStrat.FIXED:
            for _ in range(self._n_layers-1):
                activations.append(torch.ones(d_batch, self._h_dim).to(self.device))
        elif self.activation_init_strat == ActInitStrat.RANDN:
            for _ in range(self._n_layers-1):
                activations.append(torch.randn(d_batch, self._h_dim).to(self.device))
        elif self.activation_init_strat == ActInitStrat.SAMPLE:
            raise NotImplementedError()
        else:
            raise NotImplementedError()
        return ActivationGroup(activations=activations)

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
