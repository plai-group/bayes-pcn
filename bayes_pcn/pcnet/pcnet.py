import torch
from typing import List, Tuple

from bayes_pcn.const import *
from .layers import *
from .util import *


class PCNet:
    def __init__(self, n_layers: int, x_dim: int, h_dim: int, act_fn: ActFn,
                 sigma_prior: float, sigma_obs: float, scale_layer: bool) -> None:
        self._layers = self._init_layers(n_layers=n_layers, x_dim=x_dim, d_h=h_dim,
                                         sigma_prior=sigma_prior, sigma_obs=sigma_obs,
                                         act_fn=act_fn, scale_layer=scale_layer)
        self.layer_log_prob_strat = None
        self.layer_sample_strat = None
        self.layer_update_strat = None
        self.device = torch.device('cpu')

    def sample(self, d_batch: int = 1) -> Tuple[torch.Tensor, ActivationGroup]:
        """Sample observations from the model via ancestral sampling.

        Args:
            d_batch (int, optional): Number of datapoints to sample. Defaults to 1.

        Returns:
            Tuple[torch.Tensor, ActivationGroup]: Samples of shape <d_batch x d_obs>
                and the full traces associated with samples.
        """
        result = None
        traces = []
        for layer in reversed(self.layers):
            result = layer.sample(d_batch=d_batch, X_in=result)
            traces.append(result)
        return result, ActivationGroup(activations=traces[::-1])

    def log_joint(self, a_group: ActivationGroup) -> torch.Tensor:
        """Return log joint of network layer activations.

        Args:
            a_group (ActivationGroup): Activation values for all layers.

        Returns:
            torch.Tensor: Log probability vector of shape <d_batch>.
        """
        result = torch.zeros(a_group.d_batch).to(self.device)
        for i, layer in enumerate(self.layers):
            upper_activation = a_group.get_acts(layer_index=i+1, detach=False)
            lower_activation = a_group.get_acts(layer_index=i, detach=False)
            result = result + layer.log_prob(X_obs=lower_activation, X_in=upper_activation)
        return result

    def update_weights(self, a_group: ActivationGroup, **kwargs) -> None:
        """Update all layer weights according to self._layer_update_strat given
        all layer activations.

        Args:
            a_group (ActivationGroup): Activation values for all layers.
        """
        for i, layer in enumerate(self.layers):
            upper_activation = a_group.get_acts(layer_index=i+1, detach=True)
            lower_activation = a_group.get_acts(layer_index=i, detach=True)
            layer.update(X_obs=lower_activation, X_in=upper_activation, **kwargs)

    def _init_layers(self, n_layers: int, x_dim: int, d_h: int, sigma_prior: float,
                     sigma_obs: float, act_fn: ActFn, scale_layer: bool, **kwargs
                     ) -> List[AbstractPCLayer]:
        shared_args = dict(sigma_prior=sigma_prior, sigma_obs=sigma_obs)
        layers = [PCLayer(d_in=d_h, d_out=x_dim, act_fn=act_fn, scale_layer=scale_layer,
                          **shared_args, **kwargs)]
        for _ in range(n_layers-2):
            layers.append(PCLayer(d_in=d_h, d_out=d_h, act_fn=act_fn, scale_layer=scale_layer,
                                  **shared_args, **kwargs))
        layers.append(PCTopLayer(d_out=d_h, **shared_args, **kwargs))
        return layers

    @property
    def layers(self):
        return self._layers

    @property
    def layer_log_prob_strat(self):
        return self._layer_log_prob_strat

    @layer_log_prob_strat.setter
    def layer_log_prob_strat(self, value: LayerLogProbStrat):
        self._layer_log_prob_strat = value
        for layer in self.layers:
            layer.log_prob_strat = value

    @property
    def layer_sample_strat(self):
        return self._layer_sample_strat

    @layer_sample_strat.setter
    def layer_sample_strat(self, value: LayerSampleStrat):
        self._layer_sample_strat = value
        for layer in self.layers:
            layer.sample_strat = value

    @property
    def layer_update_strat(self):
        return self._layer_update_strat

    @layer_update_strat.setter
    def layer_update_strat(self, value: LayerUpdateStrat):
        self._layer_update_strat = value
        for layer in self.layers:
            layer.update_strat = value

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value: torch.device):
        self._device = value
        for layer in self.layers:
            layer.device = value
        for var in vars(self):
            if isinstance(self.__dict__[var], torch.Tensor):
                self.__dict__[var] = self.__dict__[var].to(value)
