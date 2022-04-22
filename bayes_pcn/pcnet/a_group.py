from copy import deepcopy
import torch
import torch.nn as nn

from typing import List, Union


class ActivationGroup:
    def __init__(self, activations: List[torch.Tensor], no_param: bool = False) -> None:
        """Contains all layer-wise activations of PCNets within PCNetEnsemble. Makes things
        easier to work with PyTorch optimizers. Does not modify bottom-most activations.
        NOTE: Layer activations are not parameters if no_param is not true to
            have the method work with torch.autograd.hessian.

        Args:
            activations (List[torch.Tensor]): A list of PCNetEnsemble activations.
                The list enumerates over layers PCNetEnsemble layers in ascending order.
                The element tensor shapes should be <d_batch x d_layer>.
        """
        self._device: torch.device = torch.device('cpu')
        self._original_obs: torch.Tensor = None if no_param else deepcopy(activations[0])
        layer_acts = activations[0] if no_param else nn.Parameter(deepcopy(activations[0]))
        self._data: List[torch.Tensor] = [layer_acts]
        self._dims: List[int] = [activations[0].shape[-1]]
        for layer_acts in activations[1:]:
            layer_acts = layer_acts if no_param else nn.Parameter(deepcopy(layer_acts))
            self._data.append(layer_acts)
            self._dims.append(layer_acts.shape[-1])
        self._d_batch: int = layer_acts.shape[0]

    def get_acts(self, layer_index: int, detach: bool = True) -> torch.Tensor:
        if layer_index >= len(self._data):
            return None
        result = self._data[layer_index]
        return result.detach() if detach else result

    def set_acts(self, layer_index: int, value: torch.Tensor):
        if layer_index >= len(self._data):
            return None
        with torch.no_grad():
            self._data[layer_index] += value - self._data[layer_index]

    def get_data(self, flatten: bool = False, no_obs: bool = False
                 ) -> Union[List[torch.Tensor], torch.Tensor]:
        result = self._data
        if no_obs:
            result = result[1:]
        return torch.cat(result, dim=-1) if flatten else result

    def get_datapoint(self, data_index: int, flatten: bool = False
                      ) -> Union[List[torch.Tensor], torch.Tensor]:
        result = [layer_acts[data_index:data_index+1] for layer_acts in self._data]
        return torch.cat(result, dim=-1) if flatten else result

    def clamp(self, obs: bool = None, hidden: bool = None) -> None:
        assert (obs is not None) or (hidden is not None)
        if obs is not None:
            self._data[0].requires_grad = not obs
        if hidden is not None:
            for i in range(1, len(self._data)):
                self._data[i].requires_grad = not hidden

    @classmethod
    def from_concatenated(cls, activations: torch.Tensor, dims: List[int]) -> 'ActivationGroup':
        """Given activation matrix, return an ActivationGroup object that chunks them according
        to dimensions in dims. Preserves gradients if they exist in activations.

        Args:
            activations (torch.Tensor): Activation matrix of shape <d_batch x |network neurons|>
            dims (List[int]): A list of activation dimensions from bottom to top layers.

        Returns:
            ActivationGroup: ActivationGroup object created from activations.
        """
        separated = []
        curr_loc = 0
        for dim in dims:
            layer_acts = activations[:, curr_loc:curr_loc+dim]
            separated.append(layer_acts)
            curr_loc = curr_loc + dim
        return cls(activations=separated, no_param=True)

    @classmethod
    def merge(cls, a_groups: List['ActivationGroup']) -> 'ActivationGroup':
        """Merge a list of ActivationGroup into a single ActivationGroup by stacking them
        along the batch dimension.

        Args:
            a_groups (List[ActivationGroup]): A list of valid activation groups.

        Returns:
            ActivationGroup: A new activation group that combines elements of a_groups.
        """
        activations = []
        for i in range(len(a_groups[0].dims)):
            layer_acts = torch.cat([a_group.get_acts(layer_index=i, detach=True)
                                    for a_group in a_groups], dim=0)
            activations.append(layer_acts)
        return cls(activations=activations)

    @property
    def data(self):
        return self._data

    @property
    def dims(self):
        return self._dims

    @property
    def original_obs(self):
        return self._original_obs

    @property
    def d_batch(self):
        return self._d_batch

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device: torch.device):
        self._device = device
        if self._original_obs is not None:
            self._original_obs = self._original_obs.to(device)
        for i in range(len(self._data)):
            self._data[i] = self._data[i].to(device)
