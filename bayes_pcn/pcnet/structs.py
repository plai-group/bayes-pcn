import torch
from typing import Any, Dict, List, NamedTuple, Tuple

from .activations.a_group import ActivationGroup


class DataBatch(NamedTuple):
    train: Tuple[torch.Tensor, torch.Tensor]
    tests: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
    original_shape: torch.Size
    train_pred: Tuple[torch.Tensor, torch.Tensor] = None
    tests_pred: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = None
    info: Dict[str, Any] = None


class LogProbResult(NamedTuple):
    log_prob: torch.Tensor
    layer_log_probs: List[torch.Tensor]


class Prediction(NamedTuple):
    data: torch.Tensor
    a_group: ActivationGroup
    info: Dict[str, Any] = None


class Sample(NamedTuple):
    data: torch.Tensor
    log_joint: float
    info: Dict[str, Any] = None


class UpdateResult(NamedTuple):
    pcnets: List[Any]
    log_weights: torch.Tensor
    info: Dict[str, Any] = None
