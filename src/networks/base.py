from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from src.config import NetworkConfig


class BaseNetwork(nn.Module, ABC):
    def __init__(self, network_config: "NetworkConfig", num_classes: int):
        super().__init__()
        self.network_config = network_config
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
