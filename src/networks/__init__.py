from abc import ABC, abstractmethod

import torch
from torch import nn

from src.config.config import NetworkConfig, WeightsConfig


class BaseNetwork(nn.Module, ABC):
    def __init__(self, network_config: NetworkConfig, weights_config: WeightsConfig, num_classes: int = 1000):
        super().__init__()
        self.network_config = network_config
        self.weights_config = weights_config
        self.num_classes = num_classes
        self.model = self._create_model()

    @abstractmethod
    def _create_model(self) -> nn.Module:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
