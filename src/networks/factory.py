from typing import Dict, Type

import torch.nn as nn

from src.config.config import NetworkConfig, WeightsConfig
from src.networks import BaseNetwork
from src.networks.efficientnetv2 import EfficientNetV2
from src.networks.enums import Network
from src.networks.resnet50 import ResNet50
from src.networks.vision_transformer import VisionTransformer


class NetworkRegistry:
    _registry: Dict[Network, Type[BaseNetwork]] = {}

    @classmethod
    def register(cls, name: Network, network_class: Type[BaseNetwork]) -> None:
        cls._registry[name] = network_class

    @classmethod
    def get(cls, name: Network) -> Type[BaseNetwork]:
        if name not in cls._registry:
            raise ValueError(f"Network '{name}' not found in registry")
        return cls._registry[name]


NetworkRegistry.register(Network.RESNET50, ResNet50)
NetworkRegistry.register(Network.EFFICIENTNETV2, EfficientNetV2)
NetworkRegistry.register(Network.VISION_TRANSFORMER, VisionTransformer)


class NetworkFactory:
    @staticmethod
    def create(
        network_config: NetworkConfig, weights_config: WeightsConfig, num_classes: int = 1000
    ) -> nn.Module:
        network_class = NetworkRegistry.get(network_config.network)
        return network_class(network_config, weights_config, num_classes=num_classes)
