from typing import Dict, Type

import torch.nn as nn

from networks import BaseNetwork, Networks
from networks.efficientnetv2 import EfficientNetV2
from networks.resnet50 import ResNet50
from networks.vision_transformer import VisionTransformer


class NetworkRegistry:
    _registry: Dict[Networks, Type[BaseNetwork]] = {}

    @classmethod
    def register(cls, name: Networks, network_class: Type[BaseNetwork]) -> None:
        cls._registry[name] = network_class

    @classmethod
    def get(cls, name: Networks) -> Type[BaseNetwork]:
        if name not in cls._registry:
            raise ValueError(f"Network '{name}' not found in registry")
        return cls._registry[name]


NetworkRegistry.register("ResNet50", ResNet50)
NetworkRegistry.register("EfficientNetV2", EfficientNetV2)
NetworkRegistry.register("VisionTransformer", VisionTransformer)


class NetworkFactory:
    @staticmethod
    def create(
        name: Networks,
        include_top: bool = True,
        weights: str = None,
        num_classes: int = 1000,
    ) -> nn.Module:
        network_class = NetworkRegistry.get(name)
        return network_class(
            include_top=include_top,
            weights=weights,
            num_classes=num_classes,
        )
