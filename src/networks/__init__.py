from enum import StrEnum
from typing import TYPE_CHECKING

from src.networks.base import BaseNetwork
from src.networks.efficientnetv2 import EfficientNetV2
from src.networks.resnet50 import ResNet50
from src.networks.vision_transformer import VisionTransformer

if TYPE_CHECKING:
    from src.config import NetworkConfig


class Network(StrEnum):
    RESNET50 = "ResNet50"
    EFFICIENTNETV2 = "EfficientNetV2"
    VISION_TRANSFORMER = "VisionTransformer"


NETWORK_MAPPING = {
    Network.RESNET50: ResNet50,
    Network.EFFICIENTNETV2: EfficientNetV2,
    Network.VISION_TRANSFORMER: VisionTransformer,
}


def create_network(config: "NetworkConfig", num_classes: int) -> BaseNetwork:
    if config.network not in NETWORK_MAPPING:
        raise ValueError(f"Unknown dataset: {config.network}")

    network_class = NETWORK_MAPPING[config.network]
    return network_class(config, num_classes)
