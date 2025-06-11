from abc import ABC, abstractmethod
from enum import StrEnum
from typing import List, Optional

import torch
from torch import nn


class Network(StrEnum):
    RESNET50 = "ResNet50"
    EFFICIENTNETV2 = "EfficientNetV2"
    VISION_TRANSFORMER = "VisionTransformer"


class BaseNetwork(nn.Module, ABC):
    def __init__(
        self,
        include_top: bool = True,
        weights: Optional[str] = None,
        num_classes: int = 1000,
    ):
        super().__init__()
        self.include_top = include_top
        self.weights = weights
        self.num_classes = num_classes
        self.model = self._create_model()

    @abstractmethod
    def _create_model(self) -> nn.Module:
        pass

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(batch)

    def get_gradcam_layer(self) -> List[nn.Module]:
        raise NotImplementedError("Grad-CAM not implemented for this network")
