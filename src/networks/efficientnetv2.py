import torch
from torch import nn
from torchvision import models

from src.networks.base import BaseNetwork


class EfficientNetV2(BaseNetwork):
    def __init__(self) -> None:
        if self.network_config.pytorch_weights == "ImageNet":
            model = models.efficientnet_v2_s(weights="IMAGENET1K_V1")
        else:
            model = models.efficientnet_v2_s(weights=None)
            if self.network_config.pytorch_weights is not None:
                model.load_state_dict(torch.load(self.network_config.pytorch_weights))

        if not self.network_config.include_top:
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(model.classifier[1].in_features, self.num_classes),
            )

        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)