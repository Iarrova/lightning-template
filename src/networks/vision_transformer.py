import torch
from torch import nn
from torchvision import models

from src.config import NetworkConfig
from src.networks.base import BaseNetwork


class VisionTransformer(BaseNetwork):
    def __init__(self, config: NetworkConfig, num_classes: int) -> None:
        super().__init__(config, num_classes)

        if self.network_config.pytorch_weights == "ImageNet":
            model = models.vit_b_16(weights="IMAGENET1K_V1")
        else:
            model = models.vit_b_16(weights=None)
            if self.network_config.pytorch_weights is not None:
                model.load_state_dict(torch.load(self.network_config.pytorch_weights))

        if not self.network_config.include_top:
            model.heads = nn.Sequential(
                nn.Linear(model.heads[0].in_features, self.num_classes),
            )

        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
