from typing import Optional

import torch
from torch import nn
from torchvision import models


class ResNet50(nn.Module):
    def __init__(
        self,
        include_top: bool = True,
        weights: Optional[str] = None,
        num_classes: int = 1000,
    ):
        super().__init__()

        if weights == "imagenet":
            model = models.resnet50(weights="IMAGENET1K_V2")
        else:
            model = models.resnet50(weights=None)
            if weights is not None:
                model.load_state_dict(torch.load(weights))

        if not include_top:
            model.fc = nn.Linear(model.fc.in_features, num_classes)

        self.model = model

    def forward(self, batch):
        return self.model(batch)
