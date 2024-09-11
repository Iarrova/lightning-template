import torch
from torch import nn
from torchvision import models
from typing import Optional


class EfficientNetV2(nn.Module):
    def __init__(
        self,
        include_top: bool = True,
        weights: Optional[str] = None,
        num_classes: int = 1000,
    ):
        super().__init__()

        if weights == "imagenet":
            model = models.efficientnet_v2_s(weights="IMAGENET1K_V1")
        else:
            model = models.efficientnet_v2_s(weights=None)
            if weights is not None:
                model.load_state_dict(torch.load(weights))

        if not include_top:
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(model.classifier[1].in_features, num_classes),
            )

        self.model = model

    def forward(self, batch):
        return self.model(batch)
