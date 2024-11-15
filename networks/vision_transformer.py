from typing import Optional

import torch
from torch import nn
from torchvision import models


class VisionTransformer(nn.Module):
    def __init__(
        self,
        include_top: bool = True,
        weights: Optional[str] = None,
        num_classes: int = 1000,
    ):
        super().__init__()

        if weights == "imagenet":
            model = models.vit_b_16(weights="IMAGENET1K_V1")
        else:
            model = models.vit_b_16(weights=None)
            if weights is not None:
                model.load_state_dict(torch.load(weights))

        if not include_top:
            model.heads = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(model.heads[0].in_features, num_classes),
            )

        self.model = model

    def forward(self, batch):
        return self.model(batch)
