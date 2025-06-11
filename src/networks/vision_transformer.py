from typing import List

import torch
from torch import nn
from torchvision import models

from networks import BaseNetwork


class VisionTransformer(BaseNetwork):
    def _create_model(self) -> nn.Module:
        if self.weights == "imagenet":
            model = models.vit_b_16(weights="IMAGENET1K_V1")
        else:
            model = models.vit_b_16(weights=None)
            if self.weights is not None:
                model.load_state_dict(torch.load(self.weights))

        if not self.include_top:
            model.heads = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(model.heads[0].in_features, self.num_classes),
            )

        return model

    def get_gradcam_layer(self) -> List[nn.Module]:
        return [self.model.encoder.layers[-1].ln_1]
