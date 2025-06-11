from typing import List

import torch
from torch import nn
from torchvision import models

from networks import BaseNetwork


class EfficientNetV2(BaseNetwork):
    def _create_model(self) -> nn.Module:
        if self.weights == "imagenet":
            model = models.efficientnet_v2_s(weights="IMAGENET1K_V1")
        else:
            model = models.efficientnet_v2_s(weights=None)
            if self.weights is not None:
                model.load_state_dict(torch.load(self.weights))

        if not self.include_top:
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(model.classifier[1].in_features, self.num_classes),
            )

        return model

    def get_gradcam_layer(self) -> List[nn.Module]:
        return [self.model.features[-1][-1]]
