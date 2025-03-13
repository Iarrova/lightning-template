from typing import List

import torch
from torch import nn
from torchvision import models

from networks import BaseNetwork


class ResNet50(BaseNetwork):
    def _create_model(self) -> nn.Module:
        if self.weights == "imagenet":
            model = models.resnet50(weights="IMAGENET1K_V2")
        else:
            model = models.resnet50(weights=None)
            if self.weights is not None:
                model.load_state_dict(torch.load(self.weights))

        if not self.include_top:
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)

        return model

    def get_gradcam_layer(self) -> List[nn.Module]:
        return [self.model.layer4[-1]]
