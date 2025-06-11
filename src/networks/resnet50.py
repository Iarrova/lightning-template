import torch
from torch import nn
from torchvision import models

from src.networks import BaseNetwork


class ResNet50(BaseNetwork):
    def _create_model(self) -> nn.Module:
        if self.weights_config.pretrained_weights == "ImageNet":
            model = models.resnet50(weights="IMAGENET1K_V2")
        else:
            model = models.resnet50(weights=None)
            if self.weights_config.pretrained_weights is not None:
                model.load_state_dict(torch.load(self.weights_config.pretrained_weights))

        if not self.network_config.include_top:
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)

        return model
