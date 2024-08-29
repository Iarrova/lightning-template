from enum import StrEnum
from pydantic import BaseModel, Field
from typing import Annotated


class Datasets(StrEnum):
    CIFAR10 = "CIFAR10"


class Networks(StrEnum):
    RESNET50 = "ResNet50"
    EFFICIENTNETV2 = "EfficientNetV2"


class Config(BaseModel):
    seed: int
    dataset: Datasets
    batch_size: int
    validation_size: Annotated[float, Field(strict=True, gt=0, lt=1)]
    augment: bool
    network: Networks
    include_top: bool
    weights: str
    learning_rate: Annotated[float, Field(strict=True, gt=0)]
    num_epochs: int
    weights_dir: str
    weights_path: str
