from enum import StrEnum
from pydantic import BaseModel, Field
from typing import Annotated


class Datasets(StrEnum):
    CIFAR10 = "CIFAR10"


class Networks(StrEnum):
    RESNET50 = "ResNet50"


class Config(BaseModel):
    seed: int = 42
    dataset: Datasets
    batch_size: int = 64
    validation_size: Annotated[float, Field(strict=True, gt=0, lt=1)] = 0.2
    augment: bool = True
    network: Networks
    learning_rate: Annotated[float, Field(strict=True, gt=0)] = 0.001
    num_epochs: int = 10
    weights_dir: str
    weights_path: str
