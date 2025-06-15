from enum import StrEnum
from typing import TYPE_CHECKING

from src.datasets.base import BaseDataModule
from src.datasets.cifar10 import CIFAR10Dataset
from src.datasets.imagenette import ImagenetteDataset

if TYPE_CHECKING:
    from src.config import DatasetConfig


class Dataset(StrEnum):
    CIFAR10 = "CIFAR10"
    Imagenette = "Imagenette"


DATASET_MAPPING = {
    Dataset.CIFAR10: CIFAR10Dataset,
    Dataset.Imagenette: ImagenetteDataset,
}


def create_dataset(config: "DatasetConfig") -> BaseDataModule:
    if config.dataset not in DATASET_MAPPING:
        raise ValueError(f"Unknown dataset: {config.dataset}")

    dataset_class = DATASET_MAPPING[config.dataset]
    return dataset_class(config)
