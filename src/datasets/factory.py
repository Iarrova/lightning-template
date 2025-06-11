from typing import Dict, Type

from config.config import DatasetConfig
from datasets import BaseDataset
from datasets.CIFAR10 import CIFAR10Dataset
from datasets.enums import Dataset
from datasets.Imagenette import ImagenetteDataset


class DatasetRegistry:
    _registry: Dict[Dataset, Type[BaseDataset]] = {}

    @classmethod
    def register(cls, name: Dataset, dataset_class: Type[BaseDataset]) -> None:
        cls._registry[name] = dataset_class

    @classmethod
    def get(cls, name: Dataset) -> Type[BaseDataset]:
        if name not in cls._registry:
            raise ValueError(f"Dataset '{name}' not found in registry")
        return cls._registry[name]


DatasetRegistry.register(Dataset.CIFAR10, CIFAR10Dataset)
DatasetRegistry.register(Dataset.Imagenette, ImagenetteDataset)


class DatasetFactory:
    @staticmethod
    def create(config: DatasetConfig) -> BaseDataset:
        dataset_class = DatasetRegistry.get(config.dataset)
        return dataset_class(config)
