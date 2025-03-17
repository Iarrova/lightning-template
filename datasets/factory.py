from typing import Dict, Type

from datasets import Dataset, Datasets
from datasets.CIFAR10 import CIFAR10
from datasets.Imagenette import Imagenette


class DatasetRegistry:
    _registry: Dict[Datasets, Type[Dataset]] = {}

    @classmethod
    def register(cls, name: Datasets, dataset_class: Type[Dataset]) -> None:
        cls._registry[name] = dataset_class

    @classmethod
    def get(cls, name: Datasets) -> Type[Dataset]:
        if name not in cls._registry:
            raise ValueError(f"Dataset '{name}' not found in registry")
        return cls._registry[name]


DatasetRegistry.register(Datasets.CIFAR10, CIFAR10)
DatasetRegistry.register(Datasets.Imagenette, Imagenette)


class DatasetFactory:
    @staticmethod
    def create(
        dataset: Datasets,
        batch_size: int = 128,
        validation_size: float = 0.2,
        augment: bool = True,
        num_workers: int = 15,
    ) -> Dataset:
        dataset_class = DatasetRegistry.get(dataset)
        return dataset_class(
            dataset=dataset,
            batch_size=batch_size,
            validation_size=validation_size,
            augment=augment,
            num_workers=num_workers,
        )
