from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms.v2 import Compose

if TYPE_CHECKING:
    from src.config import DatasetConfig


class BaseDataset(ABC):
    NUM_CLASSES: int

    def __init__(self, config: "DatasetConfig"):
        self.config = config

    @property
    def num_classes(self) -> int:
        return self.NUM_CLASSES

    @abstractmethod
    def get_train_dataset(self, transform_train: Compose) -> Dataset:
        pass

    @abstractmethod
    def get_test_dataset(self, transform_test: Compose) -> Dataset:
        pass

    @abstractmethod
    def get_class_mapping(self) -> Dict[str, int]:
        pass

    @abstractmethod
    def get_transforms(self) -> Tuple[Compose, Compose]:
        pass

    def generate_train_loaders(self) -> Tuple[DataLoader, DataLoader]:
        transform_train, _ = self.get_transforms()
        train_dataset = self.get_train_dataset(transform_train)

        train_set, validation_set = random_split(
            train_dataset, lengths=[1 - self.config.validation_size, self.config.validation_size]
        )

        train_loader = DataLoader(
            train_set,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        validation_loader = DataLoader(
            validation_set,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        return train_loader, validation_loader

    def generate_test_loader(self, shuffle: bool = False) -> DataLoader:
        _, transform_test = self.get_transforms()
        test_dataset = self.get_test_dataset(transform_test)

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        return test_loader
