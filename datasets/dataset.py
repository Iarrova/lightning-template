from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

from torch.utils.data import DataLoader, random_split
from torchvision.transforms.v2 import Compose


class Dataset(ABC):
    def __init__(
        self,
        batch_size: int = 128,
        validation_size: float = 0.2,
        augment: bool = True,
        num_workers: int = 15,
    ):
        self.batch_size: int = batch_size
        self.validation_size: float = validation_size
        self.augment: bool = augment
        self.num_workers: int = num_workers

    @abstractmethod
    def get_train_dataset(self, transform_train: Compose) -> Any:
        """Returns the training dataset (Torchvision's built in, or ImageFolder and similars) for our data"""
        pass

    @abstractmethod
    def get_test_dataset(self, transform_test: Compose) -> Any:
        """Returns the testing dataset (Torchvision's built in, or ImageFolder and similars) for our data"""
        pass

    @abstractmethod
    def get_transforms(self) -> Tuple[Compose, Compose]:
        """Returns the training and testing transforms for our data"""
        pass

    @abstractmethod
    def get_class_mapping(self) -> Dict[str, int]:
        """A dicionary with the class mappings. For example {"airplane": 0}"""
        pass

    def generate_train_loaders(self) -> Tuple[DataLoader, DataLoader]:
        transform_train, _ = self.get_transforms()
        train_dataset = self.get_train_dataset(transform_train)

        train_set, validation_set = random_split(
            train_dataset, lengths=[1 - self.validation_size, self.validation_size]
        )

        train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        validation_loader = DataLoader(
            validation_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        return train_loader, validation_loader

    def generate_test_loader(self, shuffle: bool = False) -> DataLoader:
        _, transform_test = self.get_transforms()
        test_dataset = self.get_test_dataset(transform_test)

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )

        return test_loader
