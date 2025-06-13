from typing import TYPE_CHECKING, Dict, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import v2

from src.datasets.base import BaseDataset

if TYPE_CHECKING:
    from src.config import DatasetConfig


class CIFAR10Dataset(BaseDataset):
    NUM_CLASSES: int = 10

    def __init__(self, config: "DatasetConfig"):
        super().__init__(config)
        self.data_dir = "./data"
        self._class_mapping: Dict[str, int] | None = None

    def get_transforms(self) -> Tuple[v2.Compose, v2.Compose]:
        normalize = [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        if self.config.augment:
            train_transforms = [v2.RandomHorizontalFlip(p=0.5)] + normalize
        else:
            train_transforms = normalize

        transform_train = v2.Compose(train_transforms)
        transform_test = v2.Compose(normalize)

        return transform_train, transform_test

    def get_train_dataset(self, transform_train: v2.Compose) -> Dataset:
        train_dataset = datasets.CIFAR10(
            root=self.data_dir, train=True, download=True, transform=transform_train
        )

        self._class_mapping = train_dataset.class_to_idx
        return train_dataset

    def get_test_dataset(self, transform_test: v2.Compose) -> Dataset:
        return datasets.CIFAR10(root=self.data_dir, train=False, download=True, transform=transform_test)

    def get_class_mapping(self) -> Dict[str, int]:
        if self._class_mapping is None:
            _ = datasets.CIFAR10(root=self.data_dir, train=True, download=False)
            self._class_mapping = _.class_to_idx

        return self._class_mapping
