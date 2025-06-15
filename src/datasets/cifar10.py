from typing import TYPE_CHECKING, Tuple, Optional

import torch
from torchvision import datasets
from torch.utils.data import random_split, DataLoader, Subset
from torchvision.transforms import v2

from src.datasets.base import BaseDataModule

if TYPE_CHECKING:
    from src.config import DatasetConfig


class CIFAR10Dataset(BaseDataModule):
    NUM_CLASSES: int = 10

    def __init__(self, config: "DatasetConfig", data_dir: str = "./data"):
        super().__init__(config, data_dir)

    def prepare_data(self) -> None:
        datasets.CIFAR10(root=self.data_dir, train=True, download=True)
        datasets.CIFAR10(root=self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit":
            train_transform, val_transform = self.get_transforms()
            
            cifar10 = datasets.CIFAR10(self.data_dir, train=True, transform=None)
            train_subset, val_subset = random_split(
                cifar10, lengths=[1 - self.config.validation_size, self.config.validation_size]
            )
            
            cifar10_train = datasets.CIFAR10(self.data_dir, train=True, transform=train_transform)
            cifar10_val = datasets.CIFAR10(self.data_dir, train=True, transform=val_transform)
            
            self.cifar_train = Subset(cifar10_train, train_subset.indices)
            self.cifar_val = Subset(cifar10_val, val_subset.indices)

        if stage == "test":
            _, test_transform = self.get_transforms()
            self.cifar_test = datasets.CIFAR10(self.data_dir, train=False, transform=test_transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.cifar_train,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.cifar_val,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.cifar_test,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

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
