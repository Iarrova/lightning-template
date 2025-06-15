from typing import TYPE_CHECKING, Tuple

import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets
from torchvision.transforms import v2

from src.datasets.base import BaseDataModule

if TYPE_CHECKING:
    from src.config import DatasetConfig


class ImagenetteDataset(BaseDataModule):
    NUM_CLASSES: int = 10

    def __init__(self, config: "DatasetConfig", data_dir: str = "./data"):
        super().__init__(config, data_dir)

    def prepare_data(self) -> None:
        datasets.Imagenette(root=self.data_dir, split="train", size="full", download=True)
        datasets.Imagenette(root=self.data_dir, split="val", size="full", download=True)

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit":
            train_transform, val_transform = self.get_transforms()
            
            imagenette = datasets.Imagenette(self.data_dir, split="train", transform=None)
            train_subset, val_subset = random_split(
                imagenette, lengths=[1 - self.config.validation_size, self.config.validation_size]
            )
            
            imagenette_train = datasets.Imagenette(self.data_dir, split="train", transform=train_transform)
            imagenette_val = datasets.Imagenette(self.data_dir, split="train", transform=val_transform)
            
            self.imagenette_train = Subset(imagenette_train, train_subset.indices)
            self.imagenette_val = Subset(imagenette_val, val_subset.indices)

        if stage == "test":
            _, test_transform = self.get_transforms()
            self.imagenette_test = datasets.Imagenette(self.data_dir, split="val", transform=test_transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.imagenette_train,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.imagenette_val,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.imagenette_test,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def get_transforms(self) -> Tuple[v2.Compose, v2.Compose]:
        normalize = [
            v2.ToImage(),
            v2.Resize(size=(224, 224)),
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
