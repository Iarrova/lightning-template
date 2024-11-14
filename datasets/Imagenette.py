from typing import Dict, Tuple

import torch
from torchvision import datasets
from torchvision.transforms import v2

from datasets.dataset import Dataset


class Imagenette(Dataset):
    num_classes = 10

    def __init__(
        self,
        batch_size: int = 128,
        validation_size: float = 0.2,
        augment: bool = True,
        num_workers: int = 15,
    ):
        super().__init__(batch_size, validation_size, augment, num_workers)

    def get_train_dataset(self, transform_train: v2.Compose):
        train_dataset = datasets.Imagenette(
            root="./data",
            split="train",
            size="full",
            download=False,
            transform=transform_train,
        )
        return train_dataset

    def get_test_dataset(self, transform_test: v2.Compose):
        test_dataset = datasets.Imagenette(
            root="./data",
            split="val",
            size="full",
            download=False,
            transform=transform_test,
        )
        return test_dataset

    def get_transforms(self) -> Tuple[v2.Compose, v2.Compose]:
        normalize = [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        transform_train = (
            v2.Compose([v2.RandomHorizontalFlip(p=0.5)] + normalize)
            if self.augment
            else v2.Compose(normalize)
        )
        transform_test = v2.Compose(normalize)

        return transform_train, transform_test

    def get_class_mapping(self) -> Dict[str, int]:
        return {
            "tench": 0,
            "english_springer": 1,
            "cassette_player": 2,
            "chainsaw": 3,
            "church": 4,
            "french_horn": 5,
            "garbage_truck": 6,
            "gas_pump": 7,
            "golf_ball": 8,
            "parachute": 9,
        }
