from pathlib import Path
from typing import Dict

from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import v2

from config.config import DatasetConfig
from datasets import BaseDataset


class ImagenetteDataset(BaseDataset):
    NUM_CLASSES: int = 10

    def __init__(self, config: DatasetConfig):
        super().__init__(config)
        self.root = Path(__file__).resolve().parent.parent / config.data_dir

    def get_train_dataset(self, transform_train: v2.Compose) -> Dataset:
        train_dataset = datasets.Imagenette(
            root=self.root,
            split="train",
            size="full",
            download=True,
            transform=transform_train,
        )
        return train_dataset

    def get_test_dataset(self, transform_test: v2.Compose) -> Dataset:
        test_dataset = datasets.Imagenette(
            root=self.root,
            split="val",
            size="full",
            download=True,
            transform=transform_test,
        )
        return test_dataset

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
