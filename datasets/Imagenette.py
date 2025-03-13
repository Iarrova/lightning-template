import os
from typing import Dict

from torchvision import datasets
from torchvision.transforms import v2

from datasets import DATA_DIR, Dataset, Datasets


class Imagenette(Dataset):
    num_classes: int = 10

    def __init__(
        self,
        dataset: Datasets = Datasets.Imagenette,
        batch_size: int = 128,
        validation_size: float = 0.2,
        augment: bool = True,
        num_workers: int = 15,
    ):
        super().__init__(dataset, batch_size, validation_size, augment, num_workers)
        self.root = os.path.join(os.path.dirname(__file__), "..", DATA_DIR)

    def get_train_dataset(self, transform_train: v2.Compose):
        train_dataset = datasets.Imagenette(
            root=self.root,
            split="train",
            size="full",
            download=False,
            transform=transform_train,
        )
        return train_dataset

    def get_test_dataset(self, transform_test: v2.Compose):
        test_dataset = datasets.Imagenette(
            root=self.root,
            split="val",
            size="full",
            download=False,
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
