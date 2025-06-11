from typing import Dict

from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import v2

from config.config import DatasetConfig
from datasets import BaseDataset


class CIFAR10Dataset(BaseDataset):
    NUM_CLASSES: int = 10

    def __init__(self, config: DatasetConfig):
        super().__init__(config)
        self._class_mapping: Dict[str, int] | None = None

    def get_train_dataset(self, transform_train: v2.Compose) -> Dataset:
        train_dataset = datasets.CIFAR10(
            root=self.config.data_dir, train=True, download=True, transform=transform_train
        )

        self._class_mapping = train_dataset.class_to_idx
        return train_dataset

    def get_test_dataset(self, transform_test: v2.Compose) -> Dataset:
        test_dataset = datasets.CIFAR10(
            root=self.config.data_dir, train=False, download=True, transform=transform_test
        )
        return test_dataset

    def get_class_mapping(self) -> Dict[str, int]:
        if self._class_mapping is None:
            _ = datasets.CIFAR10(root=self.config.data_dir, train=True, download=False)
            self._class_mapping = _.class_to_idx

        return self._class_mapping
