from typing import Dict

from torchvision import datasets
from torchvision.transforms import v2

from datasets import DATA_DIR, Dataset, Datasets


class CIFAR10(Dataset):
    NUM_CLASSES: int = 10

    def __init__(
        self,
        dataset: Datasets = Datasets.CIFAR10,
        batch_size: int = 128,
        validation_size: float = 0.2,
        augment: bool = True,
        num_workers: int = 15,
    ):
        super().__init__(dataset, batch_size, validation_size, augment, num_workers)
        self.class_mapping = None

    def get_train_dataset(self, transform_train: v2.Compose):
        train_dataset = datasets.CIFAR10(
            root=DATA_DIR, train=True, download=True, transform=transform_train
        )

        self.class_mapping = train_dataset.class_to_idx
        return train_dataset

    def get_test_dataset(self, transform_test: v2.Compose):
        test_dataset = datasets.CIFAR10(
            root=DATA_DIR, train=False, download=True, transform=transform_test
        )
        return test_dataset

    def get_class_mapping(self) -> Dict[str, int]:
        if self.class_mapping is None:
            _ = datasets.CIFAR10(root=DATA_DIR, train=True, download=True)
            self.class_mapping = _.class_to_idx

        return self.class_mapping
