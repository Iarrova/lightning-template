from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Tuple

import lightning as L
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose

if TYPE_CHECKING:
    from src.config import DatasetConfig


class BaseDataModule(L.LightningDataModule, ABC):
    NUM_CLASSES: int

    def __init__(self, config: "DatasetConfig", data_dir: str = "./data"):
        super().__init__()
        self.config = config
        self.data_dir = data_dir

    @property
    def num_classes(self) -> int:
        return self.NUM_CLASSES

    @abstractmethod
    def prepare_data(self) -> None:
        pass

    @abstractmethod
    def setup(self, stage: Optional[str] = None) -> None:
        pass

    @abstractmethod
    def train_dataloader(self) -> DataLoader:
        pass

    @abstractmethod
    def val_dataloader(self) -> DataLoader:
        pass

    @abstractmethod
    def test_dataloader(self) -> DataLoader:
        pass

    @abstractmethod
    def get_transforms(self) -> Tuple[Compose, Compose]:
        pass
