from abc import ABC, abstractmethod
from typing import Dict, Tuple, Type

import torch
from torchvision.transforms import v2

from datasets.enums import Dataset


class TransformStrategy(ABC):
    @abstractmethod
    def get_transforms(self, augment: bool = True) -> Tuple[v2.Compose, v2.Compose]:
        pass


class CIFAR10Transforms(TransformStrategy):
    def get_transforms(self, augment: bool = True) -> Tuple[v2.Compose, v2.Compose]:
        normalize = [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        if augment:
            train_transforms = [v2.RandomHorizontalFlip(p=0.5)] + normalize
        else:
            train_transforms = normalize

        transform_train = v2.Compose(train_transforms)
        transform_test = v2.Compose(normalize)

        return transform_train, transform_test


class ImagenetteTransforms(TransformStrategy):
    def get_transforms(self, augment: bool = True) -> Tuple[v2.Compose, v2.Compose]:
        normalize = [
            v2.ToImage(),
            v2.Resize(size=(224, 224)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        if augment:
            train_transforms = [v2.RandomHorizontalFlip(p=0.5)] + normalize
        else:
            train_transforms = normalize

        transform_train = v2.Compose(train_transforms)
        transform_test = v2.Compose(normalize)

        return transform_train, transform_test


class TransformRegistry:
    _registry: Dict[Dataset, Type[TransformStrategy]] = {}

    @classmethod
    def register(cls, name: Dataset, transform_class: Type[TransformStrategy]) -> None:
        cls._registry[name] = transform_class

    @classmethod
    def get(cls, name: Dataset) -> Type[TransformStrategy]:
        if name not in cls._registry:
            raise ValueError(f"Dataset '{name}' not found in registry")
        return cls._registry[name]


TransformRegistry.register(Dataset.CIFAR10, CIFAR10Transforms)
TransformRegistry.register(Dataset.Imagenette, ImagenetteTransforms)


class TransformFactory:
    @staticmethod
    def create(dataset: Dataset) -> TransformStrategy:
        transform_class = TransformRegistry.get(dataset)
        return transform_class()
