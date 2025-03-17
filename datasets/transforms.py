from typing import Tuple

import torch
from torchvision.transforms import v2

from datasets.enums import Datasets


class TransformStrategy:
    def get_transforms(self, augment: bool = True) -> Tuple[v2.Compose, v2.Compose]:
        raise NotImplementedError


class CIFAR10Transforms(TransformStrategy):
    def get_transforms(self, augment: bool = True) -> Tuple[v2.Compose, v2.Compose]:
        normalize = [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        transform_train = (
            v2.Compose([v2.RandomHorizontalFlip(p=0.5)] + normalize) if augment else v2.Compose(normalize)
        )
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

        transform_train = (
            v2.Compose([v2.RandomHorizontalFlip(p=0.5)] + normalize) if augment else v2.Compose(normalize)
        )
        transform_test = v2.Compose(normalize)

        return transform_train, transform_test


class TransformFactory:
    @staticmethod
    def create(dataset_name: Datasets) -> TransformStrategy:
        if dataset_name == Datasets.CIFAR10:
            return CIFAR10Transforms()
        elif dataset_name == Datasets.Imagenette:
            return ImagenetteTransforms()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
