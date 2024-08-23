from typing import Tuple, Dict, Any

import torch
from torch.utils.data import DataLoader, random_split

from torchvision import datasets
from torchvision.transforms import v2


def generate_CIFAR10(
    batch_size: int = 128,
    validation_size: float = 0.2,
    augment: bool = True,
    num_workers: int = 15,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[Any, int]]:
    normalize = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    transform_train = (
        v2.Compose([v2.RandomHorizontalFlip(p=0.5)] + normalize)
        if augment
        else v2.Compose(normalize)
    )
    transform_test = v2.Compose(normalize)

    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    train_set, validation_set = random_split(
        train_dataset, lengths=[1 - validation_size, validation_size]
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    validation_loader = DataLoader(
        validation_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    class_mapping = train_dataset.class_to_idx

    return train_loader, validation_loader, test_loader, class_mapping
