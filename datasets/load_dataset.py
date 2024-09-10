from utils.config import Config


def load_dataset(config: Config):
    if config.dataset == "CIFAR10":
        from datasets.CIFAR10 import generate_CIFAR10 as generate_dataset

        num_classes = 10
    elif config.dataset == "Upscaled-CIFAR10":
        from datasets.Upscaled_CIFAR10 import (
            generate_upscaled_CIFAR10 as generate_dataset,
        )

        num_classes = 10
    elif config.dataset == "CIFAR100":
        from datasets.CIFAR100 import generate_CIFAR100 as generate_dataset

        num_classes = 100

    else:
        print(
            "[ERROR] Currently only CIFAR10, Upscaled-CIFAR10 and CIFAR100 datasets are supported. Exiting..."
        )
        exit(1)

    train_loader, validation_loader, test_loader, classes = generate_dataset(
        batch_size=config.batch_size,
        validation_size=config.validation_size,
        augment=config.augment,
    )

    return train_loader, validation_loader, test_loader, classes, num_classes
