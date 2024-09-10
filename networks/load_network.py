from utils.config import Config


def load_network(config: Config, num_classes: int):
    if config.network == "ResNet50":
        from networks.resnet50 import ResNet50 as network
    elif config.network == "EfficientNetV2":
        from networks.efficientnetv2 import EfficientNetV2 as network
    else:
        print(
            "[ERROR] Currently only ResNet50 and EfficientNetV2 networks are supported. Exiting..."
        )

    model: network = network(
        include_top=config.include_top, weights=config.weights, num_classes=num_classes
    )

    return model
