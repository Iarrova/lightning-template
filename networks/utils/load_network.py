from config.config import Config
from networks.constants import Networks


def load_network(config: Config, num_classes: int):
    if config.network == Networks.RESNET50:
        from networks.resnet50 import ResNet50 as network
    elif config.network == Networks.EFFICIENTNETV2:
        from networks.efficientnetv2 import EfficientNetV2 as network
    elif config.network == Networks.VISION_TRANSFORMER:
        from networks.vision_transformer import VisionTransformer as network
    else:
        print("[ERROR] Invalid network passed in configuration file. Exiting...")
        exit(1)

    model: network = network(
        include_top=config.include_top, weights=config.weights, num_classes=num_classes
    )

    return model
