import os
from argparse import ArgumentParser

import lightning as L
import torch
from lightning.pytorch.loggers import CSVLogger

from src.config.config import Config
from src.config.json_parser import ConfigParser
from src.datasets.factory import DatasetFactory
from src.metrics.visualizer import MetricsVisualizer
from src.model.model import Model
from src.networks.factory import NetworkFactory


def test(config: Config):
    L.seed_everything(config.seed)

    if torch.cuda.is_available():
        print("[INFO] CUDA is available! Testing on GPU...")
    else:
        print("[INFO] CUDA is not available. Testing on CPU...")

    dataset = DatasetFactory.create(config.dataset)
    test_loader = dataset.generate_test_loader()

    network = NetworkFactory.create(config.network, config.weights, dataset.num_classes)

    if not os.path.exists(config.weights.save_weights_path):
        raise FileNotFoundError(f"Checkpoint not found at {config.weights.save_weights_path}")

    model = Model.load_from_checkpoint(
        checkpoint_path=config.weights.save_weights_path,
        network=network,
        config=config,
        num_classes=dataset.NUM_CLASSES,
    )

    log_dir = config.logging.log_dir / config.weights.save_weights_path.parent
    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        logger=CSVLogger(save_dir=os.path.join(log_dir, "test_results")),
    )

    print(f"[INFO] Testing model from {config.weights.save_weights_path}")
    print(f"[INFO] Testing on {len(test_loader.dataset)} samples")
    trainer.test(model=model, dataloaders=test_loader)

    class_mapping = dataset.get_class_mapping()
    class_names = list(class_mapping.keys())

    print("[INFO] Visualizing confusion matrix...")
    fig_cm = MetricsVisualizer.plot_confusion_matrix(
        confusion_matrix=model.confusion_matrix,
        class_names=class_names,
        save_path=os.path.join(log_dir, "confusion_matrix.png"),
    )

    print("[INFO] Visualizing ROC curves...")
    fig_roc = MetricsVisualizer.plot_roc_curve(
        roc=model.roc_curve,
        class_names=class_names,
        save_path=os.path.join(log_dir, "roc_curves.png"),
    )

    print("[INFO] Test results:")
    metrics = model.metrics["test"].compute()
    for name, value in metrics.items():
        print(f"[INFO] {name}: {value:.4f}")


def main():
    parser = ArgumentParser(description="Testing script for image classification.")
    parser.add_argument(
        "--config-path",
        required=True,
        help="Path to the configuration file.",
    )

    args = parser.parse_args()

    config = ConfigParser.parse_json(args.config_path)

    test(config)


if __name__ == "__main__":
    main()
