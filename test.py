import os
from argparse import ArgumentParser

import lightning as L
import torch
from lightning.pytorch.loggers import CSVLogger

from config.config import Config
from config.json_parser import ConfigParser
from datasets.factory import DatasetFactory
from metrics.visualization import MetricsVisualizer
from model.model import Model
from networks.factory import NetworkFactory


def test(config: Config, checkpoint_path: str = None):
    L.seed_everything(config.seed)

    if torch.cuda.is_available():
        print("[INFO] CUDA is available! Testing on GPU...")
    else:
        print("[INFO] CUDA is not available. Testing on CPU...")

    dataset = DatasetFactory.create(
        name=config.dataset.dataset,
        batch_size=config.training.batch_size,
        validation_size=config.training.validation_size,
        augment=False,
        num_workers=config.dataset.num_workers,
    )

    test_loader = dataset.generate_test_loader()

    network = NetworkFactory.create(
        name=config.network.network,
        include_top=config.network.include_top,
        weights=None,
        num_classes=dataset.NUM_CLASSES,
    )

    if checkpoint_path is None:
        checkpoint_path = os.path.join(
            config.logging.weights_dir,
            f"{config.logging.weights_path}.ckpt",
        )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    model = Model.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        network=network,
        config=config,
        num_classes=dataset.NUM_CLASSES,
    )

    log_dir = os.path.join(config.logging.log_dir, config.logging.weights_path)
    os.makedirs(log_dir, exist_ok=True)

    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        logger=CSVLogger(save_dir=os.path.join(log_dir, "test_results")),
    )

    print(f"[INFO] Testing model from {checkpoint_path}")
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
    parser.add_argument(
        "--checkpoint-path",
        help="Path to model checkpoint. If not provided, uses the default path from config.",
    )
    args = parser.parse_args()

    config = ConfigParser.parse_json(args.config_path)

    test(config, args.checkpoint_path)


if __name__ == "__main__":
    main()
