import os
from argparse import ArgumentParser

import lightning as L
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from config.config import Config
from config.json_parser import ConfigParser
from datasets.factory import DatasetFactory
from metrics.visualizer import MetricsVisualizer
from model.model import Model
from networks.factory import NetworkFactory


def train(config: Config) -> None:
    L.seed_everything(42)

    if torch.cuda.is_available():
        print("[INFO] CUDA is available! Training on GPU...")
        print(f"[INFO] Using {torch.cuda.device_count()} GPU(s)")
        for i in range(torch.cuda.device_count()):
            print(f"[INFO] GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("[INFO] CUDA is not available. Training on CPU...")

    dataset = DatasetFactory.create(
        name=config.dataset.dataset,
        batch_size=config.training.batch_size,
        validation_size=config.training.validation_size,
        augment=config.dataset.augment,
        num_workers=config.dataset.num_workers,
    )

    train_loader, validation_loader = dataset.generate_train_loaders()

    network = NetworkFactory.create(
        name=config.network.network,
        include_top=config.network.include_top,
        weights=config.network.pytorch_weights,
        num_classes=dataset.NUM_CLASSES,
    )

    model = Model(network=network, config=config, num_classes=dataset.NUM_CLASSES)

    log_dir = os.path.join(config.logging.log_dir, config.logging.weights_path)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(config.logging.weights_dir, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=config.logging.weights_dir,
            filename=config.logging.weights_path,
            monitor="val_loss",
            mode="min",
            verbose=True,
        ),
        EarlyStopping(
            monitor="val_loss", mode="min", patience=config.training.early_stopping_patience, verbose=True
        ),
        LearningRateMonitor(logging_interval="epoch"),
        RichModelSummary(max_depth=3),
        RichProgressBar(),
    ]

    loggers = []
    if config.logging.tensorboard:
        loggers.append(TensorBoardLogger(save_dir=log_dir, name="tensorboard"))
    if config.logging.csv:
        loggers.append(CSVLogger(save_dir=log_dir, name="csv_logs"))

    trainer = L.Trainer(
        max_epochs=config.training.num_epochs,
        callbacks=callbacks,
        accelerator="auto",
        devices="auto",
        logger=loggers,
        precision="16-mixed" if config.mixed_precision and torch.cuda.is_available() else "32",
        gradient_clip_val=1.0,
        log_every_n_steps=50,
        deterministic=True,
    )

    print(f"[INFO] Training model with {dataset.NUM_CLASSES} classes")
    print(f"[INFO] Training on {len(train_loader.dataset)} samples")
    print(f"[INFO] Validating on {len(validation_loader.dataset)} samples")
    print(f"[INFO] Using batch size {config.training.batch_size}")
    print(f"[INFO] Using learning rate {config.training.learning_rate}")
    print(f"[INFO] Using network {config.network.network}")
    print(f"[INFO] Using weights {config.network.pytorch_weights}")
    print(f"[INFO] Using optimizer {config.training.optimizer}")
    print(f"[INFO] Using scheduler {config.training.scheduler}")
    print(f"[INFO] Using mixed precision: {config.mixed_precision}")

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=validation_loader)

    print(f"[INFO] Best model path: {trainer.checkpoint_callback.best_model_path}")
    print(f"[INFO] Best model score: {trainer.checkpoint_callback.best_model_score:.4f}")

    if config.logging.csv:
        metrics_file = os.path.join(log_dir, "csv_logs", "metrics.csv")
        if os.path.exists(metrics_file):
            print("[INFO] Visualizing training metrics...")
            fig = MetricsVisualizer.plot_training_metrics(
                metrics_file=metrics_file,
                save_path=os.path.join(log_dir, "training_metrics.png"),
            )

    return trainer.checkpoint_callback.best_model_path


def main():
    parser = ArgumentParser(description="Training script for image classification.")
    parser.add_argument(
        "--config-path",
        required=True,
        help="Path to the configuration file.",
    )
    args = parser.parse_args()

    config = ConfigParser.parse_json(args.config_path)

    train(config)


if __name__ == "__main__":
    main()
