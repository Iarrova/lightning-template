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

from src.config import Config
from src.datasets import create_dataset
from src.model.model import Model


def train(config: Config) -> None:
    L.seed_everything(42)

    if torch.cuda.is_available():
        print("[INFO] CUDA is available! Training on GPU...")
        print(f"[INFO] Using {torch.cuda.device_count()} GPU(s)")
        for i in range(torch.cuda.device_count()):
            print(f"[INFO] GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("[INFO] CUDA is not available. Training on CPU...")

    dataset = create_dataset(config.dataset)
    train_loader, validation_loader = dataset.generate_train_loaders()

    if config.network.lightning_checkpoint:
        print(f"[INFO] Loading model from checkpoint: {config.network.lightning_checkpoint}")
        model = Model.load_from_checkpoint(
            checkpoint_path=config.network.lightning_checkpoint,
            config=config,
            num_classes=dataset.NUM_CLASSES,
        )
    else:
        print("[INFO] Creating new model instance")
        model = Model(config=config, num_classes=dataset.NUM_CLASSES)

    callbacks = [
        ModelCheckpoint(
            dirpath=config.save_weights_path.parent,
            filename=config.save_weights_path.stem,
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

    log_dir = config.logging.log_dir / config.save_weights_path.stem
    log_dir.mkdir(parents=True, exist_ok=True)

    loggers = []
    if config.logging.tensorboard:
        loggers.append(TensorBoardLogger(save_dir=log_dir, name="tensorboard"))
    if config.logging.csv:
        loggers.append(CSVLogger(save_dir=log_dir, name="csv_logs"))

    trainer = L.Trainer(
        max_epochs=config.training.num_epochs, callbacks=callbacks, logger=loggers, log_every_n_steps=50
    )

    print(f"[INFO] Training model with {dataset.NUM_CLASSES} classes")
    print(f"[INFO] Training on {len(train_loader.dataset)} samples")
    print(f"[INFO] Validating on {len(validation_loader.dataset)} samples")
    print(f"[INFO] Using batch size {config.dataset.batch_size}")
    print(f"[INFO] Using learning rate {config.training.learning_rate}")
    print(f"[INFO] Using network {config.network.network}")
    print(f"[INFO] Using weights {config.network.pytorch_weights}")
    print(f"[INFO] Using optimizer {config.training.optimizer}")
    print(f"[INFO] Using scheduler {config.training.scheduler}")

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=validation_loader,
        ckpt_path=config.network.lightning_checkpoint if config.training.resume_training else None,
    )

    print(f"[INFO] Best model path: {trainer.checkpoint_callback.best_model_path}")
    print(f"[INFO] Best model score: {trainer.checkpoint_callback.best_model_score:.4f}")

    return trainer.checkpoint_callback.best_model_path


def main():
    parser = ArgumentParser(description="Training script for image classification.")
    parser.add_argument(
        "--config-path",
        required=True,
        help="Path to the configuration file.",
    )
    args = parser.parse_args()

    config = Config.from_json(args.config_path)
    train(config)


if __name__ == "__main__":
    main()
