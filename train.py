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
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from src.config import Config
from src.datasets import create_dataset
from src.model.model import Model


def train(config: Config) -> None:
    L.seed_everything(config.seed)

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
            num_classes=dataset.num_classes,
        )
    else:
        print("[INFO] Creating new model instance")
        model = Model(config=config, num_classes=dataset.num_classes)

    callbacks = [
        ModelCheckpoint(
            dirpath="./weights",
            filename=f"{config.project_name}",
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
        loggers.append(TensorBoardLogger(save_dir=f"./logs/{config.project_name}", name="tensorboard"))
    if config.logging.wandb:
        loggers.append(
            WandbLogger(
                save_dir=f"./logs/{config.project_name}",
                name=f"{config.project_name}",
                project="Lightning-Starter",
                log_model=True,
            )
        )

    trainer = L.Trainer(
        max_epochs=config.training.num_epochs, callbacks=callbacks, logger=loggers, log_every_n_steps=50
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=validation_loader,
        ckpt_path=config.network.lightning_checkpoint if config.training.resume_training else None,
    )


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
