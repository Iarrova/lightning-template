import argparse

import torch
import torch.nn as nn
import torch.optim as optim

import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)

from torchmetrics.classification import Accuracy

from utils.json_parser import parse_json

parser = argparse.ArgumentParser(
    description="Template for PyTorch Lightning prototyping."
)
parser.add_argument(
    "--config_path",
    help="The path to the configuration file to use in training/testing.",
)
args = parser.parse_args()

config = parse_json(args.config_path)

L.seed_everything(config.seed)
if torch.cuda.is_available():
    print("[INFO] CUDA is available! Training on GPU...")
else:
    print("[INFO] CUDA is not available. Training on CPU...")


# Dataset
if config.dataset == "CIFAR10":
    from datasets.CIFAR10 import generate_CIFAR10 as generate_dataset

    num_classes = 10
else:
    print("[ERROR] Currently only CIFAR10 dataset is supported. Exiting...")
    exit(1)

train_loader, validation_loader, test_loader, classes = generate_dataset(
    batch_size=config.batch_size,
    validation_size=config.validation_size,
    augment=config.augment,
)


# Network
if config.network == "ResNet50":
    from networks.resnet50 import ResNet50 as network
else:
    print("[ERROR] Currently only ResNet50 network is supported. Exiting...")

model = network(include_top=False, weights="imagenet", num_classes=num_classes)


# PyTorch Ligtning
class LitModel(L.LightningModule):
    def __init__(self, model, criterion):
        super().__init__()
        self.model = model
        self.criterion = criterion

        # Metrics
        self.train_accuracy = Accuracy(task="multiclass", num_classes=len(classes))
        self.validation_accuracy = Accuracy(task="multiclass", num_classes=len(classes))

    def step(self, batch):
        inputs, target = batch
        output = self.model(inputs)
        loss = self.criterion(output, target)
        return loss, output, target

    def training_step(self, batch, batch_idx):
        loss, output, target = self.step(batch)
        self.train_accuracy.update(output, target)
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_acc", self.train_accuracy, on_step=False, on_epoch=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, output, target = self.step(batch)
        self.validation_accuracy.update(output, target)
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "val_acc",
            self.validation_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=config.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=5, cooldown=1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }


criterion = nn.CrossEntropyLoss()
litmodel = LitModel(model, criterion)

# Callbacks
callbacks = [
    ModelCheckpoint(
        dirpath=config.weights_dir,
        filename=config.weights_path,
        monitor="val_loss",
        mode="min",
        verbose=True,
    ),
    EarlyStopping(monitor="val_loss", mode="min", patience=10, verbose=True),
    LearningRateMonitor(),
]

trainer = L.Trainer(
    max_epochs=config.num_epochs,
    callbacks=callbacks,
    accelerator="auto",
    devices="auto",
    deterministic=True,
)

trainer.fit(
    model=litmodel, train_dataloaders=train_loader, val_dataloaders=validation_loader
)
