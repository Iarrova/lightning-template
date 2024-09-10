from argparse import ArgumentParser, Namespace

import torch
import torch.nn as nn
import torch.optim as optim

import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)

from torchmetrics import MetricCollection
from torchmetrics.classification import (
    Accuracy,
    Precision,
    Recall,
    AUROC,
)

from datasets.load_dataset import load_dataset
from networks.load_network import load_network
from utils.config import Config
from utils.json_parser import parse_json

parser: ArgumentParser = ArgumentParser(
    description="Template for PyTorch Lightning prototyping."
)
parser.add_argument(
    "--config-path",
    help="The path to the configuration file to use in training/testing.",
)
args: Namespace = parser.parse_args()

config: Config = parse_json(args.config_path)

L.seed_everything(config.seed)
if torch.cuda.is_available():
    print("[INFO] CUDA is available! Training on GPU...")
else:
    print("[INFO] CUDA is not available. Training on CPU...")

train_loader, validation_loader, test_loader, classes, num_classes = load_dataset(
    config
)
model = load_network(config, num_classes)


class Model(L.LightningModule):
    def __init__(self, model, criterion):
        super().__init__()
        self.model = model
        self.criterion = criterion

        metrics = MetricCollection(
            {
                "accuracy": Accuracy(task="multiclass", num_classes=num_classes),
                "precision": Precision(
                    task="multiclass", num_classes=num_classes, average="macro"
                ),
                "recall": Recall(
                    task="multiclass", num_classes=num_classes, average="macro"
                ),
                "auc": AUROC(task="multiclass", num_classes=num_classes),
            }
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.validation_metrics = metrics.clone(prefix="val_")

    def step(self, batch):
        inputs, target = batch
        output = self.model(inputs)
        loss = self.criterion(output, target)
        return loss, output, target

    def training_step(self, batch, batch_idx):
        loss, output, target = self.step(batch)
        self.train_metrics.update(output, target)
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log_dict(
            self.train_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, output, target = self.step(batch)
        self.validation_metrics.update(output, target)
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log_dict(
            self.validation_metrics,
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
model = Model(model, criterion)

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
)

trainer.fit(
    model=model, train_dataloaders=train_loader, val_dataloaders=validation_loader
)
