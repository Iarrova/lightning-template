import lightning as L
import torch.nn as nn
import torch.optim as optim
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    AUROC,
    Accuracy,
    ConfusionMatrix,
    Precision,
    Recall,
)

from config.config import Config


class Model(L.LightningModule):
    def __init__(self, network, config, num_classes):
        super().__init__()
        self.network = network
        self.criterion = nn.CrossEntropyLoss()
        self.config: Config = config

        self.metrics = nn.ModuleDict(
            {
                "_train": self._create_metrics(num_classes, prefix="train_"),
                "val": self._create_metrics(num_classes, prefix="val_"),
                "test": self._create_metrics(num_classes, prefix="test_"),
            }
        )

        self.confusion_matrix = ConfusionMatrix(
            task="multiclass", num_classes=num_classes
        )

    def _create_metrics(self, num_classes: int, prefix: str):
        return MetricCollection(
            {
                "accuracy": Accuracy(task="multiclass", num_classes=num_classes),
                "precision": Precision(
                    task="multiclass", num_classes=num_classes, average="macro"
                ),
                "recall": Recall(
                    task="multiclass", num_classes=num_classes, average="macro"
                ),
                "auc": AUROC(task="multiclass", num_classes=num_classes),
            },
            prefix=prefix,
        )

    def step(self, batch):
        inputs, target = batch
        output = self.network(inputs)
        loss = self.criterion(output, target)
        return loss, output, target

    def _shared_step(self, batch, stage: str):
        loss, output, target = self.step(batch)
        metrics = self.metrics[stage]
        metrics.update(output, target)

        self.log(
            f"{stage}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "_train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=5, cooldown=1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
