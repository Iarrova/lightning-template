import lightning as L
import torch.nn as nn

from config.config import Config
from metrics.factory import MetricsFactory
from model.optimizers import OptimizerFactory
from model.schedulers import SchedulerFactory


class Model(L.LightningModule):
    def __init__(self, network: nn.Module, config: Config, num_classes: int):
        super().__init__()
        self.network = network
        self.criterion = nn.CrossEntropyLoss()
        self.config: Config = config
        self.num_classes = num_classes

        self._create_metrics()
        # TODO: See impact of this! self.save_hyperparameters(ignore=["network"])

    def _create_metrics(self):
        self.metrics = nn.ModuleDict(
            {
                "_train": MetricsFactory.create_classification_metrics(self.num_classes, prefix="train_"),
                "val": MetricsFactory.create_classification_metrics(self.num_classes, prefix="val_"),
                "test": MetricsFactory.create_classification_metrics(self.num_classes, prefix="test_"),
            }
        )

        self.confusion_matrix = MetricsFactory.create_confusion_matrix(self.num_classes)
        self.roc_curve = MetricsFactory.create_roc_curve(self.num_classes)

    def forward(self, batch):
        return self.network(batch)

    def step(self, batch):
        inputs, target = batch
        output = self.network(inputs)
        loss = self.criterion(output, target)
        return loss, output, target

    def _shared_step(self, batch, stage: str):
        loss, output, target = self.step(batch)
        metrics = self.metrics[stage]
        metrics.update(output, target)

        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "_train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        loss, output, target = self.step(batch)

        self.metrics["test"].update(output, target)
        self.confusion_matrix.update(output, target)
        self.roc_curve.update(output, target)

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(self.metrics["test"], on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = OptimizerFactory.create(
            self.parameters(),
            self.config.training.optimizer,
            self.config.training.learning_rate,
        )
        scheduler = SchedulerFactory.create(
            optimizer,
            self.config.training.scheduler,
            patience=self.config.training.scheduler_patience,
            factor=self.config.training.scheduler_factor,
            monitor="val_loss",
        )

        return {"optimizer": optimizer, **scheduler}
