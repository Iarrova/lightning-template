import lightning as L
import torch.nn as nn

from src.config import Config
from src.model.criterion import CriterionFactory
from src.model.metrics import MetricFactory
from src.model.optimizers import OptimizerFactory
from src.model.schedulers import SchedulerFactory
from src.networks import create_network


class Model(L.LightningModule):
    def __init__(self, config: Config, num_classes: int):
        super().__init__()
        self.config: Config = config
        self.num_classes = num_classes

        self._create_network()
        self._create_criterion()
        self._create_metrics()

    def _create_metrics(self):
        self.metrics = nn.ModuleDict(
            {
                "_train": MetricFactory.create(
                    self.config.training.metrics, self.num_classes, prefix="train_"
                ),
                "val": MetricFactory.create(self.config.training.metrics, self.num_classes, prefix="val_"),
                "test": MetricFactory.create(self.config.training.metrics, self.num_classes, prefix="test_"),
            }
        )

        # self.confusion_matrix = MetricsFactory.create_confusion_matrix(self.num_classes)
        # self.roc_curve = MetricsFactory.create_roc_curve(self.num_classes)

    def _create_network(self):
        self.network = create_network(self.config.network, self.num_classes)

    def _create_criterion(self):
        self.criterion = CriterionFactory.create(self.config.training.criterion)

    def forward(self, x):
        return self.network(x)

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
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
