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

        self.network = self._create_network()
        self.criterion = self._create_criterion()
        self.metrics = self._create_metrics()

        self.save_hyperparameters()

    def _create_metrics(self):
        return nn.ModuleDict(
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
        return create_network(self.config.network, self.num_classes)

    def _create_criterion(self):
        return CriterionFactory.create(self.config.training.criterion)

    def forward(self, x):
        return self.network(x)

    def _compute_step(self, batch):
        inputs, targets = batch
        outputs = self.network(inputs)
        loss = self.criterion(outputs, targets)
        return loss, outputs, targets

    def _shared_step(self, batch, stage: str):
        loss, outputs, targets = self._compute_step(batch)

        self.metrics[stage].update(outputs, targets)
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log_dict(self.metrics[stage], prog_bar=True, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "_train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        loss, outputs, targets = self._compute_step(batch)

        self.metrics["test"].update(outputs, targets)
        # self.confusion_matrix.update(output, target)
        # self.roc_curve.update(output, target)

        self.log("test_loss", loss)
        self.log_dict(self.metrics["test"], on_step=False, on_epoch=True)

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
