from enum import StrEnum
from typing import Any, Dict, Union

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler


class Scheduler(StrEnum):
    REDUCE_ON_PLATEAU = "ReduceOnPlateau"


class SchedulerFactory:
    @staticmethod
    def create(
        optimizer: torch.optim.Optimizer, scheduler: Scheduler, **kwargs
    ) -> Union[_LRScheduler, Dict[str, Any]]:
        if scheduler == Scheduler.REDUCE_ON_PLATEAU:
            return {
                "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode=kwargs.get("mode", "min"),
                    factor=kwargs.get("factor", 0.1),
                    patience=kwargs.get("patience", 5),
                    cooldown=kwargs.get("cooldown", 1),
                    verbose=True,
                ),
                "monitor": kwargs.get("monitor", "val_loss"),
                "interval": kwargs.get("interval", "epoch"),
                "frequency": kwargs.get("frequency", 1),
            }
        else:
            raise ValueError(f"Unknown scheduler: {scheduler}")
