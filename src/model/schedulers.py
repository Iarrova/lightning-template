from enum import StrEnum

import torch.optim as optim


class Scheduler(StrEnum):
    REDUCE_ON_PLATEAU = "ReduceOnPlateau"


class SchedulerFactory:
    @staticmethod
    def create(optimizer: optim.Optimizer, scheduler: Scheduler, factor: float, patience: int):
        match scheduler:
            case Scheduler.REDUCE_ON_PLATEAU:
                return optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    factor=factor,
                    patience=patience,
                    cooldown=1,
                    verbose="True",
                )
            case _:
                raise ValueError(f"Unknown scheduler: {scheduler}")
