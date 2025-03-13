from enum import StrEnum

import torch
import torch.optim as optim


class Optimizers(StrEnum):
    ADAM = "Adam"


class OptimizerFactory:
    @staticmethod
    def create(
        params, optimizer: Optimizers, learning_rate: float, **kwargs
    ) -> torch.optim.Optimizer:
        if optimizer == Optimizers.ADAM:
            return optim.Adam(params, lr=learning_rate, **kwargs)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
