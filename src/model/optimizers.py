from enum import StrEnum

import torch
import torch.optim as optim


class Optimizer(StrEnum):
    ADAM = "Adam"


class OptimizerFactory:
    @staticmethod
    def create(params, optimizer: Optimizer, learning_rate: float, **kwargs) -> torch.optim.Optimizer:
        if optimizer == Optimizer.ADAM:
            return optim.Adam(params, lr=learning_rate, **kwargs)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
