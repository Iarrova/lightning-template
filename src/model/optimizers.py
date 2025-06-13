from enum import StrEnum

import torch
import torch.optim as optim


class Optimizer(StrEnum):
    ADAM = "Adam"


class OptimizerFactory:
    @staticmethod
    def create(params, optimizer: Optimizer, learning_rate: float) -> torch.optim.Optimizer:
        match optimizer:
            case Optimizer.ADAM:
                return optim.Adam(params, lr=learning_rate)
            case _:
                raise ValueError(f"Unknown optimizer: {optimizer}")
