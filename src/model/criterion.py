from enum import StrEnum

import torch.nn as nn


class Criterion(StrEnum):
    CrossEntropy = "CrossEntropy"


class CriterionFactory:
    @staticmethod
    def create(criterion: Criterion) -> nn.Module:
        match criterion:
            case Criterion.CrossEntropy:
                return nn.CrossEntropyLoss()
            case _:
                raise ValueError(f"Unknown criterion: {criterion}")
