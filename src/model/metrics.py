from enum import StrEnum

from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)


class Metric(StrEnum):
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"
    AUC = "auc"
    ALL = "all"


class MetricFactory:
    @staticmethod
    def create(metrics: list[Metric], num_classes: int, prefix: str = "") -> MetricCollection:
        metric_map = {
            Metric.ACCURACY: MulticlassAccuracy(num_classes=num_classes),
            Metric.PRECISION: MulticlassPrecision(num_classes=num_classes, average="macro"),
            Metric.RECALL: MulticlassRecall(num_classes=num_classes, average="macro"),
            Metric.F1: MulticlassF1Score(num_classes=num_classes, average="macro"),
            Metric.AUC: MulticlassAUROC(num_classes=num_classes),
        }

        if Metric.ALL in metrics:
            return MetricCollection(metric_map, prefix=prefix)

        selected = {metric: metric_map[metric] for metric in metrics}
        return MetricCollection(selected, prefix=prefix)
