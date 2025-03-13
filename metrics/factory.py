from typing import Optional

from torchmetrics import MetricCollection
from torchmetrics.classification import (
    AUROC,
    ROC,
    Accuracy,
    ConfusionMatrix,
    F1Score,
    Precision,
    Recall,
    Specificity,
)


class MetricsFactory:
    @staticmethod
    def create_classification_metrics(
        num_classes: int, prefix: str = ""
    ) -> MetricCollection:
        task = "multiclass" if num_classes > 2 else "binary"

        metrics = {
            "accuracy": Accuracy(task=task, num_classes=num_classes),
            "precision": Precision(task=task, num_classes=num_classes, average="macro"),
            "recall": Recall(task=task, num_classes=num_classes, average="macro"),
            "f1": F1Score(task=task, num_classes=num_classes, average="macro"),
            # Be aware AUC can't be used on multilabel
            # If we integrate multilabel, we can check for task type with an if, and add AUC only for binary and multiclass
            "auc": AUROC(task=task, num_classes=num_classes),
        }

        if task == "binary":
            metrics["specificity"] = Specificity(task=task)

        return MetricCollection(metrics, prefix=prefix)

    @staticmethod
    def create_confusion_matrix(
        num_classes: int, normalize: Optional[str] = None
    ) -> ConfusionMatrix:
        return ConfusionMatrix(
            task="multiclass",
            num_classes=num_classes,
            normalize=normalize,
        )

    @staticmethod
    def create_roc_curve(num_classes: int) -> ROC:
        return ROC(task="multiclass", num_classes=num_classes)
