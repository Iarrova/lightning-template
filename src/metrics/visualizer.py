from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from torchmetrics.classification import ROC, ConfusionMatrix


class MetricsVisualizer:
    @staticmethod
    def plot_confusion_matrix(
        confusion_matrix: ConfusionMatrix,
        class_names: List[str],
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(16, 10))

        cm = confusion_matrix.compute().cpu().numpy()
        cm = cm / cm.sum(axis=1, keepdims=True)

        sns.heatmap(
            cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax
        )

        ax.tick_params(axis="x", labelrotation=45)
        ax.set_xlabel("Predicted Labels")
        ax.set_ylabel("True Labels")
        ax.set_title("Normalized Confusion Matrix")

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=300)

        return fig

    @staticmethod
    def plot_roc_curve(roc: ROC, class_names: List[str], save_path: Optional[str] = None) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(16, 10))

        fpr, tpr, _ = roc.compute()

        for i, class_name in enumerate(class_names):
            ax.plot(
                fpr[i].cpu().numpy(),
                tpr[i].cpu().numpy(),
                lw=2,
                label=f"{class_name} (Class {i})",
            )

        ax.plot([0, 1], [0, 1], "k--", lw=2)
        ax.set_xlim((0.0, 1.0))
        ax.set_ylim((0.0, 1.05))
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver Operating Characteristic (ROC) Curves")
        ax.legend(loc="lower right")

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=300)

        return fig

    @staticmethod
    def plot_training_metrics(metrics_file: str, save_path: Optional[str] = None) -> plt.Figure:
        metrics_df = pd.read_csv(metrics_file)

        train_metrics = {col for col in metrics_df.columns if col.startswith("train_")}
        val_metrics = {col for col in metrics_df.columns if col.startswith("val_")}

        train_metric_names = {metric.split("train_")[1] for metric in train_metrics if "step" not in metric}
        val_metric_names = {metric.split("val_")[1] for metric in val_metrics if "step" not in metric}
        common_metrics = train_metric_names.intersection(val_metric_names)

        n_metrics = len(common_metrics) + 1  # +1 for loss
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 10))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        ax = axes[0]
        ax.plot(metrics_df["epoch"], metrics_df["train_loss"], label="Train")
        ax.plot(metrics_df["epoch"], metrics_df["val_loss"], label="Validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss")
        ax.legend()
        ax.grid(alpha=0.3)

        for i, metric in enumerate(sorted(common_metrics)):
            ax = axes[i + 1]
            ax.plot(metrics_df["epoch"], metrics_df[f"train_{metric}"], label="Train")
            ax.plot(metrics_df["epoch"], metrics_df[f"val_{metric}"], label="Validation")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric.capitalize())
            ax.set_title(metric.capitalize())
            ax.legend()
            ax.grid(alpha=0.3)

        for i in range(n_metrics, len(axes)):
            axes[i].axis("off")

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=300)

        return fig
