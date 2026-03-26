import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from typing import Optional, Dict, Any

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


class ClassificationMetrics:
    """
    Computes classification metrics and provides plotting utilities
    for confusion matrix and classification report.

    Attributes:
        y_true (np.ndarray or list): True labels.
        y_pred (np.ndarray or list): Predicted labels.
        y_proba (Optional[np.ndarray or list]): Predicted probabilities for class 1.
    """

    def __init__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ):
        """
        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            y_proba: Optional predicted probabilities for positive class (for ROC-AUC).
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_proba = np.array(y_proba) if y_proba is not None else None

        self.cm = confusion_matrix(self.y_true, self.y_pred)
        self.tn, self.fp, self.fn, self.tp = self.cm.ravel()

    def roc_auc(self) -> Optional[float]:
        """Compute ROC-AUC score, rounded to 2 decimals. Returns None if probabilities are not provided."""
        if self.y_proba is None:
            return None
        return round(roc_auc_score(self.y_true, self.y_proba), 2)

    def f1_macro(self) -> float:
        """Compute F1 macro score, rounded to 2 decimals."""
        return round(f1_score(self.y_true, self.y_pred, average="macro"), 2)

    def sensitivity(self) -> float:
        """Compute sensitivity (recall for positive class), rounded to 2 decimals."""
        return round(self.tp / (self.tp + self.fn), 2)

    def specificity(self) -> float:
        """Compute specificity (recall for negative class), rounded to 2 decimals."""
        return round(self.tn / (self.tn + self.fp), 2)

    def get_metrics(self) -> Dict[str, Optional[float]]:
        """
        Return all metrics as a dictionary with values rounded to 2 decimals.

        Returns:
            Dictionary with keys: 'roc_auc', 'f1_macro', 'sensitivity', 'specificity'.
        """
        return {
            "roc_auc": self.roc_auc(),
            "f1_macro": self.f1_macro(),
            "sensitivity": self.sensitivity(),
            "specificity": self.specificity(),
        }

    def plot_confusion_matrix(
        self, class_names: Optional[list] = None, return_fig: bool = True
    ) -> Optional[plt.Figure]:
        """
        Plot the confusion matrix.

        Args:
            class_names: Optional list of class names (default ["Malignant", "Benign"]).
            return_fig: If True, returns matplotlib Figure object; else shows plot.

        Returns:
            Matplotlib Figure if return_fig=True, else None.
        """
        if class_names is None:
            class_names = ["Malignant", "Benign"]

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            self.cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Confusion Matrix")
        fig.tight_layout()

        if return_fig:
            return fig
        else:
            plt.show()
            return None

    def plot_classification_report(
        self, print_df: bool = True, return_fig: bool = True
    ) -> Optional[plt.Figure]:
        """
        Plot the classification report as a table.

        Args:
            print_df: If True, prints the dataframe with metrics.
            return_fig: If True, returns matplotlib Figure object; else shows plot.

        Returns:
            Matplotlib Figure if return_fig=True, else None.
        """
        report: Dict[str, Any] = classification_report(
            self.y_true, self.y_pred, output_dict=True, zero_division=0
        )
        df: pd.DataFrame = pd.DataFrame(report).transpose()
        df = df.round(2)

        if print_df:
            display(df)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")

        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            rowLabels=df.index,
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(len(df.columns))))

        ax.set_title("Classification Report")
        fig.tight_layout()

        if return_fig:
            return fig
        else:
            plt.show()
            return None
