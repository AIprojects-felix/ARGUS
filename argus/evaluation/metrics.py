"""
Evaluation Metrics.

Comprehensive metrics for multi-label genomic prediction evaluation.

Copyright 2024-2025 VTP Consortium
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


def compute_auroc(
    y_true: NDArray,
    y_score: NDArray,
    average: str = "macro",
) -> float | NDArray:
    """
    Compute Area Under the ROC Curve (AUROC).

    Args:
        y_true: True binary labels [n_samples] or [n_samples, n_labels].
        y_score: Predicted probabilities.
        average: Averaging method ('macro', 'micro', 'weighted', None).
            Default: 'macro'

    Returns:
        AUROC score(s).

    Example:
        >>> auroc = compute_auroc(y_true, y_score, average='macro')
    """
    from sklearn.metrics import roc_auc_score

    # Handle single-label case
    if y_true.ndim == 1:
        if len(np.unique(y_true)) < 2:
            return 0.5  # Undefined, return random
        return roc_auc_score(y_true, y_score)

    # Multi-label case
    valid_aucs = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) < 2:
            continue
        try:
            auc = roc_auc_score(y_true[:, i], y_score[:, i])
            valid_aucs.append(auc)
        except ValueError:
            continue

    if not valid_aucs:
        return 0.5

    if average is None:
        return np.array(valid_aucs)
    elif average == "macro":
        return np.mean(valid_aucs)
    elif average == "micro":
        # Compute micro-average across all labels
        return roc_auc_score(y_true.ravel(), y_score.ravel())
    else:
        return np.mean(valid_aucs)


def compute_auprc(
    y_true: NDArray,
    y_score: NDArray,
    average: str = "macro",
) -> float | NDArray:
    """
    Compute Area Under the Precision-Recall Curve (AUPRC).

    AUPRC is more informative than AUROC for imbalanced datasets,
    which is common in genomic prediction tasks.

    Args:
        y_true: True binary labels.
        y_score: Predicted probabilities.
        average: Averaging method ('macro', 'micro', 'weighted', None).
            Default: 'macro'

    Returns:
        AUPRC score(s).

    Example:
        >>> auprc = compute_auprc(y_true, y_score, average='macro')
    """
    from sklearn.metrics import average_precision_score

    # Handle single-label case
    if y_true.ndim == 1:
        if len(np.unique(y_true)) < 2:
            return y_true.mean()  # Prevalence as baseline
        return average_precision_score(y_true, y_score)

    # Multi-label case
    valid_aps = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) < 2:
            continue
        try:
            ap = average_precision_score(y_true[:, i], y_score[:, i])
            valid_aps.append(ap)
        except ValueError:
            continue

    if not valid_aps:
        return y_true.mean()

    if average is None:
        return np.array(valid_aps)
    elif average == "macro":
        return np.mean(valid_aps)
    elif average == "micro":
        return average_precision_score(y_true.ravel(), y_score.ravel())
    else:
        return np.mean(valid_aps)


def compute_f1_score(
    y_true: NDArray,
    y_pred: NDArray,
    average: str = "macro",
    threshold: float = 0.5,
) -> float | NDArray:
    """
    Compute F1 score.

    Args:
        y_true: True binary labels.
        y_pred: Predicted probabilities or binary predictions.
        average: Averaging method ('macro', 'micro', 'weighted', 'samples', None).
            Default: 'macro'
        threshold: Classification threshold if y_pred contains probabilities.
            Default: 0.5

    Returns:
        F1 score(s).
    """
    from sklearn.metrics import f1_score

    # Binarize predictions if needed
    if y_pred.dtype in [np.float32, np.float64]:
        y_pred = (y_pred >= threshold).astype(int)

    return f1_score(y_true, y_pred, average=average, zero_division=0)


def compute_precision_recall(
    y_true: NDArray,
    y_pred: NDArray,
    threshold: float = 0.5,
) -> tuple[float, float]:
    """
    Compute precision and recall.

    Args:
        y_true: True binary labels.
        y_pred: Predicted probabilities or binary predictions.
        threshold: Classification threshold.
            Default: 0.5

    Returns:
        Tuple of (precision, recall).
    """
    from sklearn.metrics import precision_score, recall_score

    # Binarize predictions if needed
    if y_pred.dtype in [np.float32, np.float64]:
        y_pred = (y_pred >= threshold).astype(int)

    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

    return precision, recall


@dataclass
class ClassificationMetrics:
    """
    Container for binary classification metrics.

    Computes comprehensive metrics for a single binary classification task.

    Attributes:
        auroc: Area under ROC curve
        auprc: Area under precision-recall curve
        f1: F1 score
        precision: Precision
        recall: Recall (sensitivity)
        specificity: Specificity
        accuracy: Overall accuracy
        balanced_accuracy: Balanced accuracy
        mcc: Matthews correlation coefficient
        threshold: Optimal threshold
    """

    auroc: float = 0.0
    auprc: float = 0.0
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    specificity: float = 0.0
    accuracy: float = 0.0
    balanced_accuracy: float = 0.0
    mcc: float = 0.0
    threshold: float = 0.5

    @classmethod
    def from_predictions(
        cls,
        y_true: NDArray,
        y_score: NDArray,
        threshold: float | None = None,
    ) -> "ClassificationMetrics":
        """
        Compute metrics from predictions.

        Args:
            y_true: True binary labels [n_samples].
            y_score: Predicted probabilities [n_samples].
            threshold: Classification threshold.
                If None, uses optimal threshold from ROC curve.

        Returns:
            ClassificationMetrics instance.
        """
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            confusion_matrix,
            matthews_corrcoef,
            precision_score,
            recall_score,
            f1_score,
        )

        # Handle constant predictions
        if len(np.unique(y_true)) < 2:
            return cls()

        # Compute AUROC and AUPRC
        auroc = compute_auroc(y_true, y_score)
        auprc = compute_auprc(y_true, y_score)

        # Find optimal threshold if not provided
        if threshold is None:
            threshold = cls._find_optimal_threshold(y_true, y_score)

        # Binarize predictions
        y_pred = (y_score >= threshold).astype(int)

        # Compute confusion matrix
        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).ravel()

        # Compute metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        return cls(
            auroc=auroc,
            auprc=auprc,
            f1=f1_score(y_true, y_pred, zero_division=0),
            precision=precision_score(y_true, y_pred, zero_division=0),
            recall=recall_score(y_true, y_pred, zero_division=0),
            specificity=specificity,
            accuracy=accuracy_score(y_true, y_pred),
            balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
            mcc=matthews_corrcoef(y_true, y_pred),
            threshold=threshold,
        )

    @staticmethod
    def _find_optimal_threshold(
        y_true: NDArray,
        y_score: NDArray,
        metric: str = "youden",
    ) -> float:
        """
        Find optimal classification threshold.

        Args:
            y_true: True labels.
            y_score: Predicted scores.
            metric: Optimization metric ('youden', 'f1', 'precision_recall').

        Returns:
            Optimal threshold.
        """
        from sklearn.metrics import roc_curve, precision_recall_curve

        if metric == "youden":
            # Maximize Youden's J statistic (sensitivity + specificity - 1)
            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            j_scores = tpr - fpr
            best_idx = np.argmax(j_scores)
            return thresholds[best_idx]

        elif metric == "f1":
            # Maximize F1 score
            precision, recall, thresholds = precision_recall_curve(y_true, y_score)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            best_idx = np.argmax(f1_scores[:-1])  # Last threshold is always 1
            return thresholds[best_idx]

        else:
            return 0.5

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "auroc": self.auroc,
            "auprc": self.auprc,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "specificity": self.specificity,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "mcc": self.mcc,
            "threshold": self.threshold,
        }


@dataclass
class MultiLabelMetrics:
    """
    Container for multi-label classification metrics.

    Computes per-label and aggregated metrics for multi-label tasks.

    Example:
        >>> metrics = MultiLabelMetrics.from_predictions(
        ...     y_true, y_score, target_names=['TP53', 'KRAS', ...]
        ... )
        >>> print(f"Mean AUROC: {metrics.mean_auroc:.3f}")
        >>> print(metrics.per_label_metrics['TP53'].auroc)
    """

    # Aggregated metrics
    mean_auroc: float = 0.0
    std_auroc: float = 0.0
    mean_auprc: float = 0.0
    std_auprc: float = 0.0
    mean_f1: float = 0.0
    macro_f1: float = 0.0
    micro_f1: float = 0.0

    # Per-label metrics
    per_label_metrics: dict[str, ClassificationMetrics] = field(default_factory=dict)

    # Label information
    target_names: list[str] = field(default_factory=list)
    n_labels: int = 0
    n_samples: int = 0

    @classmethod
    def from_predictions(
        cls,
        y_true: NDArray,
        y_score: NDArray,
        target_names: list[str] | None = None,
        mask: NDArray | None = None,
    ) -> "MultiLabelMetrics":
        """
        Compute multi-label metrics from predictions.

        Args:
            y_true: True binary labels [n_samples, n_labels].
            y_score: Predicted probabilities [n_samples, n_labels].
            target_names: Names for each label/target.
            mask: Valid label mask [n_samples, n_labels].
                True indicates valid label.

        Returns:
            MultiLabelMetrics instance.
        """
        n_samples, n_labels = y_true.shape

        if target_names is None:
            target_names = [f"target_{i}" for i in range(n_labels)]

        # Compute per-label metrics
        per_label_metrics = {}
        aurocs = []
        auprcs = []

        for i, name in enumerate(target_names):
            # Get valid samples for this label
            if mask is not None:
                valid_mask = mask[:, i] if mask.ndim > 1 else mask
                y_true_i = y_true[valid_mask, i]
                y_score_i = y_score[valid_mask, i]
            else:
                y_true_i = y_true[:, i]
                y_score_i = y_score[:, i]

            # Skip if not enough samples or only one class
            if len(y_true_i) < 10 or len(np.unique(y_true_i)) < 2:
                continue

            # Compute metrics
            label_metrics = ClassificationMetrics.from_predictions(
                y_true_i, y_score_i
            )
            per_label_metrics[name] = label_metrics
            aurocs.append(label_metrics.auroc)
            auprcs.append(label_metrics.auprc)

        # Aggregated metrics
        mean_auroc = np.mean(aurocs) if aurocs else 0.0
        std_auroc = np.std(aurocs) if aurocs else 0.0
        mean_auprc = np.mean(auprcs) if auprcs else 0.0
        std_auprc = np.std(auprcs) if auprcs else 0.0

        # F1 scores
        y_pred = (y_score >= 0.5).astype(int)
        macro_f1 = compute_f1_score(y_true, y_pred, average="macro")
        micro_f1 = compute_f1_score(y_true, y_pred, average="micro")
        mean_f1 = np.mean([m.f1 for m in per_label_metrics.values()]) if per_label_metrics else 0.0

        return cls(
            mean_auroc=mean_auroc,
            std_auroc=std_auroc,
            mean_auprc=mean_auprc,
            std_auprc=std_auprc,
            mean_f1=mean_f1,
            macro_f1=macro_f1,
            micro_f1=micro_f1,
            per_label_metrics=per_label_metrics,
            target_names=target_names,
            n_labels=n_labels,
            n_samples=n_samples,
        )

    def get_top_k_labels(
        self,
        k: int = 10,
        metric: str = "auroc",
        ascending: bool = False,
    ) -> list[tuple[str, float]]:
        """
        Get top-k labels by a specific metric.

        Args:
            k: Number of labels to return.
            metric: Metric to sort by ('auroc', 'auprc', 'f1').
            ascending: Sort in ascending order.

        Returns:
            List of (label_name, metric_value) tuples.
        """
        label_scores = [
            (name, getattr(metrics, metric))
            for name, metrics in self.per_label_metrics.items()
        ]

        label_scores.sort(key=lambda x: x[1], reverse=not ascending)
        return label_scores[:k]

    def to_dataframe(self) -> Any:
        """
        Convert per-label metrics to pandas DataFrame.

        Returns:
            DataFrame with metrics for each label.
        """
        try:
            import pandas as pd

            rows = []
            for name, metrics in self.per_label_metrics.items():
                row = {"target": name, **metrics.to_dict()}
                rows.append(row)

            return pd.DataFrame(rows).sort_values("auroc", ascending=False)
        except ImportError:
            return self.per_label_metrics

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "Multi-Label Classification Metrics",
            "=" * 40,
            f"Samples: {self.n_samples}, Labels: {self.n_labels}",
            f"Mean AUROC: {self.mean_auroc:.4f} ± {self.std_auroc:.4f}",
            f"Mean AUPRC: {self.mean_auprc:.4f} ± {self.std_auprc:.4f}",
            f"Macro F1: {self.macro_f1:.4f}",
            f"Micro F1: {self.micro_f1:.4f}",
            "",
            "Top 10 Labels by AUROC:",
        ]

        for name, score in self.get_top_k_labels(k=10, metric="auroc"):
            lines.append(f"  {name}: {score:.4f}")

        return "\n".join(lines)


def compute_multilabel_metrics_by_group(
    y_true: NDArray,
    y_score: NDArray,
    groups: dict[str, list[int]],
    target_names: list[str] | None = None,
) -> dict[str, MultiLabelMetrics]:
    """
    Compute metrics separately for different target groups.

    Useful for analyzing performance across different gene categories,
    pathways, or biomarker types.

    Args:
        y_true: True labels [n_samples, n_labels].
        y_score: Predicted scores [n_samples, n_labels].
        groups: Dictionary mapping group names to label indices.
        target_names: Names for each label.

    Returns:
        Dictionary mapping group names to MultiLabelMetrics.

    Example:
        >>> groups = {
        ...     'oncogenes': [0, 1, 5, 10],
        ...     'tumor_suppressors': [2, 3, 6, 11],
        ...     'biomarkers': [40, 41, 42],
        ... }
        >>> group_metrics = compute_multilabel_metrics_by_group(
        ...     y_true, y_score, groups
        ... )
    """
    results = {}

    for group_name, indices in groups.items():
        group_y_true = y_true[:, indices]
        group_y_score = y_score[:, indices]

        if target_names is not None:
            group_names = [target_names[i] for i in indices]
        else:
            group_names = [f"target_{i}" for i in indices]

        results[group_name] = MultiLabelMetrics.from_predictions(
            group_y_true, group_y_score, target_names=group_names
        )

    return results
