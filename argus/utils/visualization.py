"""
Visualization Utilities.

Plotting functions for model evaluation and analysis.

Copyright 2024-2025 VTP Consortium
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def plot_roc_curve(
    y_true: NDArray[np.integer[Any]],
    y_score: NDArray[np.floating[Any]],
    title: str = "ROC Curve",
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (8, 6),
    show_ci: bool = False,
    n_bootstrap: int = 100,
) -> dict[str, Any]:
    """
    Plot ROC curve with optional confidence intervals.

    Args:
        y_true: True binary labels.
        y_score: Predicted probabilities.
        title: Plot title.
        save_path: Path to save figure.
        figsize: Figure size.
        show_ci: Whether to show confidence intervals.
        n_bootstrap: Number of bootstrap samples for CI.

    Returns:
        Dictionary with ROC curve data (fpr, tpr, auc).

    Example:
        >>> result = plot_roc_curve(y_true, y_pred, save_path='roc.png')
        >>> print(f"AUC: {result['auc']:.3f}")
    """
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc
    except ImportError:
        raise ImportError(
            "matplotlib and scikit-learn are required for visualization. "
            "Install with: pip install matplotlib scikit-learn"
        )

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot ROC curve
    ax.plot(
        fpr, tpr,
        color="darkorange",
        lw=2,
        label=f"ROC curve (AUC = {roc_auc:.3f})"
    )

    # Confidence interval
    if show_ci:
        ci_lower, ci_upper = _bootstrap_roc_ci(
            y_true, y_score, n_bootstrap=n_bootstrap
        )
        ax.fill_between(
            fpr, ci_lower, ci_upper,
            color="darkorange", alpha=0.2,
            label="95% CI"
        )

    # Reference line
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close()

    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "auc": roc_auc,
    }


def plot_precision_recall_curve(
    y_true: NDArray[np.integer[Any]],
    y_score: NDArray[np.floating[Any]],
    title: str = "Precision-Recall Curve",
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (8, 6),
) -> dict[str, Any]:
    """
    Plot precision-recall curve.

    Args:
        y_true: True binary labels.
        y_score: Predicted probabilities.
        title: Plot title.
        save_path: Path to save figure.
        figsize: Figure size.

    Returns:
        Dictionary with PR curve data.

    Example:
        >>> result = plot_precision_recall_curve(y_true, y_pred)
        >>> print(f"AUPRC: {result['auprc']:.3f}")
    """
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve, average_precision_score
    except ImportError:
        raise ImportError(
            "matplotlib and scikit-learn are required for visualization"
        )

    # Compute PR curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    # Baseline (random classifier)
    baseline = y_true.sum() / len(y_true)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        recall, precision,
        color="darkorange",
        lw=2,
        label=f"PR curve (AP = {ap:.3f})"
    )

    ax.axhline(
        y=baseline,
        color="navy",
        lw=2,
        linestyle="--",
        label=f"Baseline (prevalence = {baseline:.3f})"
    )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close()

    return {
        "precision": precision,
        "recall": recall,
        "thresholds": thresholds,
        "auprc": ap,
    }


def plot_confusion_matrix(
    y_true: NDArray[np.integer[Any]],
    y_pred: NDArray[np.integer[Any]],
    title: str = "Confusion Matrix",
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (8, 6),
    normalize: bool = True,
    cmap: str = "Blues",
) -> NDArray[np.integer[Any]]:
    """
    Plot confusion matrix.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        title: Plot title.
        save_path: Path to save figure.
        figsize: Figure size.
        normalize: Whether to normalize the matrix.
        cmap: Colormap name.

    Returns:
        Confusion matrix array.

    Example:
        >>> cm = plot_confusion_matrix(y_true, y_pred, save_path='cm.png')
    """
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
    except ImportError:
        raise ImportError(
            "matplotlib and scikit-learn are required for visualization"
        )

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm_display = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
    else:
        cm_display = cm
        fmt = "d"

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(cm_display, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # Labels
    classes = ["Negative", "Positive"]
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True label",
        xlabel="Predicted label"
    )

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm_display.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm_display[i, j], fmt),
                ha="center", va="center",
                color="white" if cm_display[i, j] > thresh else "black"
            )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close()

    return cm


def plot_calibration_curve(
    y_true: NDArray[np.integer[Any]],
    y_prob: NDArray[np.floating[Any]],
    n_bins: int = 10,
    title: str = "Calibration Curve",
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 4),
) -> dict[str, Any]:
    """
    Plot calibration curve (reliability diagram).

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities.
        n_bins: Number of calibration bins.
        title: Plot title.
        save_path: Path to save figure.
        figsize: Figure size.

    Returns:
        Dictionary with calibration data.

    Example:
        >>> result = plot_calibration_curve(y_true, y_prob)
        >>> print(f"ECE: {result['ece']:.3f}")
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for visualization")

    from argus.evaluation.calibration import (
        calibration_curve,
        expected_calibration_error,
    )

    # Compute calibration curve
    prob_true, prob_pred, bin_counts = calibration_curve(
        y_true, y_prob, n_bins=n_bins
    )
    ece = expected_calibration_error(y_true, y_prob, n_bins=n_bins)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Calibration curve
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax1.plot(prob_pred, prob_true, "o-", color="darkorange", label=f"Model (ECE={ece:.3f})")

    ax1.set_xlabel("Mean predicted probability")
    ax1.set_ylabel("Fraction of positives")
    ax1.set_title("Reliability Diagram")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Histogram of predictions
    ax2.hist(y_prob, bins=n_bins, range=(0, 1), edgecolor="black", alpha=0.7)
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Count")
    ax2.set_title("Prediction Distribution")
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close()

    return {
        "prob_true": prob_true,
        "prob_pred": prob_pred,
        "bin_counts": bin_counts,
        "ece": ece,
    }


def plot_feature_importance(
    importance_scores: NDArray[np.floating[Any]],
    feature_names: list[str] | None = None,
    top_k: int = 20,
    title: str = "Feature Importance",
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 8),
    horizontal: bool = True,
) -> None:
    """
    Plot feature importance scores.

    Args:
        importance_scores: Feature importance scores.
        feature_names: Names for features.
        top_k: Number of top features to show.
        title: Plot title.
        save_path: Path to save figure.
        figsize: Figure size.
        horizontal: Whether to use horizontal bars.

    Example:
        >>> plot_feature_importance(
        ...     importance_scores=shap_values,
        ...     feature_names=feature_names,
        ...     top_k=15
        ... )
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for visualization")

    n_features = len(importance_scores)

    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(n_features)]

    # Get top k features by absolute importance
    abs_importance = np.abs(importance_scores)
    top_indices = np.argsort(abs_importance)[-top_k:]

    # Sort for display
    sorted_indices = top_indices[np.argsort(abs_importance[top_indices])]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    names = [feature_names[i] for i in sorted_indices]
    values = importance_scores[sorted_indices]

    colors = ["green" if v > 0 else "red" for v in values]

    if horizontal:
        ax.barh(range(len(values)), values, color=colors, edgecolor="black")
        ax.set_yticks(range(len(values)))
        ax.set_yticklabels(names)
        ax.set_xlabel("Importance Score")
        ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
    else:
        ax.bar(range(len(values)), values, color=colors, edgecolor="black")
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("Importance Score")
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="x" if horizontal else "y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close()


def plot_training_history(
    history: dict[str, list[float]],
    metrics: list[str] | None = None,
    title: str = "Training History",
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (12, 4),
) -> None:
    """
    Plot training history curves.

    Args:
        history: Dictionary mapping metric names to lists of values.
        metrics: List of metrics to plot. If None, plot all.
        title: Plot title.
        save_path: Path to save figure.
        figsize: Figure size.

    Example:
        >>> history = {'train_loss': [...], 'val_loss': [...], 'val_auroc': [...]}
        >>> plot_training_history(history, metrics=['train_loss', 'val_loss'])
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for visualization")

    if metrics is None:
        metrics = list(history.keys())

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        if metric in history:
            values = history[metric]
            epochs = range(1, len(values) + 1)

            ax.plot(epochs, values, "o-", linewidth=2, markersize=4)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric)
            ax.set_title(metric)
            ax.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close()


def _bootstrap_roc_ci(
    y_true: NDArray[np.integer[Any]],
    y_score: NDArray[np.floating[Any]],
    n_bootstrap: int = 100,
    confidence_level: float = 0.95,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Compute bootstrap confidence intervals for ROC curve."""
    from sklearn.metrics import roc_curve

    n_samples = len(y_true)
    rng = np.random.default_rng(42)

    # Get base FPR grid
    base_fpr = np.linspace(0, 1, 100)
    tpr_samples = []

    for _ in range(n_bootstrap):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_score_boot = y_score[indices]

        # Skip if all same class
        if len(np.unique(y_true_boot)) < 2:
            continue

        fpr, tpr, _ = roc_curve(y_true_boot, y_score_boot)

        # Interpolate to common FPR grid
        tpr_interp = np.interp(base_fpr, fpr, tpr)
        tpr_samples.append(tpr_interp)

    if len(tpr_samples) < 2:
        return base_fpr, base_fpr

    tpr_samples = np.array(tpr_samples)

    alpha = (1 - confidence_level) / 2
    ci_lower = np.percentile(tpr_samples, 100 * alpha, axis=0)
    ci_upper = np.percentile(tpr_samples, 100 * (1 - alpha), axis=0)

    return ci_lower, ci_upper
