"""
Calibration Analysis Utilities.

Model calibration assessment and visualization for ARGUS predictions.

Copyright 2024-2025 VTP Consortium
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class CalibrationMetrics:
    """
    Calibration metrics for a single prediction target.

    Attributes:
        expected_calibration_error: ECE metric (lower is better).
        maximum_calibration_error: MCE metric (worst-case error).
        brier_score: Brier score (lower is better).
        reliability_bins: Bin-wise reliability statistics.
        n_samples: Number of samples evaluated.
    """
    expected_calibration_error: float
    maximum_calibration_error: float
    brier_score: float
    reliability_bins: dict[str, NDArray[np.floating[Any]]]
    n_samples: int


@dataclass
class CalibrationAnalysis:
    """
    Comprehensive calibration analysis for multi-label predictions.

    Performs calibration assessment across multiple prediction targets
    with support for various calibration metrics and visualization.

    Args:
        n_bins: Number of bins for calibration curve.
            Default: 10
        strategy: Binning strategy ('uniform' or 'quantile').
            Default: 'uniform'

    Attributes:
        per_target_calibration: Calibration metrics per prediction target.
        mean_ece: Mean ECE across all targets.
        mean_mce: Mean MCE across all targets.
        mean_brier: Mean Brier score across all targets.

    Example:
        >>> analysis = CalibrationAnalysis(n_bins=10)
        >>> results = analysis.compute(
        ...     y_true=labels,
        ...     y_prob=predictions
        ... )
        >>> print(f"ECE: {results.mean_ece:.4f}")
    """
    n_bins: int = 10
    strategy: str = "uniform"

    # Results (populated after compute())
    per_target_calibration: dict[str, CalibrationMetrics] = field(
        default_factory=dict, init=False
    )
    mean_ece: float = field(default=0.0, init=False)
    mean_mce: float = field(default=0.0, init=False)
    mean_brier: float = field(default=0.0, init=False)

    def compute(
        self,
        y_true: NDArray[np.integer[Any]],
        y_prob: NDArray[np.floating[Any]],
        target_names: list[str] | None = None,
    ) -> "CalibrationAnalysis":
        """
        Compute calibration metrics for all targets.

        Args:
            y_true: True binary labels of shape (n_samples, n_targets).
            y_prob: Predicted probabilities of shape (n_samples, n_targets).
            target_names: Names for each target (optional).

        Returns:
            Self with computed calibration metrics.
        """
        n_samples, n_targets = y_prob.shape

        if target_names is None:
            target_names = [f"target_{i}" for i in range(n_targets)]

        ece_values = []
        mce_values = []
        brier_values = []

        for i, name in enumerate(target_names):
            y_true_i = y_true[:, i]
            y_prob_i = y_prob[:, i]

            # Compute calibration curve
            prob_true, prob_pred, bin_counts = calibration_curve(
                y_true_i,
                y_prob_i,
                n_bins=self.n_bins,
                strategy=self.strategy,
            )

            # Compute ECE and MCE
            ece = expected_calibration_error(
                y_true_i, y_prob_i, n_bins=self.n_bins, strategy=self.strategy
            )
            mce = maximum_calibration_error(
                y_true_i, y_prob_i, n_bins=self.n_bins, strategy=self.strategy
            )

            # Compute Brier score
            brier = brier_score(y_true_i, y_prob_i)

            self.per_target_calibration[name] = CalibrationMetrics(
                expected_calibration_error=ece,
                maximum_calibration_error=mce,
                brier_score=brier,
                reliability_bins={
                    "prob_true": prob_true,
                    "prob_pred": prob_pred,
                    "bin_counts": bin_counts,
                },
                n_samples=n_samples,
            )

            ece_values.append(ece)
            mce_values.append(mce)
            brier_values.append(brier)

        self.mean_ece = float(np.mean(ece_values))
        self.mean_mce = float(np.mean(mce_values))
        self.mean_brier = float(np.mean(brier_values))

        return self

    def to_dict(self) -> dict[str, Any]:
        """Convert calibration analysis to dictionary format."""
        return {
            "mean_ece": self.mean_ece,
            "mean_mce": self.mean_mce,
            "mean_brier": self.mean_brier,
            "per_target": {
                name: {
                    "ece": cal.expected_calibration_error,
                    "mce": cal.maximum_calibration_error,
                    "brier": cal.brier_score,
                    "n_samples": cal.n_samples,
                }
                for name, cal in self.per_target_calibration.items()
            },
        }


def calibration_curve(
    y_true: NDArray[np.integer[Any]],
    y_prob: NDArray[np.floating[Any]],
    n_bins: int = 10,
    strategy: str = "uniform",
) -> tuple[
    NDArray[np.floating[Any]],
    NDArray[np.floating[Any]],
    NDArray[np.integer[Any]],
]:
    """
    Compute calibration curve (reliability diagram data).

    The calibration curve shows the relationship between predicted
    probabilities and actual frequencies of the positive class.

    Args:
        y_true: True binary labels of shape (n_samples,).
        y_prob: Predicted probabilities of shape (n_samples,).
        n_bins: Number of bins for grouping predictions.
            Default: 10
        strategy: Binning strategy.
            - 'uniform': Equally spaced bins in [0, 1].
            - 'quantile': Bins with equal number of samples.
            Default: 'uniform'

    Returns:
        Tuple of (prob_true, prob_pred, bin_counts):
            - prob_true: Mean true probability in each bin.
            - prob_pred: Mean predicted probability in each bin.
            - bin_counts: Number of samples in each bin.

    Example:
        >>> prob_true, prob_pred, counts = calibration_curve(
        ...     y_true=labels,
        ...     y_prob=predictions,
        ...     n_bins=10
        ... )
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    if strategy == "uniform":
        bins = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        quantiles = np.linspace(0, 100, n_bins + 1)
        bins = np.percentile(y_prob, quantiles)
        bins[0] = 0.0
        bins[-1] = 1.0
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'uniform' or 'quantile'.")

    # Digitize predictions into bins
    bin_indices = np.digitize(y_prob, bins[1:-1])

    prob_true_list = []
    prob_pred_list = []
    bin_counts_list = []

    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        if mask.sum() > 0:
            prob_true_list.append(y_true[mask].mean())
            prob_pred_list.append(y_prob[mask].mean())
            bin_counts_list.append(mask.sum())
        else:
            prob_true_list.append(np.nan)
            prob_pred_list.append(np.nan)
            bin_counts_list.append(0)

    return (
        np.array(prob_true_list),
        np.array(prob_pred_list),
        np.array(bin_counts_list),
    )


def expected_calibration_error(
    y_true: NDArray[np.integer[Any]],
    y_prob: NDArray[np.floating[Any]],
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE is the weighted average of the absolute difference between
    predicted probability and actual frequency across all bins.

    ECE = Σ (|B_m| / n) * |acc(B_m) - conf(B_m)|

    where B_m is bin m, acc is accuracy, and conf is confidence.

    Args:
        y_true: True binary labels of shape (n_samples,).
        y_prob: Predicted probabilities of shape (n_samples,).
        n_bins: Number of bins.
            Default: 10
        strategy: Binning strategy ('uniform' or 'quantile').
            Default: 'uniform'

    Returns:
        Expected Calibration Error (lower is better).

    Example:
        >>> ece = expected_calibration_error(labels, predictions, n_bins=15)
        >>> print(f"ECE: {ece:.4f}")
    """
    prob_true, prob_pred, bin_counts = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy=strategy
    )

    # Remove empty bins
    valid_mask = bin_counts > 0
    prob_true = prob_true[valid_mask]
    prob_pred = prob_pred[valid_mask]
    bin_counts = bin_counts[valid_mask]

    if len(bin_counts) == 0:
        return 0.0

    # Weighted average of calibration errors
    total_samples = bin_counts.sum()
    weights = bin_counts / total_samples
    calibration_errors = np.abs(prob_true - prob_pred)

    return float(np.sum(weights * calibration_errors))


def maximum_calibration_error(
    y_true: NDArray[np.integer[Any]],
    y_prob: NDArray[np.floating[Any]],
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """
    Compute Maximum Calibration Error (MCE).

    MCE is the maximum absolute difference between predicted
    probability and actual frequency across all bins.

    MCE = max_m |acc(B_m) - conf(B_m)|

    Args:
        y_true: True binary labels of shape (n_samples,).
        y_prob: Predicted probabilities of shape (n_samples,).
        n_bins: Number of bins.
            Default: 10
        strategy: Binning strategy ('uniform' or 'quantile').
            Default: 'uniform'

    Returns:
        Maximum Calibration Error (lower is better).

    Example:
        >>> mce = maximum_calibration_error(labels, predictions)
        >>> print(f"MCE: {mce:.4f}")
    """
    prob_true, prob_pred, bin_counts = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy=strategy
    )

    # Remove empty bins
    valid_mask = bin_counts > 0
    prob_true = prob_true[valid_mask]
    prob_pred = prob_pred[valid_mask]

    if len(prob_true) == 0:
        return 0.0

    calibration_errors = np.abs(prob_true - prob_pred)
    return float(np.max(calibration_errors))


def brier_score(
    y_true: NDArray[np.integer[Any]],
    y_prob: NDArray[np.floating[Any]],
) -> float:
    """
    Compute Brier score.

    The Brier score measures the mean squared error of predicted
    probabilities compared to actual outcomes.

    Brier = (1/n) * Σ (p_i - y_i)²

    Args:
        y_true: True binary labels of shape (n_samples,).
        y_prob: Predicted probabilities of shape (n_samples,).

    Returns:
        Brier score (lower is better, range [0, 1]).

    Example:
        >>> score = brier_score(labels, predictions)
        >>> print(f"Brier: {score:.4f}")
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    return float(np.mean((y_prob - y_true) ** 2))


def reliability_diagram(
    y_true: NDArray[np.integer[Any]],
    y_prob: NDArray[np.floating[Any]],
    n_bins: int = 10,
    strategy: str = "uniform",
    title: str = "Reliability Diagram",
) -> dict[str, Any]:
    """
    Generate data for reliability diagram visualization.

    A reliability diagram (calibration plot) shows how well-calibrated
    the predicted probabilities are by plotting predicted vs. actual
    frequencies.

    Args:
        y_true: True binary labels of shape (n_samples,).
        y_prob: Predicted probabilities of shape (n_samples,).
        n_bins: Number of bins.
            Default: 10
        strategy: Binning strategy ('uniform' or 'quantile').
            Default: 'uniform'
        title: Plot title.
            Default: 'Reliability Diagram'

    Returns:
        Dictionary with data for plotting:
            - prob_true: Actual frequencies per bin.
            - prob_pred: Predicted probabilities per bin.
            - bin_counts: Sample counts per bin.
            - ece: Expected Calibration Error.
            - mce: Maximum Calibration Error.
            - title: Plot title.
            - bin_edges: Bin boundaries.

    Example:
        >>> diagram_data = reliability_diagram(labels, predictions)
        >>> # Use with matplotlib:
        >>> plt.bar(diagram_data['prob_pred'], diagram_data['prob_true'])
    """
    prob_true, prob_pred, bin_counts = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy=strategy
    )

    ece = expected_calibration_error(
        y_true, y_prob, n_bins=n_bins, strategy=strategy
    )
    mce = maximum_calibration_error(
        y_true, y_prob, n_bins=n_bins, strategy=strategy
    )

    # Compute bin edges
    if strategy == "uniform":
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        quantiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(y_prob, quantiles)
        bin_edges[0] = 0.0
        bin_edges[-1] = 1.0

    return {
        "prob_true": prob_true,
        "prob_pred": prob_pred,
        "bin_counts": bin_counts,
        "ece": ece,
        "mce": mce,
        "title": title,
        "bin_edges": bin_edges,
        "n_bins": n_bins,
        "strategy": strategy,
    }


def adaptive_calibration_error(
    y_true: NDArray[np.integer[Any]],
    y_prob: NDArray[np.floating[Any]],
    n_bins: int = 10,
) -> float:
    """
    Compute Adaptive Calibration Error (ACE).

    ACE uses adaptive binning based on the distribution of predictions
    to provide more robust calibration estimates.

    Args:
        y_true: True binary labels of shape (n_samples,).
        y_prob: Predicted probabilities of shape (n_samples,).
        n_bins: Number of bins (will be adaptively sized).
            Default: 10

    Returns:
        Adaptive Calibration Error.

    Example:
        >>> ace = adaptive_calibration_error(labels, predictions)
        >>> print(f"ACE: {ace:.4f}")
    """
    # Sort by predicted probability
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    sorted_indices = np.argsort(y_prob)
    y_true_sorted = y_true[sorted_indices]
    y_prob_sorted = y_prob[sorted_indices]

    n_samples = len(y_prob)
    samples_per_bin = n_samples // n_bins

    if samples_per_bin == 0:
        return expected_calibration_error(y_true, y_prob, n_bins=n_bins)

    ace = 0.0
    for i in range(n_bins):
        start_idx = i * samples_per_bin
        if i == n_bins - 1:
            end_idx = n_samples
        else:
            end_idx = (i + 1) * samples_per_bin

        bin_true = y_true_sorted[start_idx:end_idx]
        bin_pred = y_prob_sorted[start_idx:end_idx]

        if len(bin_true) > 0:
            bin_acc = bin_true.mean()
            bin_conf = bin_pred.mean()
            ace += np.abs(bin_acc - bin_conf) * len(bin_true) / n_samples

    return float(ace)


def classwise_calibration_error(
    y_true: NDArray[np.integer[Any]],
    y_prob: NDArray[np.floating[Any]],
    n_bins: int = 10,
) -> dict[str, float]:
    """
    Compute class-wise calibration errors.

    For multi-label settings, computes calibration error separately
    for positive and negative class predictions.

    Args:
        y_true: True binary labels of shape (n_samples,).
        y_prob: Predicted probabilities of shape (n_samples,).
        n_bins: Number of bins.
            Default: 10

    Returns:
        Dictionary with positive and negative class ECE.

    Example:
        >>> cce = classwise_calibration_error(labels, predictions)
        >>> print(f"Positive ECE: {cce['positive_ece']:.4f}")
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    # Separate positive and negative samples
    pos_mask = y_true == 1
    neg_mask = y_true == 0

    # ECE for positive samples
    if pos_mask.sum() > 0:
        pos_ece = expected_calibration_error(
            y_true[pos_mask], y_prob[pos_mask], n_bins=n_bins
        )
    else:
        pos_ece = 0.0

    # ECE for negative samples
    if neg_mask.sum() > 0:
        neg_ece = expected_calibration_error(
            y_true[neg_mask], y_prob[neg_mask], n_bins=n_bins
        )
    else:
        neg_ece = 0.0

    return {
        "positive_ece": pos_ece,
        "negative_ece": neg_ece,
        "mean_classwise_ece": (pos_ece + neg_ece) / 2,
    }
