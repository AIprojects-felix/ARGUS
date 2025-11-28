"""
Bootstrap Confidence Interval Estimation.

Statistical inference utilities for ARGUS evaluation metrics.

Copyright 2024-2025 VTP Consortium
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from numpy.typing import NDArray


@dataclass
class ConfidenceInterval:
    """
    Confidence interval statistics for a metric.

    Attributes:
        point_estimate: Original metric value.
        lower: Lower bound of CI.
        upper: Upper bound of CI.
        confidence_level: Confidence level (e.g., 0.95).
        std_error: Standard error of the estimate.
        bootstrap_samples: Array of bootstrap estimates.
    """
    point_estimate: float
    lower: float
    upper: float
    confidence_level: float
    std_error: float
    bootstrap_samples: NDArray[np.floating[Any]] | None = None

    def __repr__(self) -> str:
        return (
            f"ConfidenceInterval("
            f"estimate={self.point_estimate:.4f}, "
            f"CI=[{self.lower:.4f}, {self.upper:.4f}], "
            f"level={self.confidence_level})"
        )


@dataclass
class BootstrapResult:
    """
    Complete bootstrap analysis results.

    Attributes:
        metrics: Dictionary mapping metric names to confidence intervals.
        n_bootstrap: Number of bootstrap iterations.
        confidence_level: Confidence level used.
        method: Bootstrap CI method used.
    """
    metrics: dict[str, ConfidenceInterval]
    n_bootstrap: int
    confidence_level: float
    method: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "n_bootstrap": self.n_bootstrap,
            "confidence_level": self.confidence_level,
            "method": self.method,
            "metrics": {
                name: {
                    "point_estimate": ci.point_estimate,
                    "lower": ci.lower,
                    "upper": ci.upper,
                    "std_error": ci.std_error,
                }
                for name, ci in self.metrics.items()
            },
        }


@dataclass
class BootstrapEvaluator:
    """
    Bootstrap-based confidence interval estimation for evaluation metrics.

    Implements non-parametric bootstrap resampling to estimate
    confidence intervals for arbitrary evaluation metrics.

    Args:
        n_bootstrap: Number of bootstrap iterations.
            Default: 1000
        confidence_level: Confidence level for intervals.
            Default: 0.95
        method: CI estimation method.
            - 'percentile': Simple percentile method.
            - 'bca': Bias-corrected and accelerated (BCa).
            - 'basic': Basic bootstrap (reverse percentile).
            Default: 'percentile'
        random_state: Random seed for reproducibility.
            Default: None
        n_jobs: Number of parallel jobs (-1 for all CPUs).
            Default: 1

    Example:
        >>> evaluator = BootstrapEvaluator(n_bootstrap=2000, confidence_level=0.95)
        >>> results = evaluator.evaluate(
        ...     y_true=labels,
        ...     y_pred=predictions,
        ...     metrics={'auroc': compute_auroc, 'auprc': compute_auprc}
        ... )
        >>> print(results.metrics['auroc'])
    """
    n_bootstrap: int = 1000
    confidence_level: float = 0.95
    method: str = "percentile"
    random_state: int | None = None
    n_jobs: int = 1

    # Internal state
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize random number generator."""
        self._rng = np.random.default_rng(self.random_state)

        if self.method not in ["percentile", "bca", "basic"]:
            raise ValueError(
                f"Unknown method: {self.method}. "
                f"Use 'percentile', 'bca', or 'basic'."
            )

    def evaluate(
        self,
        y_true: NDArray[np.integer[Any]],
        y_pred: NDArray[np.floating[Any]],
        metrics: dict[str, Callable[..., float]],
        stratify: NDArray[np.integer[Any]] | None = None,
    ) -> BootstrapResult:
        """
        Compute bootstrap confidence intervals for multiple metrics.

        Args:
            y_true: True labels of shape (n_samples,) or (n_samples, n_targets).
            y_pred: Predictions of shape (n_samples,) or (n_samples, n_targets).
            metrics: Dictionary mapping metric names to callable functions.
                Each function should accept (y_true, y_pred) and return float.
            stratify: Optional stratification variable for stratified bootstrap.

        Returns:
            BootstrapResult with confidence intervals for all metrics.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        n_samples = len(y_true)

        # Compute original metrics
        original_metrics = {
            name: func(y_true, y_pred) for name, func in metrics.items()
        }

        # Generate bootstrap samples
        bootstrap_estimates = {name: [] for name in metrics}

        if self.n_jobs == 1:
            # Sequential execution
            for _ in range(self.n_bootstrap):
                indices = self._generate_bootstrap_indices(n_samples, stratify)
                y_true_boot = y_true[indices]
                y_pred_boot = y_pred[indices]

                for name, func in metrics.items():
                    try:
                        value = func(y_true_boot, y_pred_boot)
                        bootstrap_estimates[name].append(value)
                    except Exception:
                        # Handle edge cases where metric computation fails
                        bootstrap_estimates[name].append(np.nan)
        else:
            # Parallel execution
            n_workers = self.n_jobs if self.n_jobs > 0 else None

            def compute_bootstrap_iteration(seed: int) -> dict[str, float]:
                rng = np.random.default_rng(seed)
                if stratify is not None:
                    indices = self._stratified_bootstrap_indices(
                        n_samples, stratify, rng
                    )
                else:
                    indices = rng.choice(n_samples, size=n_samples, replace=True)

                y_true_boot = y_true[indices]
                y_pred_boot = y_pred[indices]

                results = {}
                for name, func in metrics.items():
                    try:
                        results[name] = func(y_true_boot, y_pred_boot)
                    except Exception:
                        results[name] = np.nan
                return results

            # Generate seeds for reproducibility
            seeds = self._rng.integers(0, 2**31, size=self.n_bootstrap)

            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = [
                    executor.submit(compute_bootstrap_iteration, int(seed))
                    for seed in seeds
                ]

                for future in as_completed(futures):
                    result = future.result()
                    for name, value in result.items():
                        bootstrap_estimates[name].append(value)

        # Compute confidence intervals
        ci_results = {}
        for name in metrics:
            estimates = np.array(bootstrap_estimates[name])
            estimates = estimates[~np.isnan(estimates)]

            if len(estimates) < 10:
                # Not enough valid samples
                ci_results[name] = ConfidenceInterval(
                    point_estimate=original_metrics[name],
                    lower=np.nan,
                    upper=np.nan,
                    confidence_level=self.confidence_level,
                    std_error=np.nan,
                    bootstrap_samples=estimates,
                )
            else:
                ci_results[name] = self._compute_ci(
                    original_estimate=original_metrics[name],
                    bootstrap_estimates=estimates,
                    y_true=y_true,
                    y_pred=y_pred,
                    metric_func=metrics[name],
                )

        return BootstrapResult(
            metrics=ci_results,
            n_bootstrap=self.n_bootstrap,
            confidence_level=self.confidence_level,
            method=self.method,
        )

    def _generate_bootstrap_indices(
        self,
        n_samples: int,
        stratify: NDArray[np.integer[Any]] | None = None,
    ) -> NDArray[np.integer[Any]]:
        """Generate bootstrap sample indices."""
        if stratify is None:
            return self._rng.choice(n_samples, size=n_samples, replace=True)
        else:
            return self._stratified_bootstrap_indices(n_samples, stratify, self._rng)

    def _stratified_bootstrap_indices(
        self,
        n_samples: int,
        stratify: NDArray[np.integer[Any]],
        rng: np.random.Generator,
    ) -> NDArray[np.integer[Any]]:
        """Generate stratified bootstrap sample indices."""
        unique_strata = np.unique(stratify)
        indices = []

        for stratum in unique_strata:
            stratum_mask = stratify == stratum
            stratum_indices = np.where(stratum_mask)[0]
            n_stratum = len(stratum_indices)

            # Sample with replacement within each stratum
            boot_indices = rng.choice(stratum_indices, size=n_stratum, replace=True)
            indices.extend(boot_indices)

        return np.array(indices)

    def _compute_ci(
        self,
        original_estimate: float,
        bootstrap_estimates: NDArray[np.floating[Any]],
        y_true: NDArray[np.integer[Any]],
        y_pred: NDArray[np.floating[Any]],
        metric_func: Callable[..., float],
    ) -> ConfidenceInterval:
        """Compute confidence interval using specified method."""
        alpha = 1 - self.confidence_level
        std_error = float(np.std(bootstrap_estimates, ddof=1))

        if self.method == "percentile":
            lower = float(np.percentile(bootstrap_estimates, 100 * alpha / 2))
            upper = float(np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2)))

        elif self.method == "basic":
            # Basic bootstrap (reverse percentile)
            lower_percentile = np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2))
            upper_percentile = np.percentile(bootstrap_estimates, 100 * alpha / 2)
            lower = float(2 * original_estimate - lower_percentile)
            upper = float(2 * original_estimate - upper_percentile)

        elif self.method == "bca":
            # BCa (Bias-Corrected and Accelerated)
            lower, upper = self._bca_interval(
                original_estimate=original_estimate,
                bootstrap_estimates=bootstrap_estimates,
                y_true=y_true,
                y_pred=y_pred,
                metric_func=metric_func,
                alpha=alpha,
            )

        else:
            raise ValueError(f"Unknown method: {self.method}")

        return ConfidenceInterval(
            point_estimate=original_estimate,
            lower=lower,
            upper=upper,
            confidence_level=self.confidence_level,
            std_error=std_error,
            bootstrap_samples=bootstrap_estimates,
        )

    def _bca_interval(
        self,
        original_estimate: float,
        bootstrap_estimates: NDArray[np.floating[Any]],
        y_true: NDArray[np.integer[Any]],
        y_pred: NDArray[np.floating[Any]],
        metric_func: Callable[..., float],
        alpha: float,
    ) -> tuple[float, float]:
        """Compute BCa confidence interval."""
        from scipy import stats

        n_boot = len(bootstrap_estimates)

        # Bias correction factor (z0)
        prop_less = np.mean(bootstrap_estimates < original_estimate)
        z0 = stats.norm.ppf(max(min(prop_less, 0.9999), 0.0001))

        # Acceleration factor (a) using jackknife
        n_samples = len(y_true)
        jackknife_estimates = []

        for i in range(min(n_samples, 100)):  # Limit jackknife for large datasets
            mask = np.ones(n_samples, dtype=bool)
            mask[i] = False
            try:
                jack_est = metric_func(y_true[mask], y_pred[mask])
                jackknife_estimates.append(jack_est)
            except Exception:
                continue

        if len(jackknife_estimates) < 3:
            # Fall back to percentile method
            lower = float(np.percentile(bootstrap_estimates, 100 * alpha / 2))
            upper = float(np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2)))
            return lower, upper

        jackknife_estimates = np.array(jackknife_estimates)
        jack_mean = np.mean(jackknife_estimates)
        numerator = np.sum((jack_mean - jackknife_estimates) ** 3)
        denominator = 6 * (np.sum((jack_mean - jackknife_estimates) ** 2) ** 1.5)

        if denominator == 0:
            a = 0.0
        else:
            a = numerator / denominator

        # Compute adjusted percentiles
        z_alpha_lower = stats.norm.ppf(alpha / 2)
        z_alpha_upper = stats.norm.ppf(1 - alpha / 2)

        # BCa adjusted percentiles
        def bca_percentile(z_alpha: float) -> float:
            numerator = z0 + z_alpha
            denominator_term = 1 - a * (z0 + z_alpha)
            if denominator_term <= 0:
                return 0.5
            adjusted_z = z0 + numerator / denominator_term
            return stats.norm.cdf(adjusted_z)

        lower_percentile = bca_percentile(z_alpha_lower)
        upper_percentile = bca_percentile(z_alpha_upper)

        lower = float(np.percentile(bootstrap_estimates, 100 * lower_percentile))
        upper = float(np.percentile(bootstrap_estimates, 100 * upper_percentile))

        return lower, upper


def compute_confidence_intervals(
    y_true: NDArray[np.integer[Any]],
    y_pred: NDArray[np.floating[Any]],
    metric_func: Callable[..., float],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    method: str = "percentile",
    random_state: int | None = None,
) -> ConfidenceInterval:
    """
    Compute bootstrap confidence interval for a single metric.

    Convenience function for computing confidence intervals without
    creating a BootstrapEvaluator instance.

    Args:
        y_true: True labels.
        y_pred: Predictions.
        metric_func: Function that computes the metric.
        n_bootstrap: Number of bootstrap iterations.
            Default: 1000
        confidence_level: Confidence level.
            Default: 0.95
        method: CI estimation method.
            Default: 'percentile'
        random_state: Random seed.

    Returns:
        ConfidenceInterval for the metric.

    Example:
        >>> ci = compute_confidence_intervals(
        ...     y_true=labels,
        ...     y_pred=predictions,
        ...     metric_func=compute_auroc,
        ...     n_bootstrap=2000
        ... )
        >>> print(f"AUROC: {ci.point_estimate:.3f} [{ci.lower:.3f}, {ci.upper:.3f}]")
    """
    evaluator = BootstrapEvaluator(
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        method=method,
        random_state=random_state,
    )

    result = evaluator.evaluate(
        y_true=y_true,
        y_pred=y_pred,
        metrics={"metric": metric_func},
    )

    return result.metrics["metric"]


def paired_bootstrap_test(
    y_true: NDArray[np.integer[Any]],
    y_pred_a: NDArray[np.floating[Any]],
    y_pred_b: NDArray[np.floating[Any]],
    metric_func: Callable[..., float],
    n_bootstrap: int = 2000,
    random_state: int | None = None,
) -> dict[str, float]:
    """
    Paired bootstrap test for comparing two models.

    Tests whether model A significantly outperforms model B using
    paired bootstrap resampling.

    Args:
        y_true: True labels.
        y_pred_a: Predictions from model A.
        y_pred_b: Predictions from model B.
        metric_func: Metric function (higher is better assumed).
        n_bootstrap: Number of bootstrap iterations.
            Default: 2000
        random_state: Random seed.

    Returns:
        Dictionary with:
            - metric_a: Metric value for model A.
            - metric_b: Metric value for model B.
            - difference: A - B.
            - p_value: Two-sided p-value for H0: no difference.
            - ci_lower: Lower CI for difference.
            - ci_upper: Upper CI for difference.

    Example:
        >>> result = paired_bootstrap_test(
        ...     y_true=labels,
        ...     y_pred_a=model_a_preds,
        ...     y_pred_b=model_b_preds,
        ...     metric_func=compute_auroc
        ... )
        >>> if result['p_value'] < 0.05:
        ...     print("Significant difference!")
    """
    rng = np.random.default_rng(random_state)
    n_samples = len(y_true)

    # Original metrics
    metric_a = metric_func(y_true, y_pred_a)
    metric_b = metric_func(y_true, y_pred_b)
    original_diff = metric_a - metric_b

    # Bootstrap differences
    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        indices = rng.choice(n_samples, size=n_samples, replace=True)

        y_true_boot = y_true[indices]
        y_pred_a_boot = y_pred_a[indices]
        y_pred_b_boot = y_pred_b[indices]

        try:
            diff = metric_func(y_true_boot, y_pred_a_boot) - metric_func(
                y_true_boot, y_pred_b_boot
            )
            bootstrap_diffs.append(diff)
        except Exception:
            continue

    bootstrap_diffs = np.array(bootstrap_diffs)

    # Two-sided p-value
    # Under H0, the distribution is centered at 0
    centered_diffs = bootstrap_diffs - np.mean(bootstrap_diffs)
    p_value = float(np.mean(np.abs(centered_diffs) >= np.abs(original_diff)))

    # Confidence interval for the difference
    ci_lower = float(np.percentile(bootstrap_diffs, 2.5))
    ci_upper = float(np.percentile(bootstrap_diffs, 97.5))

    return {
        "metric_a": metric_a,
        "metric_b": metric_b,
        "difference": original_diff,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_bootstrap": n_bootstrap,
    }


def bootstrap_multi_target(
    y_true: NDArray[np.integer[Any]],
    y_pred: NDArray[np.floating[Any]],
    metric_func: Callable[..., float],
    target_names: list[str] | None = None,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int | None = None,
) -> dict[str, ConfidenceInterval]:
    """
    Compute bootstrap CIs for each target in multi-label setting.

    Args:
        y_true: True labels of shape (n_samples, n_targets).
        y_pred: Predictions of shape (n_samples, n_targets).
        metric_func: Metric function for single target.
        target_names: Names for each target.
        n_bootstrap: Number of bootstrap iterations.
        confidence_level: Confidence level.
        random_state: Random seed.

    Returns:
        Dictionary mapping target names to confidence intervals.

    Example:
        >>> cis = bootstrap_multi_target(
        ...     y_true=labels,
        ...     y_pred=predictions,
        ...     metric_func=compute_auroc,
        ...     target_names=gene_names
        ... )
        >>> for gene, ci in cis.items():
        ...     print(f"{gene}: {ci.point_estimate:.3f} [{ci.lower:.3f}, {ci.upper:.3f}]")
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_targets = y_true.shape[1]

    if target_names is None:
        target_names = [f"target_{i}" for i in range(n_targets)]

    results = {}
    for i, name in enumerate(target_names):
        ci = compute_confidence_intervals(
            y_true=y_true[:, i],
            y_pred=y_pred[:, i],
            metric_func=metric_func,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            random_state=random_state,
        )
        results[name] = ci

    return results
