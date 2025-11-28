"""
Survival Analysis Utilities.

Kaplan-Meier analysis and survival statistics for ARGUS predictions.

Copyright 2024-2025 VTP Consortium
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class SurvivalCurve:
    """
    Kaplan-Meier survival curve data.

    Attributes:
        times: Unique event times.
        survival_prob: Survival probability at each time.
        confidence_lower: Lower confidence bound.
        confidence_upper: Upper confidence bound.
        at_risk: Number at risk at each time.
        events: Number of events at each time.
        censored: Number censored at each time.
    """
    times: NDArray[np.floating[Any]]
    survival_prob: NDArray[np.floating[Any]]
    confidence_lower: NDArray[np.floating[Any]]
    confidence_upper: NDArray[np.floating[Any]]
    at_risk: NDArray[np.integer[Any]]
    events: NDArray[np.integer[Any]]
    censored: NDArray[np.integer[Any]]

    def median_survival_time(self) -> float | None:
        """
        Compute median survival time.

        Returns:
            Median survival time or None if not reached.
        """
        below_50 = self.survival_prob <= 0.5
        if not np.any(below_50):
            return None
        return float(self.times[np.argmax(below_50)])

    def survival_at_time(self, t: float) -> float:
        """
        Get survival probability at a specific time.

        Args:
            t: Time point.

        Returns:
            Survival probability at time t.
        """
        if t < self.times[0]:
            return 1.0
        idx = np.searchsorted(self.times, t, side='right') - 1
        return float(self.survival_prob[idx])


@dataclass
class LogRankResult:
    """
    Log-rank test results.

    Attributes:
        statistic: Test statistic.
        p_value: P-value.
        observed: Observed events per group.
        expected: Expected events per group.
        n_groups: Number of groups compared.
    """
    statistic: float
    p_value: float
    observed: NDArray[np.floating[Any]]
    expected: NDArray[np.floating[Any]]
    n_groups: int

    def __repr__(self) -> str:
        return (
            f"LogRankResult(statistic={self.statistic:.4f}, "
            f"p_value={self.p_value:.4e})"
        )


@dataclass
class SurvivalAnalysis:
    """
    Comprehensive survival analysis for risk stratification.

    Evaluates model predictions by stratifying patients into risk groups
    and comparing survival outcomes using Kaplan-Meier analysis.

    Args:
        confidence_level: Confidence level for survival curves.
            Default: 0.95
        n_risk_groups: Number of risk groups for stratification.
            Default: 2

    Example:
        >>> analysis = SurvivalAnalysis(n_risk_groups=3)
        >>> results = analysis.analyze(
        ...     survival_time=times,
        ...     event_indicator=events,
        ...     risk_scores=model_predictions
        ... )
        >>> print(f"Log-rank p-value: {results['log_rank'].p_value:.4f}")
    """
    confidence_level: float = 0.95
    n_risk_groups: int = 2

    # Results (populated after analyze())
    survival_curves: dict[str, SurvivalCurve] = field(
        default_factory=dict, init=False
    )
    log_rank_result: LogRankResult | None = field(default=None, init=False)
    hazard_ratios: dict[str, float] = field(default_factory=dict, init=False)
    concordance_index: float = field(default=0.0, init=False)

    def analyze(
        self,
        survival_time: NDArray[np.floating[Any]],
        event_indicator: NDArray[np.integer[Any]],
        risk_scores: NDArray[np.floating[Any]],
        group_labels: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Perform comprehensive survival analysis.

        Args:
            survival_time: Time to event or censoring.
            event_indicator: Binary indicator (1=event, 0=censored).
            risk_scores: Model-predicted risk scores (higher = worse prognosis).
            group_labels: Labels for risk groups.

        Returns:
            Dictionary with analysis results.
        """
        survival_time = np.asarray(survival_time)
        event_indicator = np.asarray(event_indicator)
        risk_scores = np.asarray(risk_scores)

        # Stratify into risk groups
        percentiles = np.linspace(0, 100, self.n_risk_groups + 1)[1:-1]
        thresholds = np.percentile(risk_scores, percentiles)

        groups = np.digitize(risk_scores, thresholds)

        if group_labels is None:
            group_labels = [f"Group_{i+1}" for i in range(self.n_risk_groups)]

        # Compute survival curves for each group
        for i, label in enumerate(group_labels):
            mask = groups == i
            if mask.sum() < 2:
                continue

            self.survival_curves[label] = kaplan_meier_analysis(
                survival_time=survival_time[mask],
                event_indicator=event_indicator[mask],
                confidence_level=self.confidence_level,
            )

        # Log-rank test between groups
        if len(self.survival_curves) >= 2:
            self.log_rank_result = log_rank_test(
                survival_time=survival_time,
                event_indicator=event_indicator,
                groups=groups,
            )

        # Compute concordance index
        self.concordance_index = concordance_index(
            survival_time=survival_time,
            event_indicator=event_indicator,
            risk_scores=risk_scores,
        )

        # Compute hazard ratios (vs reference group)
        if len(self.survival_curves) >= 2:
            reference_group = 0
            for i in range(1, self.n_risk_groups):
                mask_ref = groups == reference_group
                mask_i = groups == i

                hr = hazard_ratio(
                    survival_time_a=survival_time[mask_ref],
                    event_indicator_a=event_indicator[mask_ref],
                    survival_time_b=survival_time[mask_i],
                    event_indicator_b=event_indicator[mask_i],
                )
                self.hazard_ratios[f"{group_labels[i]}_vs_{group_labels[0]}"] = hr

        return {
            "survival_curves": self.survival_curves,
            "log_rank": self.log_rank_result,
            "hazard_ratios": self.hazard_ratios,
            "concordance_index": self.concordance_index,
            "n_risk_groups": self.n_risk_groups,
            "group_labels": group_labels,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert analysis results to dictionary format."""
        return {
            "concordance_index": self.concordance_index,
            "log_rank": {
                "statistic": self.log_rank_result.statistic if self.log_rank_result else None,
                "p_value": self.log_rank_result.p_value if self.log_rank_result else None,
            },
            "hazard_ratios": self.hazard_ratios,
            "survival_curves": {
                name: {
                    "median_survival": curve.median_survival_time(),
                    "n_events": int(curve.events.sum()),
                    "n_censored": int(curve.censored.sum()),
                }
                for name, curve in self.survival_curves.items()
            },
        }


def kaplan_meier_analysis(
    survival_time: NDArray[np.floating[Any]],
    event_indicator: NDArray[np.integer[Any]],
    confidence_level: float = 0.95,
) -> SurvivalCurve:
    """
    Compute Kaplan-Meier survival curve.

    The Kaplan-Meier estimator is a non-parametric statistic used to
    estimate the survival function from lifetime data.

    S(t) = Π_{t_i ≤ t} (1 - d_i / n_i)

    where d_i is events at time t_i and n_i is at risk at t_i.

    Args:
        survival_time: Time to event or censoring.
        event_indicator: Binary indicator (1=event, 0=censored).
        confidence_level: Confidence level for confidence bands.
            Default: 0.95

    Returns:
        SurvivalCurve with survival estimates and confidence intervals.

    Example:
        >>> curve = kaplan_meier_analysis(
        ...     survival_time=times,
        ...     event_indicator=events
        ... )
        >>> print(f"Median survival: {curve.median_survival_time()}")
    """
    from scipy import stats

    survival_time = np.asarray(survival_time).ravel()
    event_indicator = np.asarray(event_indicator).ravel()

    # Sort by time
    order = np.argsort(survival_time)
    time_sorted = survival_time[order]
    event_sorted = event_indicator[order]

    # Get unique event times
    unique_times = np.unique(time_sorted[event_sorted == 1])

    if len(unique_times) == 0:
        # No events, return flat survival curve
        return SurvivalCurve(
            times=np.array([time_sorted.min(), time_sorted.max()]),
            survival_prob=np.array([1.0, 1.0]),
            confidence_lower=np.array([1.0, 1.0]),
            confidence_upper=np.array([1.0, 1.0]),
            at_risk=np.array([len(survival_time), 0]),
            events=np.array([0, 0]),
            censored=np.array([0, len(survival_time)]),
        )

    # Compute survival probability at each time
    n_samples = len(survival_time)
    survival_probs = []
    at_risk_list = []
    events_list = []
    censored_list = []

    current_survival = 1.0
    variance_sum = 0.0

    for t in unique_times:
        # Number at risk just before time t
        at_risk = np.sum(time_sorted >= t)

        # Number of events at time t
        events_at_t = np.sum((time_sorted == t) & (event_sorted == 1))

        # Number censored at time t
        censored_at_t = np.sum((time_sorted == t) & (event_sorted == 0))

        # Update survival probability
        if at_risk > 0:
            current_survival *= (at_risk - events_at_t) / at_risk

            # Greenwood's formula for variance
            if at_risk > events_at_t:
                variance_sum += events_at_t / (at_risk * (at_risk - events_at_t))

        survival_probs.append(current_survival)
        at_risk_list.append(at_risk)
        events_list.append(events_at_t)
        censored_list.append(censored_at_t)

    survival_probs = np.array(survival_probs)
    at_risk_array = np.array(at_risk_list)
    events_array = np.array(events_list)
    censored_array = np.array(censored_list)

    # Confidence intervals using log-log transformation
    z = stats.norm.ppf(1 - (1 - confidence_level) / 2)

    # Compute standard error using Greenwood's formula
    var_cumsum = np.zeros(len(unique_times))
    variance_sum = 0.0
    for i in range(len(unique_times)):
        d = events_array[i]
        n = at_risk_array[i]
        if n > d and n > 0:
            variance_sum += d / (n * (n - d))
        var_cumsum[i] = variance_sum

    # Log-log confidence interval
    with np.errstate(divide='ignore', invalid='ignore'):
        log_log_se = np.sqrt(var_cumsum) / np.abs(np.log(survival_probs))
        log_log_se = np.nan_to_num(log_log_se, nan=0.0, posinf=0.0, neginf=0.0)

        conf_lower = survival_probs ** np.exp(z * log_log_se)
        conf_upper = survival_probs ** np.exp(-z * log_log_se)

    conf_lower = np.clip(conf_lower, 0.0, 1.0)
    conf_upper = np.clip(conf_upper, 0.0, 1.0)

    return SurvivalCurve(
        times=unique_times,
        survival_prob=survival_probs,
        confidence_lower=conf_lower,
        confidence_upper=conf_upper,
        at_risk=at_risk_array,
        events=events_array,
        censored=censored_array,
    )


def log_rank_test(
    survival_time: NDArray[np.floating[Any]],
    event_indicator: NDArray[np.integer[Any]],
    groups: NDArray[np.integer[Any]],
) -> LogRankResult:
    """
    Perform log-rank test for comparing survival between groups.

    The log-rank test is used to test the null hypothesis that there
    is no difference in survival between two or more groups.

    χ² = Σ (O_j - E_j)² / V_j

    where O_j is observed events, E_j is expected events, V_j is variance.

    Args:
        survival_time: Time to event or censoring.
        event_indicator: Binary indicator (1=event, 0=censored).
        groups: Group assignment for each sample.

    Returns:
        LogRankResult with test statistic and p-value.

    Example:
        >>> result = log_rank_test(
        ...     survival_time=times,
        ...     event_indicator=events,
        ...     groups=risk_groups
        ... )
        >>> if result.p_value < 0.05:
        ...     print("Significant difference in survival!")
    """
    from scipy import stats

    survival_time = np.asarray(survival_time)
    event_indicator = np.asarray(event_indicator)
    groups = np.asarray(groups)

    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)

    # Get all unique event times
    event_times = np.unique(survival_time[event_indicator == 1])
    event_times = np.sort(event_times)

    # Compute observed and expected events for each group
    observed = np.zeros(n_groups)
    expected = np.zeros(n_groups)

    # Covariance matrix for variance
    variance_matrix = np.zeros((n_groups - 1, n_groups - 1))

    for t in event_times:
        # At risk in each group
        at_risk_per_group = np.array([
            np.sum((survival_time >= t) & (groups == g))
            for g in unique_groups
        ])
        total_at_risk = at_risk_per_group.sum()

        if total_at_risk == 0:
            continue

        # Events in each group at time t
        events_per_group = np.array([
            np.sum((survival_time == t) & (event_indicator == 1) & (groups == g))
            for g in unique_groups
        ])
        total_events = events_per_group.sum()

        # Update observed events
        observed += events_per_group

        # Expected events under null hypothesis
        if total_at_risk > 0:
            expected += at_risk_per_group * total_events / total_at_risk

        # Update variance (using hypergeometric variance)
        if total_at_risk > 1 and total_events > 0:
            var_factor = (
                total_events
                * (total_at_risk - total_events)
                / (total_at_risk ** 2 * (total_at_risk - 1))
            )

            for i in range(n_groups - 1):
                for j in range(n_groups - 1):
                    if i == j:
                        variance_matrix[i, j] += (
                            at_risk_per_group[i]
                            * (total_at_risk - at_risk_per_group[i])
                            * var_factor
                        )
                    else:
                        variance_matrix[i, j] -= (
                            at_risk_per_group[i]
                            * at_risk_per_group[j]
                            * var_factor
                        )

    # Compute test statistic
    # Use only first n_groups-1 groups (others are linearly dependent)
    diff = (observed - expected)[:-1]

    # Add small regularization for numerical stability
    variance_matrix += np.eye(n_groups - 1) * 1e-10

    try:
        variance_inv = np.linalg.inv(variance_matrix)
        test_statistic = float(diff @ variance_inv @ diff)
    except np.linalg.LinAlgError:
        # Fallback to simpler calculation if matrix is singular
        with np.errstate(divide='ignore', invalid='ignore'):
            test_statistic = float(np.sum(
                (observed - expected) ** 2 / np.maximum(expected, 1e-10)
            ))

    # P-value from chi-squared distribution
    df = n_groups - 1
    p_value = float(1 - stats.chi2.cdf(test_statistic, df))

    return LogRankResult(
        statistic=test_statistic,
        p_value=p_value,
        observed=observed,
        expected=expected,
        n_groups=n_groups,
    )


def concordance_index(
    survival_time: NDArray[np.floating[Any]],
    event_indicator: NDArray[np.integer[Any]],
    risk_scores: NDArray[np.floating[Any]],
) -> float:
    """
    Compute Harrell's concordance index (C-index).

    The concordance index measures the ability of the model to correctly
    order patients by their survival times. A value of 0.5 indicates
    random predictions, while 1.0 indicates perfect concordance.

    C = (concordant pairs) / (comparable pairs)

    Args:
        survival_time: Time to event or censoring.
        event_indicator: Binary indicator (1=event, 0=censored).
        risk_scores: Model-predicted risk scores (higher = worse prognosis).

    Returns:
        Concordance index (0.5 = random, 1.0 = perfect).

    Example:
        >>> c_index = concordance_index(
        ...     survival_time=times,
        ...     event_indicator=events,
        ...     risk_scores=predictions
        ... )
        >>> print(f"C-index: {c_index:.3f}")
    """
    survival_time = np.asarray(survival_time).ravel()
    event_indicator = np.asarray(event_indicator).ravel()
    risk_scores = np.asarray(risk_scores).ravel()

    n = len(survival_time)
    concordant = 0
    discordant = 0
    tied_risk = 0

    for i in range(n):
        for j in range(i + 1, n):
            # Check if pair is comparable
            # A pair is comparable if the shorter time has an event
            if survival_time[i] < survival_time[j]:
                if event_indicator[i] == 0:
                    continue  # Not comparable (i is censored before j)
                shorter_idx, longer_idx = i, j
            elif survival_time[j] < survival_time[i]:
                if event_indicator[j] == 0:
                    continue  # Not comparable (j is censored before i)
                shorter_idx, longer_idx = j, i
            else:
                # Same time
                if event_indicator[i] == event_indicator[j] == 0:
                    continue  # Both censored, not comparable
                if event_indicator[i] != event_indicator[j]:
                    continue  # One event, one censored at same time
                # Both events at same time - check risk ordering
                if risk_scores[i] > risk_scores[j]:
                    concordant += 0.5
                    discordant += 0.5
                elif risk_scores[i] < risk_scores[j]:
                    concordant += 0.5
                    discordant += 0.5
                else:
                    tied_risk += 1
                continue

            # Compare risk scores
            # Higher risk should correspond to shorter survival
            if risk_scores[shorter_idx] > risk_scores[longer_idx]:
                concordant += 1
            elif risk_scores[shorter_idx] < risk_scores[longer_idx]:
                discordant += 1
            else:
                tied_risk += 1

    total_comparable = concordant + discordant + tied_risk

    if total_comparable == 0:
        return 0.5

    # Concordance index with tied pairs counted as 0.5
    return (concordant + 0.5 * tied_risk) / total_comparable


def hazard_ratio(
    survival_time_a: NDArray[np.floating[Any]],
    event_indicator_a: NDArray[np.integer[Any]],
    survival_time_b: NDArray[np.floating[Any]],
    event_indicator_b: NDArray[np.integer[Any]],
) -> float:
    """
    Compute hazard ratio between two groups.

    The hazard ratio compares the hazard rates of two groups.
    HR > 1 indicates higher risk in group B compared to group A.

    This implementation uses the Mantel-Haenszel method for estimation.

    Args:
        survival_time_a: Survival times for group A (reference).
        event_indicator_a: Event indicators for group A.
        survival_time_b: Survival times for group B.
        event_indicator_b: Event indicators for group B.

    Returns:
        Hazard ratio (HR > 1 means group B has higher risk).

    Example:
        >>> hr = hazard_ratio(
        ...     survival_time_a=control_times,
        ...     event_indicator_a=control_events,
        ...     survival_time_b=treatment_times,
        ...     event_indicator_b=treatment_events
        ... )
        >>> print(f"Hazard Ratio: {hr:.2f}")
    """
    # Combine data
    survival_time = np.concatenate([survival_time_a, survival_time_b])
    event_indicator = np.concatenate([event_indicator_a, event_indicator_b])
    groups = np.concatenate([
        np.zeros(len(survival_time_a)),
        np.ones(len(survival_time_b))
    ])

    # Get unique event times
    event_times = np.unique(survival_time[event_indicator == 1])

    # Mantel-Haenszel estimator
    numerator = 0.0
    denominator = 0.0

    for t in event_times:
        # At risk in each group
        at_risk_a = np.sum((survival_time_a >= t))
        at_risk_b = np.sum((survival_time_b >= t))
        total_at_risk = at_risk_a + at_risk_b

        if total_at_risk == 0:
            continue

        # Events at time t
        events_a = np.sum((survival_time_a == t) & (event_indicator_a == 1))
        events_b = np.sum((survival_time_b == t) & (event_indicator_b == 1))
        total_events = events_a + events_b

        # Mantel-Haenszel weights
        numerator += events_b * at_risk_a / total_at_risk
        denominator += events_a * at_risk_b / total_at_risk

    if denominator == 0:
        return 1.0

    return numerator / denominator


def restricted_mean_survival_time(
    survival_time: NDArray[np.floating[Any]],
    event_indicator: NDArray[np.integer[Any]],
    tau: float | None = None,
) -> tuple[float, float]:
    """
    Compute Restricted Mean Survival Time (RMST).

    RMST is the area under the survival curve up to a specified time τ.
    It represents the mean survival time restricted to a time horizon.

    RMST(τ) = ∫₀^τ S(t) dt

    Args:
        survival_time: Time to event or censoring.
        event_indicator: Binary indicator (1=event, 0=censored).
        tau: Time horizon (default: max observed time).

    Returns:
        Tuple of (RMST, standard error).

    Example:
        >>> rmst, se = restricted_mean_survival_time(
        ...     survival_time=times,
        ...     event_indicator=events,
        ...     tau=60  # 60-month horizon
        ... )
        >>> print(f"RMST: {rmst:.2f} ± {1.96*se:.2f}")
    """
    # Get Kaplan-Meier curve
    curve = kaplan_meier_analysis(survival_time, event_indicator)

    if tau is None:
        tau = curve.times[-1]

    # Restrict to time horizon
    mask = curve.times <= tau
    times = curve.times[mask]
    survival = curve.survival_prob[mask]

    # Add initial point if needed
    if len(times) == 0 or times[0] > 0:
        times = np.concatenate([[0], times])
        survival = np.concatenate([[1.0], survival])

    # Add endpoint at tau
    if times[-1] < tau:
        times = np.concatenate([times, [tau]])
        survival = np.concatenate([survival, [survival[-1]]])

    # Compute area under curve (trapezoidal rule)
    rmst = np.trapz(survival, times)

    # Estimate standard error using Greenwood's formula
    # (simplified approximation)
    variance = 0.0
    at_risk = curve.at_risk[mask] if len(curve.at_risk[mask]) > 0 else np.array([len(survival_time)])
    events = curve.events[mask] if len(curve.events[mask]) > 0 else np.array([0])

    for i in range(len(at_risk)):
        if at_risk[i] > events[i] and at_risk[i] > 0:
            variance += events[i] / (at_risk[i] * (at_risk[i] - events[i]))

    se = rmst * np.sqrt(variance)

    return float(rmst), float(se)
