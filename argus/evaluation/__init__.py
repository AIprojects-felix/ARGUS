"""
ARGUS Evaluation Module.

This module contains components for model evaluation:
- Metrics: Classification metrics (AUROC, AUPRC, etc.)
- Calibration: Calibration analysis and plots
- Survival: Survival analysis utilities
- Bootstrap: Confidence interval estimation
"""

from argus.evaluation.metrics import (
    ClassificationMetrics,
    MultiLabelMetrics,
    compute_auroc,
    compute_auprc,
    compute_f1_score,
    compute_precision_recall,
)
from argus.evaluation.calibration import (
    CalibrationAnalysis,
    calibration_curve,
    expected_calibration_error,
    reliability_diagram,
)
from argus.evaluation.bootstrap import (
    BootstrapEvaluator,
    compute_confidence_intervals,
)
from argus.evaluation.survival import (
    SurvivalAnalysis,
    kaplan_meier_analysis,
    log_rank_test,
)

__all__ = [
    # Metrics
    "ClassificationMetrics",
    "MultiLabelMetrics",
    "compute_auroc",
    "compute_auprc",
    "compute_f1_score",
    "compute_precision_recall",
    # Calibration
    "CalibrationAnalysis",
    "calibration_curve",
    "expected_calibration_error",
    "reliability_diagram",
    # Bootstrap
    "BootstrapEvaluator",
    "compute_confidence_intervals",
    # Survival
    "SurvivalAnalysis",
    "kaplan_meier_analysis",
    "log_rank_test",
]
