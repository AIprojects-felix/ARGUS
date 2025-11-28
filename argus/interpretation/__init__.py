"""
ARGUS Interpretation Module.

This module provides model interpretability and explainability tools:
- Attention Analysis: Attention weight visualization and interpretation
- Feature Attribution: SHAP, integrated gradients, and occlusion analysis
- Clinical Insights: Feature importance for clinical decision support

Copyright 2024-2025 VTP Consortium
Licensed under the Apache License, Version 2.0
"""

from argus.interpretation.attention import (
    AttentionAnalyzer,
    extract_attention_weights,
    attention_rollout,
    attention_flow,
)
from argus.interpretation.attribution import (
    FeatureAttributor,
    IntegratedGradients,
    OcclusionAnalysis,
    compute_shap_values,
)
from argus.interpretation.clinical import (
    ClinicalInsights,
    feature_importance_report,
    risk_factor_analysis,
    biomarker_contribution,
)

__all__ = [
    # Attention Analysis
    "AttentionAnalyzer",
    "extract_attention_weights",
    "attention_rollout",
    "attention_flow",
    # Feature Attribution
    "FeatureAttributor",
    "IntegratedGradients",
    "OcclusionAnalysis",
    "compute_shap_values",
    # Clinical Insights
    "ClinicalInsights",
    "feature_importance_report",
    "risk_factor_analysis",
    "biomarker_contribution",
]
