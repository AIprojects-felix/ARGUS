"""
Clinical Interpretation Utilities.

Tools for generating clinically meaningful insights from ARGUS predictions.

Copyright 2024-2025 VTP Consortium
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class FeatureImportance:
    """
    Feature importance with clinical context.

    Attributes:
        feature_name: Name of the clinical feature.
        importance_score: Numerical importance score.
        direction: Direction of effect ('positive', 'negative', 'complex').
        clinical_category: Category of feature (e.g., 'lab', 'vital', 'demographic').
        percentile_rank: Rank among all features (0-100).
    """
    feature_name: str
    importance_score: float
    direction: str
    clinical_category: str
    percentile_rank: float


@dataclass
class RiskFactorSummary:
    """
    Summary of risk factors for a prediction.

    Attributes:
        target_name: Name of prediction target.
        predicted_probability: Model prediction probability.
        confidence_level: Confidence classification.
        top_risk_factors: List of top contributing risk factors.
        protective_factors: List of factors decreasing risk.
        key_biomarkers: Relevant biomarker values and their impact.
    """
    target_name: str
    predicted_probability: float
    confidence_level: str
    top_risk_factors: list[FeatureImportance]
    protective_factors: list[FeatureImportance]
    key_biomarkers: dict[str, Any]


@dataclass
class ClinicalInsights:
    """
    Generate clinical insights from ARGUS predictions.

    Transforms model attributions and predictions into clinically
    meaningful summaries for clinical decision support.

    Args:
        feature_mapping: Mapping from feature indices to clinical names.
        category_mapping: Mapping from features to clinical categories.
        reference_ranges: Reference ranges for lab values.

    Example:
        >>> insights = ClinicalInsights(
        ...     feature_mapping=feature_names,
        ...     category_mapping=categories
        ... )
        >>> report = insights.generate_report(
        ...     predictions=model_output,
        ...     attributions=attribution_result,
        ...     patient_data=patient_features
        ... )
    """
    feature_mapping: dict[int, str] = field(default_factory=dict)
    category_mapping: dict[str, str] = field(default_factory=dict)
    reference_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)

    # Target name mapping
    target_names: dict[int, str] = field(default_factory=dict)

    def generate_report(
        self,
        predictions: NDArray[np.floating[Any]],
        attributions: NDArray[np.floating[Any]],
        patient_data: NDArray[np.floating[Any]] | None = None,
        target_indices: list[int] | None = None,
    ) -> dict[str, Any]:
        """
        Generate comprehensive clinical report from predictions.

        Args:
            predictions: Model predictions [n_targets].
            attributions: Feature attributions [n_features].
            patient_data: Patient feature values (optional).
            target_indices: Specific targets to report on.

        Returns:
            Dictionary containing clinical report sections.
        """
        report = {
            "summary": {},
            "risk_factors": {},
            "biomarker_analysis": {},
            "recommendations": [],
            "confidence_assessment": {},
        }

        # Determine targets to report
        if target_indices is None:
            target_indices = list(range(len(predictions)))

        for target_idx in target_indices:
            target_name = self.target_names.get(target_idx, f"Target_{target_idx}")
            pred_prob = float(predictions[target_idx])

            # Classify confidence level
            confidence = self._classify_confidence(pred_prob)

            # Get top features
            top_positive, top_negative = self._get_top_features(
                attributions, k=5
            )

            # Create risk factor summary
            risk_summary = RiskFactorSummary(
                target_name=target_name,
                predicted_probability=pred_prob,
                confidence_level=confidence,
                top_risk_factors=[
                    self._create_feature_importance(idx, score, attributions)
                    for idx, score in top_positive
                ],
                protective_factors=[
                    self._create_feature_importance(idx, score, attributions)
                    for idx, score in top_negative
                ],
                key_biomarkers=self._extract_biomarkers(patient_data),
            )

            report["risk_factors"][target_name] = risk_summary

            # Add to summary
            report["summary"][target_name] = {
                "probability": pred_prob,
                "confidence": confidence,
                "top_factor": top_positive[0][0] if top_positive else None,
            }

        # Generate recommendations
        report["recommendations"] = self._generate_recommendations(report)

        # Add confidence assessment
        report["confidence_assessment"] = self._assess_confidence(
            predictions, attributions
        )

        return report

    def _classify_confidence(self, probability: float) -> str:
        """Classify prediction confidence level."""
        if probability < 0.2:
            return "low_risk"
        elif probability < 0.4:
            return "low_moderate_risk"
        elif probability < 0.6:
            return "moderate_risk"
        elif probability < 0.8:
            return "moderate_high_risk"
        else:
            return "high_risk"

    def _get_top_features(
        self,
        attributions: NDArray[np.floating[Any]],
        k: int = 5,
    ) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
        """Get top positive and negative feature attributions."""
        # Top positive (increasing risk)
        positive_indices = np.argsort(attributions)[-k:][::-1]
        top_positive = [
            (int(idx), float(attributions[idx]))
            for idx in positive_indices
            if attributions[idx] > 0
        ]

        # Top negative (protective)
        negative_indices = np.argsort(attributions)[:k]
        top_negative = [
            (int(idx), float(attributions[idx]))
            for idx in negative_indices
            if attributions[idx] < 0
        ]

        return top_positive, top_negative

    def _create_feature_importance(
        self,
        feature_idx: int,
        score: float,
        all_attributions: NDArray[np.floating[Any]],
    ) -> FeatureImportance:
        """Create FeatureImportance object."""
        feature_name = self.feature_mapping.get(feature_idx, f"Feature_{feature_idx}")
        category = self.category_mapping.get(feature_name, "unknown")

        # Calculate percentile rank
        abs_scores = np.abs(all_attributions)
        percentile = 100 * np.mean(abs_scores <= np.abs(score))

        direction = "positive" if score > 0 else "negative"

        return FeatureImportance(
            feature_name=feature_name,
            importance_score=score,
            direction=direction,
            clinical_category=category,
            percentile_rank=percentile,
        )

    def _extract_biomarkers(
        self,
        patient_data: NDArray[np.floating[Any]] | None,
    ) -> dict[str, Any]:
        """Extract key biomarker values and assessments."""
        if patient_data is None:
            return {}

        biomarkers = {}

        # Define key biomarkers to extract
        key_biomarkers = [
            "hemoglobin", "wbc", "platelets", "albumin", "ldh",
            "cea", "ca125", "afp", "psa", "ca199"
        ]

        for biomarker in key_biomarkers:
            # Find biomarker index in feature mapping
            for idx, name in self.feature_mapping.items():
                if biomarker.lower() in name.lower():
                    if idx < len(patient_data):
                        value = float(patient_data[idx])
                        ref_range = self.reference_ranges.get(name)

                        assessment = "normal"
                        if ref_range:
                            if value < ref_range[0]:
                                assessment = "low"
                            elif value > ref_range[1]:
                                assessment = "high"

                        biomarkers[name] = {
                            "value": value,
                            "reference_range": ref_range,
                            "assessment": assessment,
                        }
                    break

        return biomarkers

    def _generate_recommendations(
        self,
        report: dict[str, Any],
    ) -> list[str]:
        """Generate clinical recommendations based on findings."""
        recommendations = []

        for target_name, risk_summary in report.get("risk_factors", {}).items():
            if isinstance(risk_summary, RiskFactorSummary):
                prob = risk_summary.predicted_probability

                if prob >= 0.8:
                    recommendations.append(
                        f"High probability of {target_name} detected. "
                        f"Consider confirmatory molecular testing."
                    )
                elif prob >= 0.5:
                    recommendations.append(
                        f"Moderate probability of {target_name}. "
                        f"Clinical correlation recommended."
                    )

                # Add factor-specific recommendations
                for factor in risk_summary.top_risk_factors[:2]:
                    if factor.clinical_category == "lab":
                        recommendations.append(
                            f"Monitor {factor.feature_name} - "
                            f"significant contributor to {target_name} prediction."
                        )

        return recommendations

    def _assess_confidence(
        self,
        predictions: NDArray[np.floating[Any]],
        attributions: NDArray[np.floating[Any]],
    ) -> dict[str, Any]:
        """Assess overall prediction confidence."""
        # Entropy-based uncertainty
        probs = np.clip(predictions, 1e-10, 1 - 1e-10)
        entropy = -np.mean(probs * np.log(probs) + (1 - probs) * np.log(1 - probs))

        # Attribution spread
        attr_concentration = np.sum(np.abs(attributions) ** 2) / (np.sum(np.abs(attributions)) ** 2 + 1e-10)

        return {
            "prediction_entropy": float(entropy),
            "attribution_concentration": float(attr_concentration),
            "overall_confidence": "high" if entropy < 0.5 and attr_concentration > 0.1 else "moderate",
        }


def feature_importance_report(
    attributions: NDArray[np.floating[Any]],
    feature_names: list[str] | None = None,
    k: int = 20,
) -> dict[str, Any]:
    """
    Generate feature importance report from attributions.

    Args:
        attributions: Feature attribution scores.
        feature_names: Names for features.
        k: Number of top features to include.

    Returns:
        Dictionary with feature importance analysis.

    Example:
        >>> report = feature_importance_report(
        ...     attributions=attr_scores,
        ...     feature_names=names,
        ...     k=10
        ... )
    """
    n_features = len(attributions)

    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(n_features)]

    # Sort by absolute importance
    abs_importance = np.abs(attributions)
    sorted_indices = np.argsort(abs_importance)[::-1]

    # Top k features
    top_features = []
    for rank, idx in enumerate(sorted_indices[:k]):
        top_features.append({
            "rank": rank + 1,
            "name": feature_names[idx],
            "importance": float(attributions[idx]),
            "abs_importance": float(abs_importance[idx]),
            "direction": "positive" if attributions[idx] > 0 else "negative",
            "percentile": 100 * (1 - rank / n_features),
        })

    # Summary statistics
    summary = {
        "total_features": n_features,
        "mean_importance": float(np.mean(abs_importance)),
        "std_importance": float(np.std(abs_importance)),
        "max_importance": float(np.max(abs_importance)),
        "top_positive_features": len([a for a in attributions[sorted_indices[:k]] if a > 0]),
        "top_negative_features": len([a for a in attributions[sorted_indices[:k]] if a < 0]),
    }

    return {
        "top_features": top_features,
        "summary": summary,
        "all_importances": {
            feature_names[i]: float(attributions[i])
            for i in range(n_features)
        },
    }


def risk_factor_analysis(
    predictions: NDArray[np.floating[Any]],
    attributions: NDArray[np.floating[Any]],
    target_names: list[str] | None = None,
    feature_names: list[str] | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Analyze risk factors across multiple prediction targets.

    Args:
        predictions: Model predictions [n_targets].
        attributions: Feature attributions [n_features] or [n_targets, n_features].
        target_names: Names for prediction targets.
        feature_names: Names for features.
        threshold: Risk threshold for classification.

    Returns:
        Dictionary with risk factor analysis.

    Example:
        >>> analysis = risk_factor_analysis(
        ...     predictions=preds,
        ...     attributions=attrs,
        ...     target_names=genes,
        ...     threshold=0.5
        ... )
    """
    n_targets = len(predictions)

    if target_names is None:
        target_names = [f"Target_{i}" for i in range(n_targets)]

    # Handle 1D vs 2D attributions
    if attributions.ndim == 1:
        attributions = np.tile(attributions, (n_targets, 1))

    results = {
        "high_risk_targets": [],
        "moderate_risk_targets": [],
        "low_risk_targets": [],
        "per_target_analysis": {},
    }

    for i, (target, prob) in enumerate(zip(target_names, predictions)):
        target_attrs = attributions[i] if attributions.ndim > 1 else attributions

        # Classify risk level
        if prob >= threshold:
            results["high_risk_targets"].append(target)
        elif prob >= threshold * 0.5:
            results["moderate_risk_targets"].append(target)
        else:
            results["low_risk_targets"].append(target)

        # Per-target analysis
        abs_attrs = np.abs(target_attrs)
        top_indices = np.argsort(abs_attrs)[-5:][::-1]

        results["per_target_analysis"][target] = {
            "probability": float(prob),
            "risk_level": "high" if prob >= threshold else ("moderate" if prob >= threshold * 0.5 else "low"),
            "top_factors": [
                {
                    "feature": feature_names[idx] if feature_names else f"Feature_{idx}",
                    "importance": float(target_attrs[idx]),
                    "direction": "risk" if target_attrs[idx] > 0 else "protective",
                }
                for idx in top_indices
            ],
        }

    # Cross-target analysis
    if attributions.ndim > 1:
        mean_importance = np.mean(np.abs(attributions), axis=0)
        consistent_factors = np.argsort(mean_importance)[-10:][::-1]

        results["consistent_factors"] = [
            {
                "feature": feature_names[idx] if feature_names else f"Feature_{idx}",
                "mean_importance": float(mean_importance[idx]),
            }
            for idx in consistent_factors
        ]

    return results


def biomarker_contribution(
    attributions: NDArray[np.floating[Any]],
    patient_values: NDArray[np.floating[Any]],
    feature_names: list[str],
    biomarker_categories: dict[str, list[str]],
) -> dict[str, Any]:
    """
    Analyze biomarker contributions to predictions.

    Groups features into biomarker categories and analyzes
    their collective contribution.

    Args:
        attributions: Feature attribution scores.
        patient_values: Patient feature values.
        feature_names: Names for features.
        biomarker_categories: Mapping from category names to feature lists.
            Example: {'tumor_markers': ['CEA', 'CA125'], 'inflammatory': ['CRP', 'ESR']}

    Returns:
        Dictionary with biomarker contribution analysis.

    Example:
        >>> categories = {
        ...     'tumor_markers': ['CEA', 'CA125', 'AFP'],
        ...     'blood_counts': ['WBC', 'Hemoglobin', 'Platelets']
        ... }
        >>> analysis = biomarker_contribution(
        ...     attributions=attrs,
        ...     patient_values=values,
        ...     feature_names=names,
        ...     biomarker_categories=categories
        ... )
    """
    # Build feature name to index mapping
    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}

    category_analysis = {}

    for category_name, biomarkers in biomarker_categories.items():
        category_attrs = []
        category_values = []
        category_features = []

        for biomarker in biomarkers:
            # Find matching feature (case-insensitive)
            for feat_name, idx in name_to_idx.items():
                if biomarker.lower() in feat_name.lower():
                    category_attrs.append(attributions[idx])
                    category_values.append(patient_values[idx])
                    category_features.append({
                        "name": feat_name,
                        "attribution": float(attributions[idx]),
                        "value": float(patient_values[idx]),
                    })
                    break

        if category_attrs:
            category_analysis[category_name] = {
                "total_contribution": float(np.sum(category_attrs)),
                "mean_contribution": float(np.mean(category_attrs)),
                "direction": "risk_increasing" if np.sum(category_attrs) > 0 else "risk_decreasing",
                "n_features": len(category_attrs),
                "features": category_features,
            }

    # Overall biomarker summary
    total_biomarker_contribution = sum(
        cat["total_contribution"]
        for cat in category_analysis.values()
    )

    summary = {
        "category_analysis": category_analysis,
        "total_biomarker_contribution": total_biomarker_contribution,
        "dominant_category": max(
            category_analysis.items(),
            key=lambda x: abs(x[1]["total_contribution"])
        )[0] if category_analysis else None,
        "n_categories_analyzed": len(category_analysis),
    }

    return summary
