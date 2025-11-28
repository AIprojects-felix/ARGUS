"""
Inference Pipeline.

End-to-end inference pipeline with preprocessing and post-processing.

Copyright 2024-2025 VTP Consortium
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """
    Configuration for inference pipeline.

    Attributes:
        model_path: Path to model checkpoint.
        preprocessor_path: Path to preprocessor config.
        device: Device for inference.
        batch_size: Batch size for inference.
        threshold: Classification threshold.
        enable_calibration: Whether to apply calibration.
        enable_interpretation: Whether to compute interpretations.
    """
    model_path: str
    preprocessor_path: str | None = None
    device: str = "auto"
    batch_size: int = 32
    threshold: float = 0.5
    enable_calibration: bool = False
    enable_interpretation: bool = False
    calibration_method: str = "temperature_scaling"
    interpretation_method: str = "integrated_gradients"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        """Load config from YAML file."""
        import yaml
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, path: str | Path) -> None:
        """Save config to YAML file."""
        import yaml
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f)


@dataclass
class PipelineOutput:
    """
    Output from inference pipeline.

    Attributes:
        predictions: Predicted probabilities.
        predicted_classes: Binary predictions.
        calibrated_predictions: Calibrated probabilities (if enabled).
        interpretations: Feature attributions (if enabled).
        metadata: Additional output metadata.
    """
    predictions: NDArray[np.floating[Any]]
    predicted_classes: NDArray[np.integer[Any]]
    calibrated_predictions: NDArray[np.floating[Any]] | None = None
    interpretations: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "predictions": self.predictions.tolist(),
            "predicted_classes": self.predicted_classes.tolist(),
            "calibrated_predictions": (
                self.calibrated_predictions.tolist()
                if self.calibrated_predictions is not None
                else None
            ),
            "interpretations": self.interpretations,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class InferencePipeline:
    """
    End-to-end inference pipeline for ARGUS model.

    Combines preprocessing, model inference, calibration, and interpretation
    into a single unified pipeline.

    Args:
        config: Pipeline configuration.

    Example:
        >>> config = PipelineConfig(
        ...     model_path='model.pt',
        ...     preprocessor_path='preprocessor.json',
        ...     enable_interpretation=True
        ... )
        >>> pipeline = InferencePipeline(config)
        >>> output = pipeline.run(patient_data)
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._initialized = False

        # Components (lazy loaded)
        self._predictor = None
        self._preprocessor = None
        self._calibrator = None
        self._interpreter = None

    def initialize(self) -> None:
        """Initialize pipeline components."""
        if self._initialized:
            return

        from argus.inference.predictor import ARGUSPredictor

        # Load predictor
        self._predictor = ARGUSPredictor.from_checkpoint(
            checkpoint_path=self.config.model_path,
            device=self.config.device,
            threshold=self.config.threshold,
            batch_size=self.config.batch_size,
        )

        # Load preprocessor if specified
        if self.config.preprocessor_path:
            self._load_preprocessor()

        # Initialize calibrator if enabled
        if self.config.enable_calibration:
            self._initialize_calibrator()

        # Initialize interpreter if enabled
        if self.config.enable_interpretation:
            self._initialize_interpreter()

        self._initialized = True
        logger.info("Inference pipeline initialized")

    def run(
        self,
        data: dict[str, Any],
        return_raw: bool = False,
    ) -> PipelineOutput:
        """
        Run inference pipeline on input data.

        Args:
            data: Input data dictionary with required fields:
                - 'static_features': Static features array.
                - 'temporal_features': Temporal features array.
                - 'temporal_mask': Attention mask (optional).
                - 'patient_id': Patient identifier (optional).
            return_raw: Whether to return raw model outputs.

        Returns:
            PipelineOutput with predictions and optional interpretations.
        """
        self.initialize()

        # Preprocess data
        processed_data = self._preprocess(data)

        # Run inference
        result = self._predictor.predict_single(
            static_features=processed_data["static_features"],
            temporal_features=processed_data["temporal_features"],
            temporal_mask=processed_data.get("temporal_mask"),
            patient_id=data.get("patient_id", "unknown"),
        )

        # Apply calibration
        calibrated = None
        if self.config.enable_calibration and self._calibrator is not None:
            calibrated = self._calibrate(result.predictions)

        # Compute interpretations
        interpretations = None
        if self.config.enable_interpretation and self._interpreter is not None:
            interpretations = self._interpret(
                processed_data["static_features"],
                processed_data["temporal_features"],
                processed_data.get("temporal_mask"),
            )

        return PipelineOutput(
            predictions=result.predictions,
            predicted_classes=result.predicted_classes,
            calibrated_predictions=calibrated,
            interpretations=interpretations,
            metadata={
                "patient_id": data.get("patient_id"),
                "threshold": self.config.threshold,
                "calibration_applied": calibrated is not None,
                "interpretation_method": (
                    self.config.interpretation_method
                    if interpretations is not None
                    else None
                ),
            },
        )

    def run_batch(
        self,
        data_list: list[dict[str, Any]],
    ) -> list[PipelineOutput]:
        """
        Run inference pipeline on batch of samples.

        Args:
            data_list: List of input data dictionaries.

        Returns:
            List of PipelineOutput for each sample.
        """
        self.initialize()

        # Process all samples
        outputs = []
        for data in data_list:
            output = self.run(data)
            outputs.append(output)

        return outputs

    def _preprocess(self, data: dict[str, Any]) -> dict[str, Any]:
        """Preprocess input data."""
        processed = {}

        # Convert to numpy arrays if needed
        static = data["static_features"]
        temporal = data["temporal_features"]

        if not isinstance(static, np.ndarray):
            static = np.array(static, dtype=np.float32)
        if not isinstance(temporal, np.ndarray):
            temporal = np.array(temporal, dtype=np.float32)

        # Apply preprocessor if available
        if self._preprocessor is not None:
            static = self._preprocessor.transform_static(static)
            temporal = self._preprocessor.transform_temporal(temporal)

        processed["static_features"] = static
        processed["temporal_features"] = temporal

        # Handle mask
        if "temporal_mask" in data:
            mask = data["temporal_mask"]
            if not isinstance(mask, np.ndarray):
                mask = np.array(mask, dtype=np.int32)
            processed["temporal_mask"] = mask

        return processed

    def _calibrate(
        self,
        predictions: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        """Apply calibration to predictions."""
        if self._calibrator is None:
            return predictions
        return self._calibrator.calibrate(predictions)

    def _interpret(
        self,
        static_features: NDArray[np.floating[Any]],
        temporal_features: NDArray[np.floating[Any]],
        temporal_mask: NDArray[np.integer[Any]] | None,
    ) -> dict[str, Any]:
        """Compute feature attributions."""
        import torch

        static = torch.from_numpy(static_features).float().unsqueeze(0)
        temporal = torch.from_numpy(temporal_features).float().unsqueeze(0)
        mask = torch.from_numpy(temporal_mask).unsqueeze(0) if temporal_mask is not None else None

        static = static.to(self._predictor.device)
        temporal = temporal.to(self._predictor.device)
        if mask is not None:
            mask = mask.to(self._predictor.device)

        result = self._interpreter.attribute(
            static_features=static,
            temporal_features=temporal,
            temporal_mask=mask,
            method=self.config.interpretation_method,
        )

        return {
            "static_attributions": result.static_attributions.tolist(),
            "temporal_attributions": result.temporal_attributions.tolist(),
            "method": result.method,
        }

    def _load_preprocessor(self) -> None:
        """Load preprocessor from config."""
        # Placeholder for preprocessor loading
        logger.info(f"Loading preprocessor from {self.config.preprocessor_path}")
        self._preprocessor = None  # Will be implemented with actual preprocessor

    def _initialize_calibrator(self) -> None:
        """Initialize prediction calibrator."""
        logger.info(f"Initializing {self.config.calibration_method} calibrator")
        self._calibrator = TemperatureScaler()

    def _initialize_interpreter(self) -> None:
        """Initialize feature interpreter."""
        from argus.interpretation.attribution import FeatureAttributor

        logger.info(f"Initializing {self.config.interpretation_method} interpreter")
        self._interpreter = FeatureAttributor(
            model=self._predictor.model,
            device=str(self._predictor.device),
        )


class TemperatureScaler:
    """
    Temperature scaling calibration.

    Applies learned temperature parameter to logits for calibration.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = temperature

    def calibrate(
        self,
        predictions: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        """Apply temperature scaling to predictions."""
        # Convert to logits
        predictions = np.clip(predictions, 1e-10, 1 - 1e-10)
        logits = np.log(predictions / (1 - predictions))

        # Apply temperature
        scaled_logits = logits / self.temperature

        # Convert back to probabilities
        return 1 / (1 + np.exp(-scaled_logits))

    def fit(
        self,
        predictions: NDArray[np.floating[Any]],
        labels: NDArray[np.integer[Any]],
    ) -> None:
        """Fit temperature parameter to validation data."""
        from scipy.optimize import minimize_scalar

        predictions = np.clip(predictions, 1e-10, 1 - 1e-10)
        logits = np.log(predictions / (1 - predictions))

        def nll_loss(temperature: float) -> float:
            scaled_logits = logits / temperature
            probs = 1 / (1 + np.exp(-scaled_logits))
            probs = np.clip(probs, 1e-10, 1 - 1e-10)
            loss = -np.mean(
                labels * np.log(probs) + (1 - labels) * np.log(1 - probs)
            )
            return loss

        result = minimize_scalar(nll_loss, bounds=(0.1, 10.0), method="bounded")
        self.temperature = result.x
        logger.info(f"Calibration temperature: {self.temperature:.4f}")


@dataclass
class ClinicalPipeline:
    """
    Clinical-grade inference pipeline with additional safety checks.

    Extends InferencePipeline with clinical-specific features:
    - Input validation
    - Missing value handling
    - Confidence thresholds
    - Clinical alerts
    - Audit logging

    Args:
        config: Pipeline configuration.
        clinical_thresholds: Target-specific clinical thresholds.
        required_features: List of required feature names.

    Example:
        >>> pipeline = ClinicalPipeline(
        ...     config=config,
        ...     clinical_thresholds={'TP53': 0.7, 'KRAS': 0.6}
        ... )
        >>> output = pipeline.run_clinical(patient_data)
    """
    config: PipelineConfig
    clinical_thresholds: dict[str, float] = field(default_factory=dict)
    required_features: list[str] = field(default_factory=list)

    # Internal components
    _pipeline: InferencePipeline = field(init=False)

    def __post_init__(self) -> None:
        """Initialize clinical pipeline."""
        self._pipeline = InferencePipeline(self.config)

    def run_clinical(
        self,
        patient_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Run clinical inference with safety checks.

        Args:
            patient_data: Patient data dictionary.

        Returns:
            Clinical output with predictions, alerts, and recommendations.
        """
        # Validate input
        validation_result = self._validate_input(patient_data)
        if not validation_result["valid"]:
            return {
                "success": False,
                "error": validation_result["errors"],
                "predictions": None,
            }

        # Run inference
        try:
            output = self._pipeline.run(patient_data)
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return {
                "success": False,
                "error": str(e),
                "predictions": None,
            }

        # Generate clinical alerts
        alerts = self._generate_alerts(output)

        # Generate recommendations
        recommendations = self._generate_recommendations(output, alerts)

        # Log for audit
        self._audit_log(patient_data, output, alerts)

        return {
            "success": True,
            "predictions": output.to_dict(),
            "alerts": alerts,
            "recommendations": recommendations,
            "confidence_assessment": self._assess_confidence(output),
        }

    def _validate_input(self, data: dict[str, Any]) -> dict[str, Any]:
        """Validate input data for clinical use."""
        errors = []

        # Check required keys
        required_keys = ["static_features", "temporal_features"]
        for key in required_keys:
            if key not in data:
                errors.append(f"Missing required field: {key}")

        # Check data types and shapes
        if "static_features" in data:
            static = data["static_features"]
            if isinstance(static, (list, np.ndarray)):
                if len(static) == 0:
                    errors.append("Empty static features")
            else:
                errors.append(f"Invalid static features type: {type(static)}")

        if "temporal_features" in data:
            temporal = data["temporal_features"]
            if isinstance(temporal, (list, np.ndarray)):
                temporal = np.array(temporal)
                if len(temporal) == 0:
                    errors.append("Empty temporal features")
            else:
                errors.append(f"Invalid temporal features type: {type(temporal)}")

        # Check for NaN values
        if "static_features" in data and isinstance(data["static_features"], np.ndarray):
            if np.any(np.isnan(data["static_features"])):
                errors.append("NaN values detected in static features")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
        }

    def _generate_alerts(self, output: PipelineOutput) -> list[dict[str, Any]]:
        """Generate clinical alerts based on predictions."""
        alerts = []

        # Check each target against clinical thresholds
        target_names = self.config.target_names if hasattr(self.config, "target_names") else []

        for i, prob in enumerate(output.predictions):
            target_name = target_names[i] if i < len(target_names) else f"Target_{i}"
            threshold = self.clinical_thresholds.get(target_name, self.config.threshold)

            if prob >= threshold:
                severity = "high" if prob >= 0.8 else "moderate"
                alerts.append({
                    "target": target_name,
                    "probability": float(prob),
                    "threshold": threshold,
                    "severity": severity,
                    "message": f"Elevated probability for {target_name}: {prob:.1%}",
                })

        return alerts

    def _generate_recommendations(
        self,
        output: PipelineOutput,
        alerts: list[dict[str, Any]],
    ) -> list[str]:
        """Generate clinical recommendations."""
        recommendations = []

        # High severity alerts
        high_alerts = [a for a in alerts if a["severity"] == "high"]
        if high_alerts:
            targets = ", ".join([a["target"] for a in high_alerts])
            recommendations.append(
                f"Consider confirmatory molecular testing for: {targets}"
            )

        # Moderate severity alerts
        moderate_alerts = [a for a in alerts if a["severity"] == "moderate"]
        if moderate_alerts:
            recommendations.append(
                "Clinical correlation recommended for moderate-probability findings"
            )

        # General recommendations
        if not alerts:
            recommendations.append(
                "No high-probability genomic alterations detected. "
                "Standard clinical follow-up recommended."
            )

        return recommendations

    def _assess_confidence(self, output: PipelineOutput) -> dict[str, Any]:
        """Assess prediction confidence for clinical use."""
        probs = output.predictions

        # Entropy-based uncertainty
        probs_clipped = np.clip(probs, 1e-10, 1 - 1e-10)
        entropy = -np.mean(
            probs_clipped * np.log(probs_clipped) +
            (1 - probs_clipped) * np.log(1 - probs_clipped)
        )

        # Confidence classification
        if entropy < 0.3:
            confidence_level = "high"
        elif entropy < 0.6:
            confidence_level = "moderate"
        else:
            confidence_level = "low"

        return {
            "entropy": float(entropy),
            "confidence_level": confidence_level,
            "n_high_confidence_predictions": int(np.sum(np.abs(probs - 0.5) > 0.3)),
        }

    def _audit_log(
        self,
        patient_data: dict[str, Any],
        output: PipelineOutput,
        alerts: list[dict[str, Any]],
    ) -> None:
        """Log prediction for audit trail."""
        audit_entry = {
            "patient_id": patient_data.get("patient_id", "unknown"),
            "n_alerts": len(alerts),
            "high_severity_alerts": len([a for a in alerts if a["severity"] == "high"]),
            "model_version": self.config.model_path,
        }
        logger.info(f"Clinical prediction audit: {audit_entry}")
