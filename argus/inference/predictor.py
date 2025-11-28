"""
ARGUS Prediction Interface.

High-level prediction interface for clinical and research use.

Copyright 2024-2025 VTP Consortium
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """
    Single sample prediction result.

    Attributes:
        patient_id: Patient identifier.
        predictions: Predicted probabilities for all targets.
        predicted_classes: Binary predictions (after thresholding).
        confidence_scores: Confidence scores for predictions.
        target_names: Names of prediction targets.
        metadata: Additional prediction metadata.
    """
    patient_id: str
    predictions: NDArray[np.floating[Any]]
    predicted_classes: NDArray[np.integer[Any]]
    confidence_scores: NDArray[np.floating[Any]]
    target_names: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_positive_predictions(
        self,
        threshold: float = 0.5,
    ) -> list[tuple[str, float]]:
        """
        Get targets predicted as positive.

        Args:
            threshold: Classification threshold.

        Returns:
            List of (target_name, probability) tuples.
        """
        positive = []
        for i, (name, prob) in enumerate(zip(self.target_names, self.predictions)):
            if prob >= threshold:
                positive.append((name, float(prob)))
        return sorted(positive, key=lambda x: x[1], reverse=True)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "patient_id": self.patient_id,
            "predictions": {
                name: {
                    "probability": float(self.predictions[i]),
                    "predicted_class": int(self.predicted_classes[i]),
                    "confidence": float(self.confidence_scores[i]),
                }
                for i, name in enumerate(self.target_names)
            },
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        positive = self.get_positive_predictions()
        pos_str = ", ".join([f"{n}: {p:.2f}" for n, p in positive[:3]])
        return f"PredictionResult(patient={self.patient_id}, positive=[{pos_str}...])"


@dataclass
class BatchPredictionResult:
    """
    Batch prediction results container.

    Attributes:
        results: List of individual prediction results.
        summary: Batch-level summary statistics.
        processing_time: Total processing time in seconds.
    """
    results: list[PredictionResult]
    summary: dict[str, Any]
    processing_time: float

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self):
        return iter(self.results)

    def __getitem__(self, idx: int) -> PredictionResult:
        return self.results[idx]

    def get_predictions_matrix(self) -> NDArray[np.floating[Any]]:
        """Get all predictions as matrix [n_samples, n_targets]."""
        return np.stack([r.predictions for r in self.results])

    def filter_by_target(
        self,
        target_name: str,
        threshold: float = 0.5,
    ) -> list[PredictionResult]:
        """Filter results where target is predicted positive."""
        filtered = []
        for result in self.results:
            if target_name in result.target_names:
                idx = result.target_names.index(target_name)
                if result.predictions[idx] >= threshold:
                    filtered.append(result)
        return filtered


class ARGUSPredictor:
    """
    High-level prediction interface for ARGUS model.

    Provides convenient methods for single-sample and batch predictions
    with automatic preprocessing and post-processing.

    Args:
        model: Trained ARGUS model.
        device: Device for inference ('cuda', 'cpu', or 'auto').
            Default: 'auto'
        target_names: Names for prediction targets.
        threshold: Default classification threshold.
            Default: 0.5
        batch_size: Batch size for inference.
            Default: 32

    Example:
        >>> predictor = ARGUSPredictor.from_checkpoint(
        ...     checkpoint_path='model.pt',
        ...     device='cuda'
        ... )
        >>> result = predictor.predict_single(
        ...     static_features=patient_static,
        ...     temporal_features=patient_temporal,
        ...     patient_id='P001'
        ... )
        >>> print(result.get_positive_predictions())
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "auto",
        target_names: list[str] | None = None,
        threshold: float = 0.5,
        batch_size: int = 32,
    ) -> None:
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.model.eval()

        self.threshold = threshold
        self.batch_size = batch_size

        # Default target names
        if target_names is None:
            self.target_names = self._default_target_names()
        else:
            self.target_names = target_names

        logger.info(f"ARGUSPredictor initialized on {self.device}")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str = "auto",
        **kwargs: Any,
    ) -> "ARGUSPredictor":
        """
        Load predictor from saved checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint.
            device: Device for inference.
            **kwargs: Additional arguments for ARGUSPredictor.

        Returns:
            Initialized ARGUSPredictor.

        Example:
            >>> predictor = ARGUSPredictor.from_checkpoint('best_model.pt')
        """
        from argus.models import ARGUS

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Get model config from checkpoint
        config = checkpoint.get("config", {})

        # Initialize model
        model = ARGUS(
            n_static_features=config.get("n_static_features", 63),
            n_temporal_features=config.get("n_temporal_features", 117),
            d_model=config.get("d_model", 256),
            n_heads=config.get("n_heads", 8),
            n_layers=config.get("n_layers", 6),
            n_targets=config.get("n_targets", 43),
        )

        # Load weights
        model.load_state_dict(checkpoint["model_state_dict"])

        # Get target names if available
        target_names = checkpoint.get("target_names", kwargs.get("target_names"))

        return cls(model=model, device=device, target_names=target_names, **kwargs)

    def predict_single(
        self,
        static_features: NDArray[np.floating[Any]] | Tensor,
        temporal_features: NDArray[np.floating[Any]] | Tensor,
        temporal_mask: NDArray[np.integer[Any]] | Tensor | None = None,
        patient_id: str = "unknown",
    ) -> PredictionResult:
        """
        Make prediction for a single patient.

        Args:
            static_features: Static features [n_static_features].
            temporal_features: Temporal features [seq_len, n_temporal_features].
            temporal_mask: Attention mask [seq_len].
            patient_id: Patient identifier.

        Returns:
            PredictionResult for the patient.

        Example:
            >>> result = predictor.predict_single(
            ...     static_features=patient_data['static'],
            ...     temporal_features=patient_data['temporal'],
            ...     patient_id='P001'
            ... )
        """
        # Convert to tensors and add batch dimension
        static = self._to_tensor(static_features).unsqueeze(0)
        temporal = self._to_tensor(temporal_features).unsqueeze(0)

        if temporal_mask is not None:
            mask = self._to_tensor(temporal_mask).unsqueeze(0)
        else:
            mask = None

        # Inference
        with torch.no_grad():
            output = self.model(
                static_features=static,
                temporal_features=temporal,
                temporal_mask=mask,
            )

        # Extract predictions
        if isinstance(output, dict):
            logits = output.get("logits", output.get("predictions"))
        else:
            logits = output

        probabilities = torch.sigmoid(logits[0]).cpu().numpy()

        # Compute confidence scores
        confidence = np.abs(probabilities - 0.5) * 2  # 0-1 scale

        # Binary predictions
        predicted_classes = (probabilities >= self.threshold).astype(np.int32)

        return PredictionResult(
            patient_id=patient_id,
            predictions=probabilities,
            predicted_classes=predicted_classes,
            confidence_scores=confidence,
            target_names=self.target_names,
            metadata={"threshold": self.threshold},
        )

    def predict_batch(
        self,
        static_features: NDArray[np.floating[Any]] | Tensor,
        temporal_features: NDArray[np.floating[Any]] | Tensor,
        temporal_mask: NDArray[np.integer[Any]] | Tensor | None = None,
        patient_ids: list[str] | None = None,
    ) -> BatchPredictionResult:
        """
        Make predictions for a batch of patients.

        Args:
            static_features: Static features [n_samples, n_static_features].
            temporal_features: Temporal features [n_samples, seq_len, n_features].
            temporal_mask: Attention mask [n_samples, seq_len].
            patient_ids: List of patient identifiers.

        Returns:
            BatchPredictionResult containing all predictions.

        Example:
            >>> batch_result = predictor.predict_batch(
            ...     static_features=batch_static,
            ...     temporal_features=batch_temporal,
            ...     patient_ids=['P001', 'P002', 'P003']
            ... )
        """
        import time
        start_time = time.time()

        # Convert to tensors
        static = self._to_tensor(static_features)
        temporal = self._to_tensor(temporal_features)
        mask = self._to_tensor(temporal_mask) if temporal_mask is not None else None

        n_samples = static.shape[0]

        if patient_ids is None:
            patient_ids = [f"sample_{i}" for i in range(n_samples)]

        all_probabilities = []

        # Process in batches
        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_samples)

            batch_static = static[start_idx:end_idx]
            batch_temporal = temporal[start_idx:end_idx]
            batch_mask = mask[start_idx:end_idx] if mask is not None else None

            with torch.no_grad():
                output = self.model(
                    static_features=batch_static,
                    temporal_features=batch_temporal,
                    temporal_mask=batch_mask,
                )

            if isinstance(output, dict):
                logits = output.get("logits", output.get("predictions"))
            else:
                logits = output

            batch_probs = torch.sigmoid(logits).cpu().numpy()
            all_probabilities.append(batch_probs)

        # Combine batches
        probabilities = np.concatenate(all_probabilities, axis=0)

        # Create individual results
        results = []
        for i in range(n_samples):
            confidence = np.abs(probabilities[i] - 0.5) * 2
            predicted_classes = (probabilities[i] >= self.threshold).astype(np.int32)

            results.append(PredictionResult(
                patient_id=patient_ids[i],
                predictions=probabilities[i],
                predicted_classes=predicted_classes,
                confidence_scores=confidence,
                target_names=self.target_names,
                metadata={"threshold": self.threshold, "batch_index": i},
            ))

        processing_time = time.time() - start_time

        # Compute summary statistics
        summary = self._compute_batch_summary(probabilities)

        return BatchPredictionResult(
            results=results,
            summary=summary,
            processing_time=processing_time,
        )

    def predict_with_uncertainty(
        self,
        static_features: NDArray[np.floating[Any]] | Tensor,
        temporal_features: NDArray[np.floating[Any]] | Tensor,
        temporal_mask: NDArray[np.integer[Any]] | Tensor | None = None,
        n_samples: int = 10,
        dropout_rate: float = 0.1,
    ) -> dict[str, Any]:
        """
        Make predictions with uncertainty estimation using MC Dropout.

        Args:
            static_features: Static features.
            temporal_features: Temporal features.
            temporal_mask: Attention mask.
            n_samples: Number of forward passes for uncertainty.
            dropout_rate: Dropout rate for MC Dropout.

        Returns:
            Dictionary with mean predictions, std, and confidence intervals.

        Example:
            >>> result = predictor.predict_with_uncertainty(
            ...     static, temporal, n_samples=20
            ... )
            >>> print(f"Mean: {result['mean']}, Std: {result['std']}")
        """
        # Enable dropout for MC sampling
        def enable_dropout(module: nn.Module) -> None:
            if isinstance(module, nn.Dropout):
                module.train()

        self.model.apply(enable_dropout)

        static = self._to_tensor(static_features).unsqueeze(0)
        temporal = self._to_tensor(temporal_features).unsqueeze(0)
        mask = self._to_tensor(temporal_mask).unsqueeze(0) if temporal_mask is not None else None

        samples = []
        for _ in range(n_samples):
            with torch.no_grad():
                output = self.model(
                    static_features=static,
                    temporal_features=temporal,
                    temporal_mask=mask,
                )

            if isinstance(output, dict):
                logits = output.get("logits", output.get("predictions"))
            else:
                logits = output

            probs = torch.sigmoid(logits[0]).cpu().numpy()
            samples.append(probs)

        # Restore eval mode
        self.model.eval()

        samples = np.stack(samples)

        return {
            "mean": np.mean(samples, axis=0),
            "std": np.std(samples, axis=0),
            "ci_lower": np.percentile(samples, 2.5, axis=0),
            "ci_upper": np.percentile(samples, 97.5, axis=0),
            "n_samples": n_samples,
            "target_names": self.target_names,
        }

    def _to_tensor(self, data: NDArray | Tensor | None) -> Tensor | None:
        """Convert data to tensor on device."""
        if data is None:
            return None
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        return data.to(self.device)

    def _compute_batch_summary(
        self,
        probabilities: NDArray[np.floating[Any]],
    ) -> dict[str, Any]:
        """Compute summary statistics for batch predictions."""
        predicted_positive = (probabilities >= self.threshold).sum(axis=0)
        n_samples = probabilities.shape[0]

        return {
            "n_samples": n_samples,
            "mean_probabilities": {
                name: float(probabilities[:, i].mean())
                for i, name in enumerate(self.target_names)
            },
            "positive_counts": {
                name: int(predicted_positive[i])
                for i, name in enumerate(self.target_names)
            },
            "positive_rates": {
                name: float(predicted_positive[i] / n_samples)
                for i, name in enumerate(self.target_names)
            },
        }

    def _default_target_names(self) -> list[str]:
        """Generate default target names."""
        driver_genes = [
            "TP53", "KRAS", "PIK3CA", "EGFR", "BRAF", "PTEN", "APC",
            "RB1", "CDKN2A", "ARID1A", "ATM", "SMAD4", "BRCA1", "BRCA2",
            "NF1", "FBXW7", "MYC", "ERBB2", "STK11", "CTNNB1", "IDH1",
            "IDH2", "NRAS", "HRAS", "KIT", "PDGFRA", "FGFR1", "FGFR2",
            "FGFR3", "MET", "ALK", "ROS1", "RET", "NTRK1", "NTRK2",
            "NTRK3", "MAP2K1", "NFE2L2", "KEAP1", "NOTCH1"
        ]
        other_targets = ["TMB_High", "MSI_High", "PD_L1_High"]

        return driver_genes + other_targets

    def get_model_info(self) -> dict[str, Any]:
        """Get model information and configuration."""
        n_params = sum(p.numel() for p in self.model.parameters())
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            "device": str(self.device),
            "n_parameters": n_params,
            "n_trainable_parameters": n_trainable,
            "n_targets": len(self.target_names),
            "target_names": self.target_names,
            "threshold": self.threshold,
            "batch_size": self.batch_size,
        }
