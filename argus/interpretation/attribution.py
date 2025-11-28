"""
Feature Attribution Methods.

Model-agnostic and gradient-based attribution for ARGUS models.

Copyright 2024-2025 VTP Consortium
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class AttributionResult:
    """
    Feature attribution results container.

    Attributes:
        static_attributions: Attribution scores for static features [n_features].
        temporal_attributions: Attribution scores for temporal features [seq_len, n_features].
        target_idx: Index of the prediction target analyzed.
        method: Attribution method used.
        baseline_prediction: Model prediction on baseline.
        actual_prediction: Model prediction on actual input.
    """
    static_attributions: NDArray[np.floating[Any]]
    temporal_attributions: NDArray[np.floating[Any]]
    target_idx: int
    method: str
    baseline_prediction: float
    actual_prediction: float

    def get_top_static_features(
        self,
        k: int = 10,
        feature_names: list[str] | None = None,
    ) -> list[tuple[str | int, float]]:
        """Get top k most important static features."""
        importance = np.abs(self.static_attributions)
        top_indices = np.argsort(importance)[-k:][::-1]

        results = []
        for idx in top_indices:
            name = feature_names[idx] if feature_names else idx
            results.append((name, float(self.static_attributions[idx])))
        return results

    def get_top_temporal_features(
        self,
        k: int = 10,
        feature_names: list[str] | None = None,
    ) -> list[tuple[str | int, int, float]]:
        """Get top k most important temporal features with time indices."""
        importance = np.abs(self.temporal_attributions)
        flat_indices = np.argsort(importance.ravel())[-k:][::-1]

        seq_len, n_features = self.temporal_attributions.shape
        results = []
        for flat_idx in flat_indices:
            time_idx = flat_idx // n_features
            feat_idx = flat_idx % n_features
            name = feature_names[feat_idx] if feature_names else feat_idx
            results.append((
                name,
                int(time_idx),
                float(self.temporal_attributions[time_idx, feat_idx])
            ))
        return results


@dataclass
class FeatureAttributor:
    """
    Unified feature attribution interface for ARGUS models.

    Provides multiple attribution methods including integrated gradients,
    occlusion analysis, and gradient-based attribution.

    Args:
        model: ARGUS model instance.
        device: Device for computation.

    Example:
        >>> attributor = FeatureAttributor(model, device='cuda')
        >>> result = attributor.attribute(
        ...     static_features=static_data,
        ...     temporal_features=temporal_data,
        ...     target_idx=0,
        ...     method='integrated_gradients'
        ... )
        >>> print(result.get_top_static_features(k=5))
    """
    model: nn.Module
    device: str = "cpu"

    def attribute(
        self,
        static_features: Tensor,
        temporal_features: Tensor,
        target_idx: int = 0,
        method: str = "integrated_gradients",
        temporal_mask: Tensor | None = None,
        **kwargs: Any,
    ) -> AttributionResult:
        """
        Compute feature attributions using specified method.

        Args:
            static_features: Static features [batch_size, n_static_features].
            temporal_features: Temporal features [batch_size, seq_len, n_features].
            target_idx: Target prediction index to analyze.
            method: Attribution method.
                - 'integrated_gradients': Integrated gradients (default).
                - 'gradient': Simple gradient.
                - 'gradient_x_input': Gradient times input.
                - 'occlusion': Occlusion-based attribution.
            temporal_mask: Attention mask.
            **kwargs: Method-specific arguments.

        Returns:
            AttributionResult with computed attributions.
        """
        if method == "integrated_gradients":
            return self._integrated_gradients(
                static_features, temporal_features, temporal_mask,
                target_idx, **kwargs
            )
        elif method == "gradient":
            return self._gradient_attribution(
                static_features, temporal_features, temporal_mask,
                target_idx, multiply_input=False
            )
        elif method == "gradient_x_input":
            return self._gradient_attribution(
                static_features, temporal_features, temporal_mask,
                target_idx, multiply_input=True
            )
        elif method == "occlusion":
            return self._occlusion_attribution(
                static_features, temporal_features, temporal_mask,
                target_idx, **kwargs
            )
        else:
            raise ValueError(f"Unknown attribution method: {method}")

    def _integrated_gradients(
        self,
        static_features: Tensor,
        temporal_features: Tensor,
        temporal_mask: Tensor | None,
        target_idx: int,
        n_steps: int = 50,
        baseline_type: str = "zero",
    ) -> AttributionResult:
        """Compute integrated gradients attribution."""
        ig = IntegratedGradients(
            model=self.model,
            n_steps=n_steps,
            baseline_type=baseline_type,
        )

        return ig.attribute(
            static_features=static_features,
            temporal_features=temporal_features,
            temporal_mask=temporal_mask,
            target_idx=target_idx,
        )

    def _gradient_attribution(
        self,
        static_features: Tensor,
        temporal_features: Tensor,
        temporal_mask: Tensor | None,
        target_idx: int,
        multiply_input: bool = False,
    ) -> AttributionResult:
        """Compute gradient-based attribution."""
        static_features = static_features.clone().requires_grad_(True)
        temporal_features = temporal_features.clone().requires_grad_(True)

        self.model.eval()
        output = self.model(
            static_features=static_features,
            temporal_features=temporal_features,
            temporal_mask=temporal_mask,
        )

        if isinstance(output, dict):
            logits = output.get("logits", output.get("predictions"))
        else:
            logits = output

        # Get target output
        target_output = logits[0, target_idx]
        target_output.backward()

        static_grad = static_features.grad[0].detach().cpu().numpy()
        temporal_grad = temporal_features.grad[0].detach().cpu().numpy()

        if multiply_input:
            static_attr = static_grad * static_features[0].detach().cpu().numpy()
            temporal_attr = temporal_grad * temporal_features[0].detach().cpu().numpy()
        else:
            static_attr = static_grad
            temporal_attr = temporal_grad

        # Get predictions
        with torch.no_grad():
            actual_pred = torch.sigmoid(target_output).item()

        return AttributionResult(
            static_attributions=static_attr,
            temporal_attributions=temporal_attr,
            target_idx=target_idx,
            method="gradient_x_input" if multiply_input else "gradient",
            baseline_prediction=0.0,
            actual_prediction=actual_pred,
        )

    def _occlusion_attribution(
        self,
        static_features: Tensor,
        temporal_features: Tensor,
        temporal_mask: Tensor | None,
        target_idx: int,
        window_size: int = 1,
        stride: int = 1,
    ) -> AttributionResult:
        """Compute occlusion-based attribution."""
        occlusion = OcclusionAnalysis(
            model=self.model,
            window_size=window_size,
            stride=stride,
        )

        return occlusion.attribute(
            static_features=static_features,
            temporal_features=temporal_features,
            temporal_mask=temporal_mask,
            target_idx=target_idx,
        )


@dataclass
class IntegratedGradients:
    """
    Integrated Gradients attribution method.

    Computes feature importance by integrating gradients along a path
    from a baseline to the actual input.

    IG(x)_i = (x_i - x'_i) × ∫_{α=0}^{1} (∂F(x' + α(x - x'))/∂x_i) dα

    Reference: Sundararajan et al., 2017

    Args:
        model: Neural network model.
        n_steps: Number of integration steps.
            Default: 50
        baseline_type: Type of baseline.
            - 'zero': Zero baseline (default).
            - 'mean': Mean of training data.
            - 'random': Random baseline.
            Default: 'zero'

    Example:
        >>> ig = IntegratedGradients(model, n_steps=100)
        >>> attributions = ig.attribute(
        ...     static_features=static,
        ...     temporal_features=temporal,
        ...     target_idx=0
        ... )
    """
    model: nn.Module
    n_steps: int = 50
    baseline_type: str = "zero"

    def attribute(
        self,
        static_features: Tensor,
        temporal_features: Tensor,
        temporal_mask: Tensor | None = None,
        target_idx: int = 0,
        static_baseline: Tensor | None = None,
        temporal_baseline: Tensor | None = None,
    ) -> AttributionResult:
        """
        Compute integrated gradients for input features.

        Args:
            static_features: Static input [1, n_static_features].
            temporal_features: Temporal input [1, seq_len, n_temporal_features].
            temporal_mask: Attention mask.
            target_idx: Target output index.
            static_baseline: Custom static baseline (optional).
            temporal_baseline: Custom temporal baseline (optional).

        Returns:
            AttributionResult with IG attributions.
        """
        # Create baselines
        if static_baseline is None:
            static_baseline = self._create_baseline(
                static_features, self.baseline_type
            )
        if temporal_baseline is None:
            temporal_baseline = self._create_baseline(
                temporal_features, self.baseline_type
            )

        # Compute path integrals
        static_grads = []
        temporal_grads = []

        for step in range(self.n_steps + 1):
            alpha = step / self.n_steps

            # Interpolate between baseline and input
            static_interp = static_baseline + alpha * (static_features - static_baseline)
            temporal_interp = temporal_baseline + alpha * (temporal_features - temporal_baseline)

            static_interp = static_interp.clone().requires_grad_(True)
            temporal_interp = temporal_interp.clone().requires_grad_(True)

            # Forward pass
            self.model.eval()
            output = self.model(
                static_features=static_interp,
                temporal_features=temporal_interp,
                temporal_mask=temporal_mask,
            )

            if isinstance(output, dict):
                logits = output.get("logits", output.get("predictions"))
            else:
                logits = output

            # Backward pass
            target_output = logits[0, target_idx]
            target_output.backward()

            static_grads.append(static_interp.grad.detach().clone())
            temporal_grads.append(temporal_interp.grad.detach().clone())

        # Average gradients
        avg_static_grad = torch.stack(static_grads).mean(dim=0)
        avg_temporal_grad = torch.stack(temporal_grads).mean(dim=0)

        # Compute integrated gradients
        static_ig = (static_features - static_baseline) * avg_static_grad
        temporal_ig = (temporal_features - temporal_baseline) * avg_temporal_grad

        # Get predictions
        with torch.no_grad():
            baseline_output = self.model(
                static_features=static_baseline,
                temporal_features=temporal_baseline,
                temporal_mask=temporal_mask,
            )
            actual_output = self.model(
                static_features=static_features,
                temporal_features=temporal_features,
                temporal_mask=temporal_mask,
            )

            if isinstance(baseline_output, dict):
                baseline_logits = baseline_output.get("logits", baseline_output.get("predictions"))
                actual_logits = actual_output.get("logits", actual_output.get("predictions"))
            else:
                baseline_logits = baseline_output
                actual_logits = actual_output

            baseline_pred = torch.sigmoid(baseline_logits[0, target_idx]).item()
            actual_pred = torch.sigmoid(actual_logits[0, target_idx]).item()

        return AttributionResult(
            static_attributions=static_ig[0].detach().cpu().numpy(),
            temporal_attributions=temporal_ig[0].detach().cpu().numpy(),
            target_idx=target_idx,
            method="integrated_gradients",
            baseline_prediction=baseline_pred,
            actual_prediction=actual_pred,
        )

    def _create_baseline(self, tensor: Tensor, baseline_type: str) -> Tensor:
        """Create baseline tensor of same shape."""
        if baseline_type == "zero":
            return torch.zeros_like(tensor)
        elif baseline_type == "mean":
            return torch.full_like(tensor, tensor.mean().item())
        elif baseline_type == "random":
            return torch.randn_like(tensor) * 0.1
        else:
            raise ValueError(f"Unknown baseline type: {baseline_type}")


@dataclass
class OcclusionAnalysis:
    """
    Occlusion-based feature attribution.

    Measures feature importance by systematically occluding (zeroing out)
    features and measuring the change in model output.

    Args:
        model: Neural network model.
        window_size: Size of occlusion window.
            Default: 1
        stride: Stride for sliding window.
            Default: 1
        baseline_value: Value to use for occlusion.
            Default: 0.0

    Example:
        >>> occlusion = OcclusionAnalysis(model)
        >>> attributions = occlusion.attribute(
        ...     static_features=static,
        ...     temporal_features=temporal,
        ...     target_idx=0
        ... )
    """
    model: nn.Module
    window_size: int = 1
    stride: int = 1
    baseline_value: float = 0.0

    def attribute(
        self,
        static_features: Tensor,
        temporal_features: Tensor,
        temporal_mask: Tensor | None = None,
        target_idx: int = 0,
    ) -> AttributionResult:
        """
        Compute occlusion-based attributions.

        Args:
            static_features: Static input [1, n_static_features].
            temporal_features: Temporal input [1, seq_len, n_temporal_features].
            temporal_mask: Attention mask.
            target_idx: Target output index.

        Returns:
            AttributionResult with occlusion attributions.
        """
        self.model.eval()

        # Get baseline prediction
        with torch.no_grad():
            baseline_output = self.model(
                static_features=static_features,
                temporal_features=temporal_features,
                temporal_mask=temporal_mask,
            )

            if isinstance(baseline_output, dict):
                baseline_logits = baseline_output.get("logits", baseline_output.get("predictions"))
            else:
                baseline_logits = baseline_output

            baseline_pred = torch.sigmoid(baseline_logits[0, target_idx]).item()

        # Occlude static features
        n_static = static_features.shape[1]
        static_attributions = np.zeros(n_static)

        for i in range(0, n_static, self.stride):
            occluded_static = static_features.clone()
            end_idx = min(i + self.window_size, n_static)
            occluded_static[0, i:end_idx] = self.baseline_value

            with torch.no_grad():
                output = self.model(
                    static_features=occluded_static,
                    temporal_features=temporal_features,
                    temporal_mask=temporal_mask,
                )

                if isinstance(output, dict):
                    logits = output.get("logits", output.get("predictions"))
                else:
                    logits = output

                occluded_pred = torch.sigmoid(logits[0, target_idx]).item()

            # Attribution is change in prediction
            for j in range(i, end_idx):
                static_attributions[j] = baseline_pred - occluded_pred

        # Occlude temporal features
        seq_len, n_temporal = temporal_features.shape[1], temporal_features.shape[2]
        temporal_attributions = np.zeros((seq_len, n_temporal))

        for t in range(0, seq_len, self.stride):
            for f in range(0, n_temporal, self.stride):
                occluded_temporal = temporal_features.clone()
                t_end = min(t + self.window_size, seq_len)
                f_end = min(f + self.window_size, n_temporal)
                occluded_temporal[0, t:t_end, f:f_end] = self.baseline_value

                with torch.no_grad():
                    output = self.model(
                        static_features=static_features,
                        temporal_features=occluded_temporal,
                        temporal_mask=temporal_mask,
                    )

                    if isinstance(output, dict):
                        logits = output.get("logits", output.get("predictions"))
                    else:
                        logits = output

                    occluded_pred = torch.sigmoid(logits[0, target_idx]).item()

                # Attribution is change in prediction
                for ti in range(t, t_end):
                    for fi in range(f, f_end):
                        temporal_attributions[ti, fi] = baseline_pred - occluded_pred

        return AttributionResult(
            static_attributions=static_attributions,
            temporal_attributions=temporal_attributions,
            target_idx=target_idx,
            method="occlusion",
            baseline_prediction=baseline_pred,
            actual_prediction=baseline_pred,
        )


def compute_shap_values(
    model: nn.Module,
    static_features: Tensor,
    temporal_features: Tensor,
    temporal_mask: Tensor | None = None,
    target_idx: int = 0,
    n_samples: int = 100,
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    """
    Compute SHAP-like values using sampling approximation.

    Approximates Shapley values using random feature coalitions.

    Args:
        model: Neural network model.
        static_features: Static input [1, n_static_features].
        temporal_features: Temporal input [1, seq_len, n_temporal_features].
        temporal_mask: Attention mask.
        target_idx: Target output index.
        n_samples: Number of samples for approximation.
        feature_names: Names for static features.

    Returns:
        Dictionary with SHAP values and analysis.

    Example:
        >>> shap_result = compute_shap_values(
        ...     model, static, temporal, target_idx=0
        ... )
        >>> print(shap_result['static_shap_values'])
    """
    model.eval()

    n_static = static_features.shape[1]
    static_shap = np.zeros(n_static)

    # Sample random coalitions
    rng = np.random.default_rng()

    # Get baseline (all zeros)
    baseline_static = torch.zeros_like(static_features)
    baseline_temporal = torch.zeros_like(temporal_features)

    with torch.no_grad():
        # Baseline prediction
        baseline_out = model(
            static_features=baseline_static,
            temporal_features=baseline_temporal,
            temporal_mask=temporal_mask,
        )
        if isinstance(baseline_out, dict):
            baseline_logits = baseline_out.get("logits", baseline_out.get("predictions"))
        else:
            baseline_logits = baseline_out
        baseline_pred = torch.sigmoid(baseline_logits[0, target_idx]).item()

        # Full prediction
        full_out = model(
            static_features=static_features,
            temporal_features=temporal_features,
            temporal_mask=temporal_mask,
        )
        if isinstance(full_out, dict):
            full_logits = full_out.get("logits", full_out.get("predictions"))
        else:
            full_logits = full_out
        full_pred = torch.sigmoid(full_logits[0, target_idx]).item()

    # Estimate SHAP values via sampling
    for feature_idx in range(n_static):
        marginal_contributions = []

        for _ in range(n_samples):
            # Random coalition (subset of features)
            coalition_size = rng.integers(0, n_static)
            coalition = rng.choice(
                [i for i in range(n_static) if i != feature_idx],
                size=min(coalition_size, n_static - 1),
                replace=False
            )

            # Without feature
            coalition_static = baseline_static.clone()
            for idx in coalition:
                coalition_static[0, idx] = static_features[0, idx]

            with torch.no_grad():
                out_without = model(
                    static_features=coalition_static,
                    temporal_features=temporal_features,
                    temporal_mask=temporal_mask,
                )
                if isinstance(out_without, dict):
                    logits_without = out_without.get("logits", out_without.get("predictions"))
                else:
                    logits_without = out_without
                pred_without = torch.sigmoid(logits_without[0, target_idx]).item()

            # With feature
            coalition_static[0, feature_idx] = static_features[0, feature_idx]

            with torch.no_grad():
                out_with = model(
                    static_features=coalition_static,
                    temporal_features=temporal_features,
                    temporal_mask=temporal_mask,
                )
                if isinstance(out_with, dict):
                    logits_with = out_with.get("logits", out_with.get("predictions"))
                else:
                    logits_with = out_with
                pred_with = torch.sigmoid(logits_with[0, target_idx]).item()

            marginal_contributions.append(pred_with - pred_without)

        static_shap[feature_idx] = np.mean(marginal_contributions)

    return {
        "static_shap_values": static_shap,
        "baseline_prediction": baseline_pred,
        "full_prediction": full_pred,
        "target_idx": target_idx,
        "n_samples": n_samples,
        "feature_names": feature_names,
    }
