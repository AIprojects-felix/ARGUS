"""
Attention Analysis Utilities.

Attention weight extraction and visualization for ARGUS models.

Copyright 2024-2025 VTP Consortium
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
from torch import Tensor

if TYPE_CHECKING:
    from argus.models.argus import ARGUS


@dataclass
class AttentionMaps:
    """
    Container for attention weight analysis results.

    Attributes:
        raw_weights: Raw attention weights per layer [n_layers, n_heads, seq_len, seq_len].
        averaged_weights: Head-averaged attention weights [n_layers, seq_len, seq_len].
        cls_attention: CLS token attention to all positions [n_layers, n_heads, seq_len].
        rollout_attention: Attention rollout aggregation [seq_len, seq_len].
        flow_attention: Attention flow aggregation [seq_len].
    """
    raw_weights: NDArray[np.floating[Any]]
    averaged_weights: NDArray[np.floating[Any]]
    cls_attention: NDArray[np.floating[Any]]
    rollout_attention: NDArray[np.floating[Any]] | None = None
    flow_attention: NDArray[np.floating[Any]] | None = None


@dataclass
class AttentionAnalyzer:
    """
    Comprehensive attention analysis for ARGUS transformer encoder.

    Extracts and analyzes attention patterns from the temporal transformer
    to understand which time points and features are important for predictions.

    Args:
        model: ARGUS model instance.
        include_rollout: Whether to compute attention rollout.
            Default: True
        include_flow: Whether to compute attention flow.
            Default: True

    Example:
        >>> analyzer = AttentionAnalyzer(model)
        >>> attention_maps = analyzer.analyze(
        ...     static_features=static_data,
        ...     temporal_features=temporal_data,
        ...     temporal_mask=mask
        ... )
        >>> print(f"Attention shape: {attention_maps.raw_weights.shape}")
    """
    model: Any  # ARGUS model
    include_rollout: bool = True
    include_flow: bool = True

    # Internal storage for attention weights
    _attention_weights: list[Tensor] = field(default_factory=list, init=False)
    _hooks: list[Any] = field(default_factory=list, init=False)

    def analyze(
        self,
        static_features: Tensor,
        temporal_features: Tensor,
        temporal_mask: Tensor | None = None,
        target_idx: int | None = None,
    ) -> AttentionMaps:
        """
        Extract and analyze attention weights for given input.

        Args:
            static_features: Static features [batch_size, n_static_features].
            temporal_features: Temporal features [batch_size, seq_len, n_temporal_features].
            temporal_mask: Attention mask [batch_size, seq_len].
            target_idx: Target index for class-specific analysis.

        Returns:
            AttentionMaps with extracted attention patterns.
        """
        self._attention_weights = []
        self._register_hooks()

        try:
            # Forward pass to collect attention weights
            self.model.eval()
            with torch.no_grad():
                _ = self.model(
                    static_features=static_features,
                    temporal_features=temporal_features,
                    temporal_mask=temporal_mask,
                )

            # Process collected attention weights
            attention_weights = torch.stack(self._attention_weights, dim=0)
            # Shape: [n_layers, batch_size, n_heads, seq_len, seq_len]

            # Average over batch dimension
            attention_weights = attention_weights.mean(dim=1)
            # Shape: [n_layers, n_heads, seq_len, seq_len]

            raw_weights = attention_weights.cpu().numpy()

            # Average over heads
            averaged_weights = raw_weights.mean(axis=1)
            # Shape: [n_layers, seq_len, seq_len]

            # Extract CLS token attention (first position)
            cls_attention = raw_weights[:, :, 0, :]
            # Shape: [n_layers, n_heads, seq_len]

            # Compute rollout if requested
            rollout_attention = None
            if self.include_rollout:
                rollout_attention = attention_rollout(
                    torch.from_numpy(averaged_weights)
                ).numpy()

            # Compute attention flow if requested
            flow_attention = None
            if self.include_flow:
                flow_attention = attention_flow(
                    torch.from_numpy(raw_weights)
                ).numpy()

            return AttentionMaps(
                raw_weights=raw_weights,
                averaged_weights=averaged_weights,
                cls_attention=cls_attention,
                rollout_attention=rollout_attention,
                flow_attention=flow_attention,
            )

        finally:
            self._remove_hooks()

    def _register_hooks(self) -> None:
        """Register forward hooks to capture attention weights."""
        def hook_fn(module: nn.Module, input: Any, output: Any) -> None:
            # Assuming output is (attended_values, attention_weights)
            if isinstance(output, tuple) and len(output) >= 2:
                attention_weights = output[1]
                if attention_weights is not None:
                    self._attention_weights.append(attention_weights.detach())

        # Register hooks on attention layers
        for name, module in self.model.named_modules():
            if "attention" in name.lower() and hasattr(module, "forward"):
                hook = module.register_forward_hook(hook_fn)
                self._hooks.append(hook)

    def _remove_hooks(self) -> None:
        """Remove registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def get_temporal_importance(
        self,
        attention_maps: AttentionMaps,
        method: str = "cls",
    ) -> NDArray[np.floating[Any]]:
        """
        Compute temporal importance scores from attention.

        Args:
            attention_maps: Extracted attention maps.
            method: Aggregation method.
                - 'cls': Use CLS token attention (default).
                - 'rollout': Use attention rollout.
                - 'flow': Use attention flow.
                - 'mean': Mean attention across all positions.

        Returns:
            Importance scores for each time step [seq_len].
        """
        if method == "cls":
            # Average CLS attention across layers and heads
            importance = attention_maps.cls_attention.mean(axis=(0, 1))
        elif method == "rollout" and attention_maps.rollout_attention is not None:
            # Use rollout from CLS position
            importance = attention_maps.rollout_attention[0, :]
        elif method == "flow" and attention_maps.flow_attention is not None:
            importance = attention_maps.flow_attention
        elif method == "mean":
            # Mean attention received by each position
            importance = attention_maps.averaged_weights.mean(axis=(0, 1))
        else:
            raise ValueError(f"Unknown method: {method}")

        # Normalize to sum to 1
        importance = importance / (importance.sum() + 1e-10)

        return importance

    def visualize_attention(
        self,
        attention_maps: AttentionMaps,
        layer_idx: int = -1,
        head_idx: int | None = None,
    ) -> dict[str, Any]:
        """
        Prepare attention data for visualization.

        Args:
            attention_maps: Extracted attention maps.
            layer_idx: Layer to visualize (-1 for last layer).
            head_idx: Specific head to visualize (None for averaged).

        Returns:
            Dictionary with visualization data.
        """
        if head_idx is not None:
            attention_matrix = attention_maps.raw_weights[layer_idx, head_idx]
        else:
            attention_matrix = attention_maps.averaged_weights[layer_idx]

        return {
            "attention_matrix": attention_matrix,
            "layer_idx": layer_idx if layer_idx >= 0 else attention_maps.raw_weights.shape[0] + layer_idx,
            "head_idx": head_idx,
            "seq_len": attention_matrix.shape[0],
            "is_averaged": head_idx is None,
        }


def extract_attention_weights(
    model: nn.Module,
    static_features: Tensor,
    temporal_features: Tensor,
    temporal_mask: Tensor | None = None,
) -> list[Tensor]:
    """
    Extract raw attention weights from model forward pass.

    Args:
        model: ARGUS model or transformer encoder.
        static_features: Static input features.
        temporal_features: Temporal input features.
        temporal_mask: Attention mask.

    Returns:
        List of attention weight tensors, one per layer.

    Example:
        >>> weights = extract_attention_weights(
        ...     model, static_data, temporal_data, mask
        ... )
        >>> print(f"Extracted {len(weights)} attention layers")
    """
    attention_weights = []

    def hook_fn(module: nn.Module, input: Any, output: Any) -> None:
        if isinstance(output, tuple) and len(output) >= 2:
            if output[1] is not None:
                attention_weights.append(output[1].detach().cpu())

    hooks = []
    for name, module in model.named_modules():
        if "attention" in name.lower():
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)

    try:
        model.eval()
        with torch.no_grad():
            _ = model(
                static_features=static_features,
                temporal_features=temporal_features,
                temporal_mask=temporal_mask,
            )
    finally:
        for hook in hooks:
            hook.remove()

    return attention_weights


def attention_rollout(
    attention_weights: Tensor,
    discard_ratio: float = 0.0,
    head_fusion: str = "mean",
) -> Tensor:
    """
    Compute attention rollout across transformer layers.

    Attention rollout accounts for the residual connections by
    multiplying attention matrices across layers.

    A_rollout = A_L @ A_{L-1} @ ... @ A_1

    Reference: Abnar & Zuidema, 2020

    Args:
        attention_weights: Attention weights [n_layers, seq_len, seq_len]
            or [n_layers, n_heads, seq_len, seq_len].
        discard_ratio: Ratio of lowest attention weights to discard.
            Default: 0.0
        head_fusion: How to fuse attention heads.
            - 'mean': Average across heads.
            - 'max': Maximum across heads.
            - 'min': Minimum across heads.
            Default: 'mean'

    Returns:
        Rolled-out attention matrix [seq_len, seq_len].

    Example:
        >>> rollout = attention_rollout(attention_weights)
        >>> # rollout[i, j] = contribution of position j to position i
    """
    # Handle 4D input (with heads)
    if attention_weights.dim() == 4:
        if head_fusion == "mean":
            attention_weights = attention_weights.mean(dim=1)
        elif head_fusion == "max":
            attention_weights = attention_weights.max(dim=1)[0]
        elif head_fusion == "min":
            attention_weights = attention_weights.min(dim=1)[0]
        else:
            raise ValueError(f"Unknown head_fusion: {head_fusion}")

    n_layers, seq_len, _ = attention_weights.shape

    # Add residual connection (identity matrix)
    result = torch.eye(seq_len, device=attention_weights.device)

    for layer_idx in range(n_layers):
        attention = attention_weights[layer_idx]

        # Optional: discard low attention weights
        if discard_ratio > 0:
            flat = attention.flatten()
            threshold_idx = int(flat.numel() * discard_ratio)
            threshold = flat.kthvalue(threshold_idx + 1)[0]
            attention = torch.where(
                attention > threshold,
                attention,
                torch.zeros_like(attention),
            )
            # Re-normalize
            attention = attention / (attention.sum(dim=-1, keepdim=True) + 1e-10)

        # Add identity for residual connection
        attention_with_residual = 0.5 * attention + 0.5 * torch.eye(
            seq_len, device=attention.device
        )

        # Normalize
        attention_with_residual = attention_with_residual / (
            attention_with_residual.sum(dim=-1, keepdim=True) + 1e-10
        )

        # Matrix multiplication for rollout
        result = torch.matmul(attention_with_residual, result)

    return result


def attention_flow(
    attention_weights: Tensor,
    output_idx: int = 0,
) -> Tensor:
    """
    Compute attention flow to measure input token importance.

    Attention flow aggregates attention from output position to all inputs,
    accounting for attention paths through multiple layers.

    Reference: Abnar & Zuidema, 2020

    Args:
        attention_weights: Attention weights [n_layers, n_heads, seq_len, seq_len].
        output_idx: Output position to analyze (default: 0 for CLS token).

    Returns:
        Input importance scores [seq_len].

    Example:
        >>> importance = attention_flow(weights, output_idx=0)
        >>> # importance[i] = contribution of input position i to output
    """
    if attention_weights.dim() == 3:
        attention_weights = attention_weights.unsqueeze(1)

    n_layers, n_heads, seq_len, _ = attention_weights.shape

    # Average over heads
    attention_avg = attention_weights.mean(dim=1)

    # Build graph adjacency matrix
    # nodes: (layer, position) pairs
    # edges: attention weights from layer l to layer l+1
    n_nodes = (n_layers + 1) * seq_len  # Input layer + transformer layers

    # Initialize adjacency matrix
    adjacency = torch.zeros(n_nodes, n_nodes, device=attention_weights.device)

    # Add edges from input to first layer
    for i in range(seq_len):
        adjacency[i, seq_len + i] = 1.0

    # Add attention edges between layers
    for layer_idx in range(n_layers):
        layer_offset = (layer_idx + 1) * seq_len
        next_layer_offset = (layer_idx + 2) * seq_len if layer_idx < n_layers - 1 else None

        for i in range(seq_len):
            for j in range(seq_len):
                src = layer_offset + j
                if next_layer_offset is not None:
                    dst = next_layer_offset + i
                    adjacency[src, dst] = attention_avg[layer_idx, i, j]

    # Compute maximum flow using iterative propagation
    # (simplified version)
    flow = torch.zeros(n_nodes, device=attention_weights.device)
    output_node = n_layers * seq_len + output_idx
    flow[output_node] = 1.0

    # Backward propagation of flow
    for layer_idx in range(n_layers - 1, -1, -1):
        layer_offset = (layer_idx + 1) * seq_len
        prev_layer_offset = layer_idx * seq_len

        for j in range(seq_len):
            total_flow = 0.0
            for i in range(seq_len):
                src = layer_offset + j
                dst_idx = (layer_idx + 2) * seq_len + i if layer_idx < n_layers - 1 else None
                if dst_idx is not None and dst_idx < n_nodes:
                    total_flow += flow[dst_idx] * adjacency[src, dst_idx]
                elif layer_idx == n_layers - 1:
                    total_flow += flow[layer_offset + i] * attention_avg[layer_idx, i, j]

            flow[prev_layer_offset + j] = total_flow

    # Return input layer importance
    return flow[:seq_len]


def head_importance(
    attention_weights: Tensor,
    gradient_weights: Tensor | None = None,
) -> Tensor:
    """
    Compute importance scores for attention heads.

    Based on gradient-weighted attention, identifies which heads
    are most important for model predictions.

    Args:
        attention_weights: Attention weights [n_layers, n_heads, seq_len, seq_len].
        gradient_weights: Gradients w.r.t. attention [same shape as attention_weights].
            If None, uses attention magnitude only.

    Returns:
        Head importance scores [n_layers, n_heads].

    Example:
        >>> importance = head_importance(attention, gradients)
        >>> top_heads = importance.flatten().topk(5)
    """
    if gradient_weights is not None:
        # Gradient-weighted importance
        importance = (attention_weights * gradient_weights).abs().sum(dim=(-2, -1))
    else:
        # Magnitude-based importance (variance as proxy)
        importance = attention_weights.var(dim=(-2, -1))

    return importance


def entropy_based_analysis(
    attention_weights: Tensor,
) -> dict[str, Tensor]:
    """
    Analyze attention patterns using entropy metrics.

    High entropy indicates diffuse attention (attending to many positions),
    while low entropy indicates focused attention (attending to few positions).

    Args:
        attention_weights: Attention weights [n_layers, n_heads, seq_len, seq_len].

    Returns:
        Dictionary with entropy metrics:
            - head_entropy: Entropy per head [n_layers, n_heads, seq_len].
            - layer_entropy: Average entropy per layer [n_layers].
            - sparsity: Attention sparsity per head [n_layers, n_heads].

    Example:
        >>> metrics = entropy_based_analysis(attention_weights)
        >>> print(f"Layer entropy: {metrics['layer_entropy']}")
    """
    # Compute entropy for each attention distribution
    eps = 1e-10
    attention_weights = attention_weights + eps  # Avoid log(0)

    # Normalize
    attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)

    # Entropy: -sum(p * log(p))
    entropy = -(attention_weights * torch.log(attention_weights)).sum(dim=-1)
    # Shape: [n_layers, n_heads, seq_len]

    # Layer-averaged entropy
    layer_entropy = entropy.mean(dim=(1, 2))

    # Sparsity: effective number of attended positions
    # Approximated as exp(entropy) / seq_len
    seq_len = attention_weights.shape[-1]
    sparsity = 1.0 - torch.exp(entropy) / seq_len
    sparsity = sparsity.mean(dim=-1)  # Average over query positions

    return {
        "head_entropy": entropy,
        "layer_entropy": layer_entropy,
        "sparsity": sparsity,
    }
