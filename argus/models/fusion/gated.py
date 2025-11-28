"""
Gated Fusion Module.

Learnable gating mechanism for adaptive feature combination.

Copyright 2024-2025 VTP Consortium
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class GatedFusion(nn.Module):
    """
    Gated feature fusion with learnable attention weights.

    Uses a gating mechanism to adaptively combine static and temporal features.
    The gate learns to weight the contribution of each modality based on the
    input content, allowing for patient-specific fusion strategies.

    Architecture:
        Gate = sigmoid(W_s * static + W_t * temporal + b)
        Fused = Gate * Proj_s(static) + (1 - Gate) * Proj_t(temporal)

    This allows:
    - Adaptive weighting based on feature content
    - Patient-specific modality emphasis
    - Smooth interpolation between static and temporal information

    Args:
        static_dim: Dimension of static features.
            Default: 256
        temporal_dim: Dimension of temporal features.
            Default: 256
        output_dim: Dimension of fused output.
            Default: 256
        hidden_dim: Hidden dimension for gate computation.
            If None, uses output_dim.
        dropout: Dropout probability.
            Default: 0.1
        gate_activation: Activation for gate ('sigmoid', 'softmax', 'tanh').
            Default: 'sigmoid'

    Example:
        >>> fusion = GatedFusion(static_dim=256, temporal_dim=256, output_dim=512)
        >>> static = torch.randn(32, 256)
        >>> temporal = torch.randn(32, 256)
        >>> fused, gate_values = fusion(static, temporal, return_gate=True)
        >>> fused.shape  # [32, 512]
        >>> gate_values.shape  # [32, 512]
    """

    def __init__(
        self,
        static_dim: int = 256,
        temporal_dim: int = 256,
        output_dim: int = 256,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
        gate_activation: str = "sigmoid",
    ) -> None:
        super().__init__()

        self.static_dim = static_dim
        self.temporal_dim = temporal_dim
        self.output_dim = output_dim

        if hidden_dim is None:
            hidden_dim = output_dim

        # Feature projections to common dimension
        self.static_projection = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        self.temporal_projection = nn.Sequential(
            nn.Linear(temporal_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        # Gate computation network
        self.gate_network = nn.Sequential(
            nn.Linear(static_dim + temporal_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        # Gate activation
        self.gate_activation = self._get_gate_activation(gate_activation)
        self._gate_activation_name = gate_activation

        # Output layer normalization
        self.output_norm = nn.LayerNorm(output_dim)

    def _get_gate_activation(self, activation: str) -> nn.Module:
        """Get gate activation function."""
        activations = {
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "softmax": nn.Softmax(dim=-1),
        }
        if activation.lower() not in activations:
            raise ValueError(f"Unknown gate activation: {activation}")
        return activations[activation.lower()]

    def forward(
        self,
        static: Tensor,
        temporal: Tensor,
        return_gate: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Fuse features using learned gates.

        Args:
            static: Static features [batch_size, static_dim]
            temporal: Temporal features [batch_size, temporal_dim]
            return_gate: Whether to return gate values for interpretability.

        Returns:
            Fused features [batch_size, output_dim]
            Optionally returns gate values [batch_size, output_dim]
        """
        # Project features to output dimension
        static_proj = self.static_projection(static)
        temporal_proj = self.temporal_projection(temporal)

        # Compute gate values
        combined = torch.cat([static, temporal], dim=-1)
        gate = self.gate_activation(self.gate_network(combined))

        # Gated fusion
        fused = gate * static_proj + (1 - gate) * temporal_proj

        # Apply output normalization
        fused = self.output_norm(fused)

        if return_gate:
            return fused, gate
        return fused

    def get_gate_importance(
        self,
        static: Tensor,
        temporal: Tensor,
    ) -> dict[str, Tensor]:
        """
        Analyze gate behavior for interpretability.

        Args:
            static: Static features [batch_size, static_dim]
            temporal: Temporal features [batch_size, temporal_dim]

        Returns:
            Dictionary with gate analysis:
                - gate_values: Raw gate values
                - static_weight: Mean contribution of static features
                - temporal_weight: Mean contribution of temporal features
        """
        combined = torch.cat([static, temporal], dim=-1)
        gate = self.gate_activation(self.gate_network(combined))

        return {
            "gate_values": gate,
            "static_weight": gate.mean(dim=-1),
            "temporal_weight": (1 - gate).mean(dim=-1),
        }

    @property
    def num_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"GatedFusion(\n"
            f"  static_dim={self.static_dim},\n"
            f"  temporal_dim={self.temporal_dim},\n"
            f"  output_dim={self.output_dim},\n"
            f"  gate_activation={self._gate_activation_name},\n"
            f"  parameters={self.num_parameters:,}\n"
            f")"
        )


class MultiGatedFusion(nn.Module):
    """
    Multi-gate fusion with separate gates for different target groups.

    Extends GatedFusion to learn separate gating strategies for different
    prediction tasks (e.g., genes vs TMB vs MSI), allowing task-specific
    feature weighting.

    Args:
        static_dim: Dimension of static features.
        temporal_dim: Dimension of temporal features.
        output_dim: Dimension per gate output.
        n_gates: Number of separate gates (e.g., for different task groups).
        dropout: Dropout probability.

    Example:
        >>> fusion = MultiGatedFusion(static_dim=256, temporal_dim=256, n_gates=4)
        >>> static = torch.randn(32, 256)
        >>> temporal = torch.randn(32, 256)
        >>> outputs = fusion(static, temporal)  # List of 4 tensors, each [32, 256]
    """

    def __init__(
        self,
        static_dim: int = 256,
        temporal_dim: int = 256,
        output_dim: int = 256,
        n_gates: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.n_gates = n_gates
        self.output_dim = output_dim

        # Create separate gated fusion modules for each task group
        self.gates = nn.ModuleList([
            GatedFusion(
                static_dim=static_dim,
                temporal_dim=temporal_dim,
                output_dim=output_dim,
                dropout=dropout,
            )
            for _ in range(n_gates)
        ])

    def forward(
        self,
        static: Tensor,
        temporal: Tensor,
        return_gates: bool = False,
    ) -> list[Tensor] | tuple[list[Tensor], list[Tensor]]:
        """
        Apply multiple gated fusions.

        Args:
            static: Static features [batch_size, static_dim]
            temporal: Temporal features [batch_size, temporal_dim]
            return_gates: Whether to return gate values.

        Returns:
            List of fused features for each gate.
            Optionally returns list of gate values.
        """
        outputs = []
        gate_values = []

        for gate_module in self.gates:
            if return_gates:
                fused, gate = gate_module(static, temporal, return_gate=True)
                outputs.append(fused)
                gate_values.append(gate)
            else:
                outputs.append(gate_module(static, temporal))

        if return_gates:
            return outputs, gate_values
        return outputs

    @property
    def num_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
