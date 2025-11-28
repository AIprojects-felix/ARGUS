"""
Concatenation Fusion Module.

Simple yet effective feature fusion via concatenation and projection.

Copyright 2024-2025 VTP Consortium
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class ConcatFusion(nn.Module):
    """
    Concatenation-based feature fusion.

    Combines static and temporal features through concatenation followed
    by a projection network. This simple approach provides a strong baseline
    and allows the model to learn complex feature interactions.

    Architecture:
        [static; temporal] → Linear → LayerNorm → Activation → Dropout → Linear → Output

    Args:
        static_dim: Dimension of static feature embeddings.
            Default: 256
        temporal_dim: Dimension of temporal feature embeddings.
            Default: 256
        output_dim: Dimension of fused output.
            Default: 256
        hidden_dim: Hidden layer dimension in projection network.
            If None, uses output_dim * 2.
        dropout: Dropout probability.
            Default: 0.1
        activation: Activation function ('relu', 'gelu', 'silu').
            Default: 'gelu'

    Example:
        >>> fusion = ConcatFusion(static_dim=256, temporal_dim=256, output_dim=512)
        >>> static = torch.randn(32, 256)
        >>> temporal = torch.randn(32, 256)
        >>> fused = fusion(static, temporal)  # [32, 512]
    """

    def __init__(
        self,
        static_dim: int = 256,
        temporal_dim: int = 256,
        output_dim: int = 256,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()

        self.static_dim = static_dim
        self.temporal_dim = temporal_dim
        self.output_dim = output_dim

        if hidden_dim is None:
            hidden_dim = output_dim * 2

        # Get activation function
        activation_fn = self._get_activation(activation)

        # Projection network
        self.projection = nn.Sequential(
            nn.Linear(static_dim + temporal_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            activation_fn,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "leaky_relu": nn.LeakyReLU(0.1),
        }
        return activations.get(activation.lower(), nn.GELU())

    def forward(self, static: Tensor, temporal: Tensor) -> Tensor:
        """
        Fuse static and temporal features via concatenation.

        Args:
            static: Static feature embeddings [batch_size, static_dim]
            temporal: Temporal feature embeddings [batch_size, temporal_dim]

        Returns:
            Fused features [batch_size, output_dim]
        """
        # Concatenate along feature dimension
        import torch
        combined = torch.cat([static, temporal], dim=-1)

        # Project to output dimension
        fused = self.projection(combined)

        return fused

    @property
    def num_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"ConcatFusion(\n"
            f"  static_dim={self.static_dim},\n"
            f"  temporal_dim={self.temporal_dim},\n"
            f"  output_dim={self.output_dim},\n"
            f"  parameters={self.num_parameters:,}\n"
            f")"
        )
