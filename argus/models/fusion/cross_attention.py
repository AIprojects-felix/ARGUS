"""
Cross-Attention Fusion Module.

Attention-based feature interaction for sophisticated fusion.

Copyright 2024-2025 VTP Consortium
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention based feature fusion.

    Uses cross-attention mechanisms to allow static and temporal features
    to interact and selectively attend to each other, enabling the model
    to learn complex multimodal relationships.

    Architecture:
        Static attends to Temporal → S2T
        Temporal attends to Static → T2S
        Combine: [S2T; T2S; static; temporal] → Projection

    This bidirectional attention allows:
    - Static features to be contextualized by temporal patterns
    - Temporal features to be modulated by static characteristics

    Args:
        static_dim: Dimension of static features.
            Default: 256
        temporal_dim: Dimension of temporal features.
            Default: 256
        output_dim: Dimension of fused output.
            Default: 256
        n_heads: Number of attention heads.
            Default: 8
        dropout: Dropout probability.
            Default: 0.1
        bidirectional: Whether to use bidirectional attention.
            Default: True

    Example:
        >>> fusion = CrossAttentionFusion(static_dim=256, temporal_dim=256)
        >>> static = torch.randn(32, 256)
        >>> temporal = torch.randn(32, 256)
        >>> fused = fusion(static, temporal)  # [32, 256]
    """

    def __init__(
        self,
        static_dim: int = 256,
        temporal_dim: int = 256,
        output_dim: int = 256,
        n_heads: int = 8,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()

        self.static_dim = static_dim
        self.temporal_dim = temporal_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.bidirectional = bidirectional

        # Ensure dimensions are compatible with attention heads
        assert static_dim % n_heads == 0, "static_dim must be divisible by n_heads"
        assert temporal_dim % n_heads == 0, "temporal_dim must be divisible by n_heads"

        # Project to common dimension if needed
        common_dim = max(static_dim, temporal_dim)
        self.static_proj = nn.Linear(static_dim, common_dim) if static_dim != common_dim else nn.Identity()
        self.temporal_proj = nn.Linear(temporal_dim, common_dim) if temporal_dim != common_dim else nn.Identity()

        # Cross-attention: Static attends to Temporal
        self.static_to_temporal = nn.MultiheadAttention(
            embed_dim=common_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Cross-attention: Temporal attends to Static (if bidirectional)
        if bidirectional:
            self.temporal_to_static = nn.MultiheadAttention(
                embed_dim=common_dim,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True,
            )

        # Layer normalization
        self.norm_static = nn.LayerNorm(common_dim)
        self.norm_temporal = nn.LayerNorm(common_dim)

        # Output projection
        fusion_dim = common_dim * 4 if bidirectional else common_dim * 3
        self.output_projection = nn.Sequential(
            nn.Linear(fusion_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
        )

        # Store common dim for repr
        self._common_dim = common_dim

    def forward(
        self,
        static: Tensor,
        temporal: Tensor,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """
        Fuse features using cross-attention.

        Args:
            static: Static features [batch_size, static_dim]
            temporal: Temporal features [batch_size, temporal_dim]
            return_attention: Whether to return attention weights.

        Returns:
            Fused features [batch_size, output_dim]
            Optionally returns attention weights dictionary.
        """
        # Project to common dimension
        static_proj = self.static_proj(static)
        temporal_proj = self.temporal_proj(temporal)

        # Add sequence dimension for attention (treat as single-token sequences)
        static_seq = static_proj.unsqueeze(1)  # [batch, 1, dim]
        temporal_seq = temporal_proj.unsqueeze(1)  # [batch, 1, dim]

        # Static attends to temporal
        s2t, s2t_weights = self.static_to_temporal(
            query=static_seq,
            key=temporal_seq,
            value=temporal_seq,
            need_weights=True,
        )
        s2t = self.norm_static(s2t.squeeze(1) + static_proj)

        attention_weights = {"static_to_temporal": s2t_weights}

        if self.bidirectional:
            # Temporal attends to static
            t2s, t2s_weights = self.temporal_to_static(
                query=temporal_seq,
                key=static_seq,
                value=static_seq,
                need_weights=True,
            )
            t2s = self.norm_temporal(t2s.squeeze(1) + temporal_proj)
            attention_weights["temporal_to_static"] = t2s_weights

            # Combine all representations
            combined = torch.cat([s2t, t2s, static_proj, temporal_proj], dim=-1)
        else:
            # Combine with static-to-temporal attention only
            combined = torch.cat([s2t, static_proj, temporal_proj], dim=-1)

        # Project to output dimension
        fused = self.output_projection(combined)

        if return_attention:
            return fused, attention_weights
        return fused

    @property
    def num_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"CrossAttentionFusion(\n"
            f"  static_dim={self.static_dim},\n"
            f"  temporal_dim={self.temporal_dim},\n"
            f"  output_dim={self.output_dim},\n"
            f"  n_heads={self.n_heads},\n"
            f"  bidirectional={self.bidirectional},\n"
            f"  parameters={self.num_parameters:,}\n"
            f")"
        )
