"""
ARGUS: AI-based Routine Genomic Understanding System.

Main model architecture implementing a dual-encoder framework for
non-invasive pan-cancer genomic profiling from longitudinal EHR data.

The architecture consists of:
1. Static Encoder: MLP for demographic and categorical features
2. Temporal Transformer Encoder: Self-attention for longitudinal clinical data
3. Feature Fusion: Concatenation of static and temporal representations
4. Multi-Label Prediction Head: Simultaneous prediction of genomic targets

Reference:
    Wu et al. "A Pan-Cancer AI Framework for Non-Invasive Genomic Profiling"

Copyright 2024-2025 VTP Consortium
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from argus.models.encoders import StaticEncoder, TemporalTransformerEncoder
from argus.models.fusion import ConcatFusion, CrossAttentionFusion, GatedFusion
from argus.models.heads import MultiLabelClassificationHead


class ARGUS(nn.Module):
    """
    ARGUS: AI-based Routine Genomic Understanding System.

    A dual-encoder deep learning architecture for inferring tumor genomic
    alterations from longitudinal electronic health record (EHR) data.

    Architecture Overview:
        Static Features (demographics) ──► Static Encoder (MLP) ──┐
                                                                   ├──► Fusion ──► Prediction Head ──► Predictions
        Temporal Features (EHR) ──────► Temporal Encoder (Transformer) ──┘

    The model processes:
    - Static features: Age, sex, cancer type (one-hot encoded)
    - Temporal features: Longitudinal clinical measurements (labs, vitals, markers)

    And predicts:
    - 40+ actionable driver gene mutations
    - TMB (Tumor Mutational Burden) status
    - MSI (Microsatellite Instability) status
    - PD-L1 expression levels

    Args:
        static_dim: Dimension of static input features.
            Default: 18 (age + sex + 16 cancer types)
        temporal_dim: Dimension of temporal input features.
            Default: 180 (180+ clinical variables)
        d_model: Model embedding dimension.
            Default: 256
        n_heads: Number of attention heads in Transformer.
            Default: 8
        n_layers: Number of Transformer encoder layers.
            Default: 6
        d_ff: Feed-forward network dimension.
            Default: 1024 (4 * d_model)
        n_targets: Number of prediction targets.
            Default: 43 (40 genes + TMB + MSI + PD-L1)
        dropout: Dropout probability.
            Default: 0.1
        max_seq_len: Maximum sequence length for temporal data.
            Default: 365 (days)
        fusion_method: Feature fusion method ('concat', 'cross_attention', 'gated').
            Default: 'concat'
        use_cls_token: Whether to use [CLS] token for temporal encoding.
            Default: True
        use_mask_token: Whether to use learnable mask token for missing values.
            Default: True
        position_encoding: Type of positional encoding ('sinusoidal', 'learnable').
            Default: 'sinusoidal'

    Attributes:
        static_encoder: MLP encoder for static features
        temporal_encoder: Transformer encoder for temporal features
        fusion: Feature fusion module
        prediction_head: Multi-label classification head

    Example:
        >>> import torch
        >>> from argus.models import ARGUS
        >>>
        >>> # Initialize model
        >>> model = ARGUS(
        ...     static_dim=18,
        ...     temporal_dim=180,
        ...     d_model=256,
        ...     n_heads=8,
        ...     n_layers=6,
        ...     n_targets=43,
        ... )
        >>>
        >>> # Prepare input tensors
        >>> batch_size = 32
        >>> seq_len = 100
        >>> static_x = torch.randn(batch_size, 18)
        >>> temporal_x = torch.randn(batch_size, seq_len, 180)
        >>>
        >>> # Forward pass
        >>> output = model(static_x, temporal_x)
        >>> predictions = output['predictions']  # Shape: [32, 43]
        >>> logits = output['logits']  # Shape: [32, 43]
        >>>
        >>> # Get embeddings for interpretability
        >>> output = model(static_x, temporal_x, return_embeddings=True)
        >>> embeddings = output['embeddings']  # Shape: [32, 512]

    Note:
        - Input tensors should be normalized/standardized
        - Missing values in temporal features should be masked (NaN or specified mask value)
        - The model uses mixed precision training by default when available
    """

    def __init__(
        self,
        static_dim: int = 18,
        temporal_dim: int = 180,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int | None = None,
        n_targets: int = 43,
        dropout: float = 0.1,
        attention_dropout: float | None = None,
        max_seq_len: int = 365,
        fusion_method: str = "concat",
        use_cls_token: bool = True,
        use_mask_token: bool = True,
        position_encoding: str = "sinusoidal",
        activation: str = "gelu",
        norm_first: bool = True,
        static_hidden_dims: list[int] | None = None,
        head_hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()

        # Store configuration
        self.static_dim = static_dim
        self.temporal_dim = temporal_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff if d_ff is not None else 4 * d_model
        self.n_targets = n_targets
        self.dropout = dropout
        self.attention_dropout = attention_dropout if attention_dropout is not None else dropout
        self.max_seq_len = max_seq_len
        self.fusion_method = fusion_method

        # Default hidden dimensions
        if static_hidden_dims is None:
            static_hidden_dims = [128, d_model]
        if head_hidden_dims is None:
            head_hidden_dims = [512, 256]

        # =================================================================
        # Static Encoder
        # =================================================================
        self.static_encoder = StaticEncoder(
            input_dim=static_dim,
            hidden_dims=static_hidden_dims,
            output_dim=d_model,
            dropout=dropout,
            activation=activation,
            batch_norm=True,
        )

        # =================================================================
        # Temporal Transformer Encoder
        # =================================================================
        self.temporal_encoder = TemporalTransformerEncoder(
            input_dim=temporal_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=self.d_ff,
            dropout=dropout,
            attention_dropout=self.attention_dropout,
            max_seq_len=max_seq_len,
            use_cls_token=use_cls_token,
            use_mask_token=use_mask_token,
            position_encoding=position_encoding,
            activation=activation,
            norm_first=norm_first,
        )

        # =================================================================
        # Feature Fusion
        # =================================================================
        if fusion_method == "concat":
            self.fusion = ConcatFusion()
            fusion_output_dim = 2 * d_model
        elif fusion_method == "cross_attention":
            self.fusion = CrossAttentionFusion(
                d_model=d_model,
                n_heads=4,
                dropout=dropout,
            )
            fusion_output_dim = 2 * d_model
        elif fusion_method == "gated":
            self.fusion = GatedFusion(
                dim=d_model,
                dropout=dropout,
            )
            fusion_output_dim = d_model
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

        # =================================================================
        # Prediction Head
        # =================================================================
        self.prediction_head = MultiLabelClassificationHead(
            input_dim=fusion_output_dim,
            hidden_dims=head_hidden_dims,
            n_targets=n_targets,
            dropout=dropout * 2,  # Higher dropout in head
            activation=activation,
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize model weights using Xavier/Glorot initialization."""
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        # Special initialization for embeddings
        if hasattr(self.temporal_encoder, "cls_token"):
            nn.init.normal_(self.temporal_encoder.cls_token, std=0.02)
        if hasattr(self.temporal_encoder, "mask_token"):
            nn.init.normal_(self.temporal_encoder.mask_token, std=0.02)

    def forward(
        self,
        static_features: Tensor,
        temporal_features: Tensor,
        temporal_mask: Tensor | None = None,
        feature_mask: Tensor | None = None,
        time_deltas: Tensor | None = None,
        return_embeddings: bool = False,
    ) -> dict[str, Tensor]:
        """
        Forward pass through the ARGUS model.

        Args:
            static_features: Static patient features.
                Shape: [batch_size, static_dim]
            temporal_features: Longitudinal clinical measurements.
                Shape: [batch_size, seq_len, temporal_dim]
            temporal_mask: Boolean mask for valid time points.
                Shape: [batch_size, seq_len]
                True indicates valid position, False indicates padding.
            feature_mask: Boolean mask for observed features.
                Shape: [batch_size, seq_len, temporal_dim]
                True indicates observed, False indicates missing.
            time_deltas: Relative time encoding for each visit.
                Shape: [batch_size, seq_len]
                Values should be normalized to [0, 1].
            return_embeddings: Whether to return intermediate embeddings
                for interpretability analysis.

        Returns:
            Dictionary containing:
                - 'predictions': Sigmoid-activated predictions [batch_size, n_targets]
                - 'logits': Raw logits before activation [batch_size, n_targets]
                - 'embeddings': (optional) Fused representations [batch_size, fusion_dim]
                - 'static_embeddings': (optional) Static encoder output [batch_size, d_model]
                - 'temporal_embeddings': (optional) Temporal encoder output [batch_size, d_model]

        Raises:
            ValueError: If input dimensions don't match expected dimensions.
        """
        # Validate input shapes
        batch_size = static_features.size(0)
        if static_features.size(1) != self.static_dim:
            raise ValueError(
                f"Expected static_dim={self.static_dim}, got {static_features.size(1)}"
            )
        if temporal_features.size(2) != self.temporal_dim:
            raise ValueError(
                f"Expected temporal_dim={self.temporal_dim}, got {temporal_features.size(2)}"
            )

        # =================================================================
        # Encode static features
        # =================================================================
        h_static = self.static_encoder(static_features)  # [batch, d_model]

        # =================================================================
        # Encode temporal features
        # =================================================================
        h_temporal = self.temporal_encoder(
            temporal_features,
            mask=temporal_mask,
            feature_mask=feature_mask,
            time_deltas=time_deltas,
        )  # [batch, d_model]

        # =================================================================
        # Fuse representations
        # =================================================================
        z = self.fusion(h_static, h_temporal)  # [batch, fusion_dim]

        # =================================================================
        # Generate predictions
        # =================================================================
        logits = self.prediction_head(z)  # [batch, n_targets]
        predictions = torch.sigmoid(logits)

        # Build output dictionary
        output = {
            "predictions": predictions,
            "logits": logits,
        }

        if return_embeddings:
            output["embeddings"] = z
            output["static_embeddings"] = h_static
            output["temporal_embeddings"] = h_temporal

        return output

    def get_embeddings(
        self,
        static_features: Tensor,
        temporal_features: Tensor,
        **kwargs: Any,
    ) -> Tensor:
        """
        Extract fused embeddings for interpretability analysis.

        This is a convenience method for extracting latent representations
        used in UMAP visualization and clustering analysis.

        Args:
            static_features: Static patient features [batch_size, static_dim]
            temporal_features: Temporal EHR features [batch_size, seq_len, temporal_dim]
            **kwargs: Additional arguments passed to forward()

        Returns:
            Fused latent representations [batch_size, fusion_dim]
        """
        output = self.forward(
            static_features,
            temporal_features,
            return_embeddings=True,
            **kwargs,
        )
        return output["embeddings"]

    def get_attention_weights(
        self,
        static_features: Tensor,
        temporal_features: Tensor,
        **kwargs: Any,
    ) -> list[Tensor]:
        """
        Extract attention weights from Transformer layers.

        Useful for analyzing which time points the model focuses on.

        Args:
            static_features: Static features [batch_size, static_dim]
            temporal_features: Temporal features [batch_size, seq_len, temporal_dim]
            **kwargs: Additional arguments

        Returns:
            List of attention weight tensors, one per layer.
            Each tensor has shape [batch_size, n_heads, seq_len, seq_len]
        """
        return self.temporal_encoder.get_attention_weights(
            temporal_features, **kwargs
        )

    @property
    def num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def config(self) -> dict[str, Any]:
        """Return model configuration as dictionary."""
        return {
            "static_dim": self.static_dim,
            "temporal_dim": self.temporal_dim,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "d_ff": self.d_ff,
            "n_targets": self.n_targets,
            "dropout": self.dropout,
            "max_seq_len": self.max_seq_len,
            "fusion_method": self.fusion_method,
        }

    def __repr__(self) -> str:
        """Return string representation of the model."""
        return (
            f"ARGUS(\n"
            f"  static_dim={self.static_dim},\n"
            f"  temporal_dim={self.temporal_dim},\n"
            f"  d_model={self.d_model},\n"
            f"  n_heads={self.n_heads},\n"
            f"  n_layers={self.n_layers},\n"
            f"  n_targets={self.n_targets},\n"
            f"  fusion={self.fusion_method},\n"
            f"  parameters={self.num_parameters:,}\n"
            f")"
        )
