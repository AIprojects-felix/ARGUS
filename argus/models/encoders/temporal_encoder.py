"""
Temporal Transformer Encoder.

Transformer-based encoder for processing longitudinal EHR data.
Captures long-range dependencies and dynamic patterns in clinical time series.

Copyright 2024-2025 VTP Consortium
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from argus.models.encoders.position_encoding import (
    SinusoidalPositionalEncoding,
    LearnablePositionalEncoding,
    TimeAwarePositionalEncoding,
)


class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer encoder layer with Pre-LN architecture.

    This implements the "Pre-LN" (Layer Normalization before attention/FFN)
    variant which provides more stable training for deep networks.

    Architecture:
        x → LN → Self-Attention → Dropout → + → LN → FFN → Dropout → +
        |___________________________________↑   |_____________________↑

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        d_ff: Feed-forward network dimension.
        dropout: Dropout probability.
        attention_dropout: Dropout on attention weights.
        activation: Activation function in FFN.
        norm_first: Whether to apply LayerNorm before (True) or after (False) attention.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation: str = "gelu",
        norm_first: bool = True,
    ) -> None:
        super().__init__()

        self.norm_first = norm_first

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=attention_dropout,
            batch_first=True,
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        # Layer normalizations
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # For storing attention weights
        self._attn_weights: Tensor | None = None

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
        }
        return activations.get(activation.lower(), nn.GELU())

    def forward(
        self,
        x: Tensor,
        src_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Forward pass through encoder layer.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            src_mask: Attention mask [seq_len, seq_len]
            src_key_padding_mask: Key padding mask [batch_size, seq_len]
            return_attention: Whether to return attention weights.

        Returns:
            Output tensor [batch_size, seq_len, d_model]
            Optionally also returns attention weights.
        """
        if self.norm_first:
            # Pre-LN: LayerNorm before attention and FFN
            # Self-attention sublayer
            x_norm = self.norm1(x)
            attn_output, attn_weights = self.self_attn(
                x_norm, x_norm, x_norm,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                need_weights=True,
            )
            x = x + self.dropout1(attn_output)

            # FFN sublayer
            x_norm = self.norm2(x)
            x = x + self.dropout2(self.ffn(x_norm))
        else:
            # Post-LN: LayerNorm after attention and FFN
            attn_output, attn_weights = self.self_attn(
                x, x, x,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                need_weights=True,
            )
            x = self.norm1(x + self.dropout1(attn_output))
            x = self.norm2(x + self.dropout2(self.ffn(x)))

        self._attn_weights = attn_weights.detach()

        if return_attention:
            return x, attn_weights
        return x


class TemporalTransformerEncoder(nn.Module):
    """
    Temporal Transformer Encoder for longitudinal EHR data.

    Processes variable-length sequences of clinical measurements using
    self-attention to capture temporal dependencies and patterns.

    Key Features:
        - Sinusoidal or learnable positional encoding
        - Learnable [CLS] token for sequence-level representation
        - Learnable mask token for handling missing values
        - Support for irregular time intervals (time-aware encoding)

    Architecture:
        Input → Projection → [CLS] + Positional Encoding → Transformer Layers → [CLS] output

    Args:
        input_dim: Dimension of input features (number of clinical variables).
            Default: 180
        d_model: Model embedding dimension.
            Default: 256
        n_heads: Number of attention heads.
            Default: 8
        n_layers: Number of Transformer encoder layers.
            Default: 6
        d_ff: Feed-forward network dimension.
            Default: 1024
        dropout: Dropout probability.
            Default: 0.1
        attention_dropout: Dropout on attention weights.
            Default: 0.1
        max_seq_len: Maximum sequence length.
            Default: 365
        use_cls_token: Whether to use [CLS] token for pooling.
            Default: True
        use_mask_token: Whether to use learnable mask token for missing values.
            Default: True
        position_encoding: Type of positional encoding ('sinusoidal', 'learnable', 'time_aware').
            Default: 'sinusoidal'
        activation: Activation function in FFN.
            Default: 'gelu'
        norm_first: Whether to use Pre-LN architecture.
            Default: True

    Example:
        >>> encoder = TemporalTransformerEncoder(
        ...     input_dim=180,
        ...     d_model=256,
        ...     n_heads=8,
        ...     n_layers=6,
        ... )
        >>> x = torch.randn(32, 100, 180)  # [batch, seq_len, features]
        >>> h = encoder(x)  # [batch, d_model]
    """

    def __init__(
        self,
        input_dim: int = 180,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        max_seq_len: int = 365,
        use_cls_token: bool = True,
        use_mask_token: bool = True,
        position_encoding: str = "sinusoidal",
        activation: str = "gelu",
        norm_first: bool = True,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.use_cls_token = use_cls_token
        self.use_mask_token = use_mask_token

        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        # Learnable [CLS] token for sequence representation
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Learnable mask token for missing values
        if use_mask_token:
            self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Positional encoding
        effective_max_len = max_seq_len + 1 if use_cls_token else max_seq_len

        if position_encoding == "sinusoidal":
            self.pos_encoding = SinusoidalPositionalEncoding(
                d_model=d_model,
                max_len=effective_max_len,
                dropout=dropout,
            )
        elif position_encoding == "learnable":
            self.pos_encoding = LearnablePositionalEncoding(
                d_model=d_model,
                max_len=effective_max_len,
                dropout=dropout,
            )
        elif position_encoding == "time_aware":
            self.pos_encoding = TimeAwarePositionalEncoding(
                d_model=d_model,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown position encoding: {position_encoding}")

        self.position_encoding_type = position_encoding

        # Transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                attention_dropout=attention_dropout,
                activation=activation,
                norm_first=norm_first,
            )
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        feature_mask: Tensor | None = None,
        time_deltas: Tensor | None = None,
    ) -> Tensor:
        """
        Encode temporal features.

        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            mask: Boolean mask for valid positions [batch_size, seq_len]
                True indicates valid position, False indicates padding.
            feature_mask: Boolean mask for observed features [batch_size, seq_len, input_dim]
                True indicates observed, False indicates missing.
            time_deltas: Relative time values [batch_size, seq_len]
                Used only with time_aware position encoding.

        Returns:
            Encoded representation [batch_size, d_model]
            If use_cls_token=True, returns the [CLS] token representation.
            Otherwise, returns mean pooling over valid positions.
        """
        batch_size, seq_len, _ = x.shape

        # Project input features
        x = self.input_projection(x)  # [batch, seq_len, d_model]

        # Handle missing values with mask token
        if self.use_mask_token and feature_mask is not None:
            # Expand mask token for broadcasting
            mask_expanded = self.mask_token.expand(batch_size, seq_len, -1)

            # Create feature-level mask (any feature missing in a timestep)
            timestep_has_missing = ~feature_mask.any(dim=-1, keepdim=True)
            timestep_has_missing = timestep_has_missing.expand(-1, -1, self.d_model)

            # Replace timesteps with missing values
            x = torch.where(timestep_has_missing, mask_expanded, x)

        # Prepend [CLS] token
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)  # [batch, seq_len+1, d_model]

            # Update mask to include CLS token
            if mask is not None:
                cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=mask.device)
                mask = torch.cat([cls_mask, mask], dim=1)

        # Add positional encoding
        if self.position_encoding_type == "time_aware" and time_deltas is not None:
            # For time-aware encoding, we need to handle CLS token
            if self.use_cls_token:
                # Add zero time delta for CLS token
                cls_time = torch.zeros(batch_size, 1, device=time_deltas.device)
                time_deltas = torch.cat([cls_time, time_deltas], dim=1)
            x = self.pos_encoding(x, time_deltas)
        else:
            x = self.pos_encoding(x)

        # Prepare attention mask (convert to key_padding_mask format)
        # True means position should be ignored
        attn_mask = None
        if mask is not None:
            attn_mask = ~mask  # Invert: True -> ignore, False -> attend

        # Pass through Transformer layers
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=attn_mask)

        # Apply final layer norm
        x = self.final_norm(x)

        # Extract output representation
        if self.use_cls_token:
            # Return [CLS] token representation
            output = x[:, 0, :]  # [batch, d_model]
        else:
            # Mean pooling over valid positions
            if mask is not None:
                # Mask invalid positions
                mask_expanded = mask.unsqueeze(-1).expand_as(x)
                x = x * mask_expanded.float()
                output = x.sum(dim=1) / mask.sum(dim=1, keepdim=True).float()
            else:
                output = x.mean(dim=1)

        return output

    def get_attention_weights(
        self,
        x: Tensor,
        **kwargs: Any,
    ) -> list[Tensor]:
        """
        Extract attention weights from all layers.

        Args:
            x: Input tensor
            **kwargs: Additional arguments passed to forward

        Returns:
            List of attention weight tensors, one per layer.
        """
        batch_size, seq_len, _ = x.shape

        # Project input
        x = self.input_projection(x)

        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

        x = self.pos_encoding(x)

        # Collect attention weights
        attention_weights = []
        for layer in self.layers:
            x, attn = layer(x, return_attention=True)
            attention_weights.append(attn)

        return attention_weights

    @property
    def num_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"TemporalTransformerEncoder(\n"
            f"  input_dim={self.input_dim},\n"
            f"  d_model={self.d_model},\n"
            f"  n_heads={self.n_heads},\n"
            f"  n_layers={self.n_layers},\n"
            f"  max_seq_len={self.max_seq_len},\n"
            f"  parameters={self.num_parameters:,}\n"
            f")"
        )
