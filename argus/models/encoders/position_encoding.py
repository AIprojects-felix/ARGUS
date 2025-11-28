"""
Positional Encoding Modules.

Various positional encoding schemes for temporal sequence modeling:
- Sinusoidal: Fixed sine/cosine encodings (Vaswani et al., 2017)
- Learnable: Trainable position embeddings
- Time-Aware: Encodings based on actual time intervals

Copyright 2024-2025 VTP Consortium
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from "Attention is All You Need".

    Uses sine and cosine functions of different frequencies to encode
    positional information. This allows the model to easily learn to
    attend by relative positions.

    The encoding is defined as:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        d_model: Model embedding dimension.
        max_len: Maximum sequence length to pre-compute.
            Default: 5000
        dropout: Dropout probability applied after adding position encoding.
            Default: 0.1

    Attributes:
        pe: Pre-computed positional encoding buffer of shape [1, max_len, d_model]

    Example:
        >>> pe = SinusoidalPositionalEncoding(d_model=256, max_len=1000)
        >>> x = torch.randn(32, 100, 256)  # [batch, seq_len, d_model]
        >>> x_with_pe = pe(x)  # Same shape, with position info added
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Compute div_term for different frequencies
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sine to even indices, cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension and register as buffer (not a parameter)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Add positional encoding to input tensor.

        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]

        Returns:
            Tensor with positional encoding added, same shape as input.

        Raises:
            RuntimeError: If sequence length exceeds max_len.
        """
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise RuntimeError(
                f"Sequence length {seq_len} exceeds maximum length {self.max_len}"
            )

        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)

    def __repr__(self) -> str:
        return f"SinusoidalPositionalEncoding(d_model={self.d_model}, max_len={self.max_len})"


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional embeddings.

    Unlike sinusoidal encoding, these embeddings are learned during training,
    allowing the model to discover optimal position representations.

    Args:
        d_model: Model embedding dimension.
        max_len: Maximum sequence length.
            Default: 5000
        dropout: Dropout probability.
            Default: 0.1

    Example:
        >>> pe = LearnablePositionalEncoding(d_model=256, max_len=1000)
        >>> x = torch.randn(32, 100, 256)
        >>> x_with_pe = pe(x)
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)

        # Learnable position embeddings
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x: Tensor) -> Tensor:
        """
        Add learnable positional encoding to input.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            Tensor with position encoding added.
        """
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise RuntimeError(
                f"Sequence length {seq_len} exceeds maximum length {self.max_len}"
            )

        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)

    def __repr__(self) -> str:
        return f"LearnablePositionalEncoding(d_model={self.d_model}, max_len={self.max_len})"


class TimeAwarePositionalEncoding(nn.Module):
    """
    Time-aware positional encoding for irregularly sampled sequences.

    Instead of using ordinal positions, this encoding uses actual time
    intervals (e.g., days since first visit) to encode temporal information.
    This is particularly useful for EHR data where visits are irregularly spaced.

    The time interval is projected to d_model dimensions and added to the input.

    Args:
        d_model: Model embedding dimension.
        dropout: Dropout probability.
            Default: 0.1
        time_scale: Scaling factor for time values.
            Default: 1.0

    Example:
        >>> pe = TimeAwarePositionalEncoding(d_model=256)
        >>> x = torch.randn(32, 100, 256)
        >>> time_deltas = torch.arange(100).float().unsqueeze(0).expand(32, -1) / 180  # Normalized
        >>> x_with_pe = pe(x, time_deltas)
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        time_scale: float = 1.0,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.time_scale = time_scale
        self.dropout = nn.Dropout(p=dropout)

        # Project time values to d_model dimensions
        self.time_projection = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )

    def forward(self, x: Tensor, time_deltas: Tensor) -> Tensor:
        """
        Add time-aware positional encoding.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            time_deltas: Relative time values [batch_size, seq_len]
                Should be normalized (e.g., to [0, 1] within observation window)

        Returns:
            Tensor with time-aware encoding added.
        """
        # Scale and project time values
        time_deltas = time_deltas * self.time_scale
        time_encoding = self.time_projection(time_deltas.unsqueeze(-1))

        x = x + time_encoding
        return self.dropout(x)

    def __repr__(self) -> str:
        return f"TimeAwarePositionalEncoding(d_model={self.d_model})"


class RotaryPositionalEncoding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Implements rotary position embeddings from "RoFormer: Enhanced Transformer
    with Rotary Position Embedding" (Su et al., 2021).

    RoPE encodes position information by rotating query and key vectors,
    allowing the model to naturally capture relative positions.

    Args:
        d_model: Model dimension (must be even).
        max_len: Maximum sequence length.
            Default: 5000
        base: Base for computing rotary frequencies.
            Default: 10000

    Note:
        This encoding is applied differently - it rotates Q and K vectors
        rather than being added to the input embeddings.
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        base: int = 10000,
    ) -> None:
        super().__init__()

        if d_model % 2 != 0:
            raise ValueError("d_model must be even for RoPE")

        self.d_model = d_model
        self.max_len = max_len
        self.base = base

        # Compute inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, d_model, 2).float() / d_model)
        )
        self.register_buffer("inv_freq", inv_freq)

        # Pre-compute sin/cos for positions
        self._build_cache(max_len)

    def _build_cache(self, seq_len: int) -> None:
        """Pre-compute rotation matrices for given sequence length."""
        positions = torch.arange(seq_len, dtype=torch.float)
        freqs = torch.outer(positions, self.inv_freq)

        # Repeat for pairs of dimensions
        emb = torch.cat([freqs, freqs], dim=-1)

        self.register_buffer("cos_cache", emb.cos().unsqueeze(0))
        self.register_buffer("sin_cache", emb.sin().unsqueeze(0))

    def _rotate_half(self, x: Tensor) -> Tensor:
        """Rotate half the hidden dims."""
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        positions: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Apply rotary position embedding to query and key tensors.

        Args:
            q: Query tensor [batch, heads, seq_len, head_dim]
            k: Key tensor [batch, heads, seq_len, head_dim]
            positions: Optional custom positions [batch, seq_len]

        Returns:
            Tuple of (rotated_q, rotated_k) with same shapes as input.
        """
        seq_len = q.size(2)

        if seq_len > self.max_len:
            self._build_cache(seq_len)

        cos = self.cos_cache[:, :seq_len, :]
        sin = self.sin_cache[:, :seq_len, :]

        # Reshape for broadcasting
        cos = cos.unsqueeze(1)  # [1, 1, seq_len, d_model]
        sin = sin.unsqueeze(1)

        # Apply rotation
        q_rotated = (q * cos) + (self._rotate_half(q) * sin)
        k_rotated = (k * cos) + (self._rotate_half(k) * sin)

        return q_rotated, k_rotated

    def __repr__(self) -> str:
        return f"RotaryPositionalEncoding(d_model={self.d_model}, max_len={self.max_len})"
