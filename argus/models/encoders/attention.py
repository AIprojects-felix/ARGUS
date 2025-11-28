"""
Attention Mechanisms.

Multi-head attention implementations for the Temporal Transformer encoder.

Copyright 2024-2025 VTP Consortium
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism.

    Implements scaled dot-product attention with multiple parallel attention heads,
    allowing the model to jointly attend to information from different
    representation subspaces.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Args:
        d_model: Total dimension of the model.
        n_heads: Number of parallel attention heads.
        dropout: Dropout probability on attention weights.
            Default: 0.1
        bias: Whether to include bias in projections.
            Default: True

    Attributes:
        head_dim: Dimension of each attention head (d_model // n_heads)
        scale: Scaling factor (1 / sqrt(head_dim))

    Example:
        >>> attn = MultiHeadAttention(d_model=256, n_heads=8)
        >>> x = torch.randn(32, 100, 256)  # [batch, seq_len, d_model]
        >>> output, weights = attn(x, x, x, return_attention=True)
        >>> output.shape  # [32, 100, 256]
        >>> weights.shape  # [32, 8, 100, 100]
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
    ) -> None:
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        # Query, Key, Value projections
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # For storing attention weights during inference
        self._attention_weights: Tensor | None = None

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None = None,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Compute multi-head attention.

        Args:
            query: Query tensor [batch_size, seq_len_q, d_model]
            key: Key tensor [batch_size, seq_len_k, d_model]
            value: Value tensor [batch_size, seq_len_v, d_model]
            mask: Optional attention mask [batch_size, seq_len_q, seq_len_k]
                or [batch_size, 1, 1, seq_len_k] for broadcasting.
                True values are masked (ignored in attention).
            return_attention: Whether to return attention weights.

        Returns:
            If return_attention is False:
                Output tensor [batch_size, seq_len_q, d_model]
            If return_attention is True:
                Tuple of (output, attention_weights)
                attention_weights shape: [batch_size, n_heads, seq_len_q, seq_len_k]
        """
        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.shape[1]

        # Project to Q, K, V
        q = self.q_proj(query)  # [batch, seq_q, d_model]
        k = self.k_proj(key)    # [batch, seq_k, d_model]
        v = self.v_proj(value)  # [batch, seq_k, d_model]

        # Reshape for multi-head attention
        # [batch, seq, d_model] -> [batch, seq, n_heads, head_dim] -> [batch, n_heads, seq, head_dim]
        q = q.view(batch_size, seq_len_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len_k, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len_k, self.n_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        # [batch, n_heads, seq_q, head_dim] @ [batch, n_heads, head_dim, seq_k]
        # -> [batch, n_heads, seq_q, seq_k]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply mask if provided
        if mask is not None:
            # Expand mask for broadcasting: [batch, 1, 1, seq_k] or [batch, 1, seq_q, seq_k]
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)

            # Mask positions get large negative value
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Store for potential inspection
        self._attention_weights = attn_weights.detach()

        # Apply attention to values
        # [batch, n_heads, seq_q, seq_k] @ [batch, n_heads, seq_k, head_dim]
        # -> [batch, n_heads, seq_q, head_dim]
        context = torch.matmul(attn_weights, v)

        # Reshape back
        # [batch, n_heads, seq_q, head_dim] -> [batch, seq_q, n_heads, head_dim] -> [batch, seq_q, d_model]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)

        # Final projection
        output = self.out_proj(context)

        if return_attention:
            return output, attn_weights
        return output

    def get_attention_weights(self) -> Tensor | None:
        """Return stored attention weights from last forward pass."""
        return self._attention_weights

    def __repr__(self) -> str:
        return (
            f"MultiHeadAttention(d_model={self.d_model}, "
            f"n_heads={self.n_heads}, head_dim={self.head_dim})"
        )


class EfficientAttention(nn.Module):
    """
    Memory-efficient attention using Flash Attention when available.

    Falls back to standard attention if Flash Attention is not available
    or not supported for the given configuration.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        dropout: Dropout probability.
        bias: Whether to use bias in projections.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        # Combined QKV projection for efficiency
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        # Check for Flash Attention availability
        self._use_flash = hasattr(F, "scaled_dot_product_attention")

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Compute self-attention.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
            return_attention: Whether to return attention weights
                (not supported with Flash Attention)

        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape

        # Combined QKV projection
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, n_heads, seq, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Use Flash Attention if available
        if self._use_flash and not return_attention:
            # Convert mask to attention bias format if needed
            attn_mask = None
            if mask is not None:
                attn_mask = mask.unsqueeze(1).unsqueeze(2)
                attn_mask = attn_mask.expand(-1, self.n_heads, seq_len, -1)

            output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
            attn_weights = None
        else:
            # Standard attention
            scale = self.head_dim ** -0.5
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

            if mask is not None:
                if mask.dim() == 2:
                    mask = mask.unsqueeze(1).unsqueeze(2)
                attn_scores = attn_scores.masked_fill(mask, float("-inf"))

            attn_weights = F.softmax(attn_scores, dim=-1)
            if self.training:
                attn_weights = F.dropout(attn_weights, p=self.dropout)

            output = torch.matmul(attn_weights, v)

        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(output)

        if return_attention:
            return output, attn_weights
        return output
