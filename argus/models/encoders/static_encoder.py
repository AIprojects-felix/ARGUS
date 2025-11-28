"""
Static Feature Encoder.

Multi-layer perceptron (MLP) encoder for time-invariant patient characteristics
such as demographics and cancer type.

Copyright 2024-2025 VTP Consortium
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class StaticEncoder(nn.Module):
    """
    MLP-based encoder for static patient features.

    Transforms time-invariant patient characteristics (demographics, cancer type)
    into a high-dimensional latent representation suitable for fusion with
    temporal features.

    Architecture:
        Input → Linear → BN/LN → Activation → Dropout → ... → Output

    Args:
        input_dim: Dimension of input static features.
            Default: 18 (age + sex + 16 cancer types)
        hidden_dims: List of hidden layer dimensions.
            Default: [128, 256]
        output_dim: Dimension of output representation.
            If None, uses last element of hidden_dims.
        dropout: Dropout probability between layers.
            Default: 0.1
        activation: Activation function ('relu', 'gelu', 'silu', 'leaky_relu').
            Default: 'gelu'
        batch_norm: Whether to use batch normalization.
            Default: True
        layer_norm: Whether to use layer normalization (mutually exclusive with batch_norm).
            Default: False

    Example:
        >>> encoder = StaticEncoder(
        ...     input_dim=18,
        ...     hidden_dims=[128, 256],
        ...     dropout=0.1
        ... )
        >>> x = torch.randn(32, 18)
        >>> h = encoder(x)  # Shape: [32, 256]
    """

    def __init__(
        self,
        input_dim: int = 18,
        hidden_dims: list[int] | None = None,
        output_dim: int | None = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        batch_norm: bool = True,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 256]

        if output_dim is None:
            output_dim = hidden_dims[-1]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        # Validate normalization options
        if batch_norm and layer_norm:
            raise ValueError("Cannot use both batch_norm and layer_norm")

        # Get activation function
        self.activation_fn = self._get_activation(activation)

        # Build MLP layers
        layers: list[nn.Module] = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Normalization
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))

            # Activation
            layers.append(self.activation_fn)

            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        # Final projection if output_dim differs from last hidden_dim
        if output_dim != hidden_dims[-1]:
            layers.append(nn.Linear(hidden_dims[-1], output_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(output_dim))

        self.mlp = nn.Sequential(*layers)

        # Output layer norm for better gradient flow
        self.output_norm = nn.LayerNorm(output_dim)

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "leaky_relu": nn.LeakyReLU(0.1),
            "elu": nn.ELU(),
            "tanh": nn.Tanh(),
        }
        if activation.lower() not in activations:
            raise ValueError(f"Unknown activation: {activation}")
        return activations[activation.lower()]

    def forward(self, x: Tensor) -> Tensor:
        """
        Encode static features.

        Args:
            x: Static feature tensor of shape [batch_size, input_dim]

        Returns:
            Encoded representation of shape [batch_size, output_dim]
        """
        h = self.mlp(x)
        h = self.output_norm(h)
        return h

    @property
    def num_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"StaticEncoder(\n"
            f"  input_dim={self.input_dim},\n"
            f"  hidden_dims={self.hidden_dims},\n"
            f"  output_dim={self.output_dim},\n"
            f"  parameters={self.num_parameters:,}\n"
            f")"
        )
