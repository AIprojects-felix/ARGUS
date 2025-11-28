"""
Multi-Label Classification Head.

Prediction head for multi-label binary classification tasks.

Copyright 2024-2025 VTP Consortium
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class MultiLabelClassificationHead(nn.Module):
    """
    Multi-label classification head for genomic predictions.

    Produces independent binary predictions for each target (gene mutations,
    MSI status, PD-L1 expression). Uses a multi-layer architecture with
    optional shared representations.

    Architecture Options:
        1. Independent: Separate classifiers for each target
        2. Shared-then-split: Shared layers followed by target-specific outputs
        3. Fully shared: Single shared network with multi-output final layer

    Args:
        input_dim: Dimension of input features.
            Default: 256
        n_targets: Number of prediction targets.
            Default: 43 (40 genes + TMB + MSI + PD-L1)
        hidden_dims: List of hidden layer dimensions.
            Default: [512, 256]
        dropout: Dropout probability.
            Default: 0.1
        activation: Activation function.
            Default: 'gelu'
        use_batch_norm: Whether to use batch normalization.
            Default: True
        share_layers: Number of shared layers before target-specific outputs.
            Default: 2 (all layers shared)

    Example:
        >>> head = MultiLabelClassificationHead(input_dim=256, n_targets=43)
        >>> features = torch.randn(32, 256)
        >>> logits = head(features)  # [32, 43]
    """

    def __init__(
        self,
        input_dim: int = 256,
        n_targets: int = 43,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_batch_norm: bool = True,
        share_layers: int = 2,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256]

        self.input_dim = input_dim
        self.n_targets = n_targets
        self.hidden_dims = hidden_dims

        # Get activation function
        activation_fn = self._get_activation(activation)

        # Build shared layers
        layers: list[nn.Module] = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims[:share_layers]):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation_fn)
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.shared_layers = nn.Sequential(*layers) if layers else nn.Identity()

        # Final output layer
        self.output_layer = nn.Linear(prev_dim, n_targets)

        # Initialize output layer with smaller weights for better initial predictions
        self._initialize_output_layer()

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "leaky_relu": nn.LeakyReLU(0.1),
            "elu": nn.ELU(),
        }
        return activations.get(activation.lower(), nn.GELU())

    def _initialize_output_layer(self) -> None:
        """Initialize output layer for balanced initial predictions."""
        # Initialize with small weights
        nn.init.xavier_uniform_(self.output_layer.weight, gain=0.1)
        # Initialize bias to slight negative for sparse targets
        # This helps with class imbalance (most genes are not mutated)
        nn.init.constant_(self.output_layer.bias, -1.0)

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute multi-label classification logits.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Logits for each target [batch_size, n_targets]
        """
        # Pass through shared layers
        h = self.shared_layers(x)

        # Output layer
        logits = self.output_layer(h)

        return logits

    def predict_proba(self, x: Tensor) -> Tensor:
        """
        Compute prediction probabilities.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Probabilities for each target [batch_size, n_targets]
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def predict(self, x: Tensor, threshold: float = 0.5) -> Tensor:
        """
        Compute binary predictions.

        Args:
            x: Input features [batch_size, input_dim]
            threshold: Classification threshold.

        Returns:
            Binary predictions [batch_size, n_targets]
        """
        probs = self.predict_proba(x)
        return (probs >= threshold).float()

    @property
    def num_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"MultiLabelClassificationHead(\n"
            f"  input_dim={self.input_dim},\n"
            f"  n_targets={self.n_targets},\n"
            f"  hidden_dims={self.hidden_dims},\n"
            f"  parameters={self.num_parameters:,}\n"
            f")"
        )


class GroupedClassificationHead(nn.Module):
    """
    Grouped classification head with separate sub-networks for different target groups.

    Useful when different targets require different processing (e.g., genes vs biomarkers).

    Args:
        input_dim: Dimension of input features.
        target_groups: Dictionary mapping group names to number of targets.
            Example: {'genes': 40, 'tmb': 1, 'msi': 1, 'pdl1': 1}
        hidden_dim: Hidden layer dimension for each group.
        dropout: Dropout probability.

    Example:
        >>> groups = {'genes': 40, 'biomarkers': 3}
        >>> head = GroupedClassificationHead(input_dim=256, target_groups=groups)
        >>> features = torch.randn(32, 256)
        >>> outputs = head(features)
        >>> outputs['genes'].shape  # [32, 40]
        >>> outputs['biomarkers'].shape  # [32, 3]
    """

    def __init__(
        self,
        input_dim: int = 256,
        target_groups: dict[str, int] | None = None,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if target_groups is None:
            target_groups = {
                "genes": 40,
                "tmb": 1,
                "msi": 1,
                "pdl1": 1,
            }

        self.input_dim = input_dim
        self.target_groups = target_groups

        # Create separate heads for each group
        self.group_heads = nn.ModuleDict()
        for group_name, n_targets in target_groups.items():
            self.group_heads[group_name] = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_targets),
            )

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """
        Compute grouped classification logits.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Dictionary mapping group names to logits tensors.
        """
        outputs = {}
        for group_name, head in self.group_heads.items():
            outputs[group_name] = head(x)
        return outputs

    def forward_flat(self, x: Tensor) -> Tensor:
        """
        Compute flat logits (concatenated across all groups).

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Concatenated logits [batch_size, total_targets]
        """
        outputs = self.forward(x)
        return torch.cat(list(outputs.values()), dim=-1)

    @property
    def n_targets(self) -> int:
        """Total number of targets across all groups."""
        return sum(self.target_groups.values())

    @property
    def num_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
