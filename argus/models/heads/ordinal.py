"""
Ordinal Classification Head.

Prediction head for ordinal regression tasks like TMB level prediction.

Copyright 2024-2025 VTP Consortium
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class OrdinalClassificationHead(nn.Module):
    """
    Ordinal classification head for ordered categorical predictions.

    Implements ordinal regression for tasks where classes have a natural ordering
    (e.g., TMB levels: Low < Medium < High). Uses cumulative threshold approach
    to ensure consistent ordinal predictions.

    Theory:
        For K ordinal classes, predict K-1 cumulative probabilities:
        P(Y > k) for k = 1, ..., K-1

        Final class probabilities:
        P(Y = 1) = 1 - P(Y > 1)
        P(Y = k) = P(Y > k-1) - P(Y > k) for k = 2, ..., K-1
        P(Y = K) = P(Y > K-1)

    This ensures:
        - Ordinal consistency: P(Y > k) >= P(Y > k+1)
        - Proper probability distribution

    Args:
        input_dim: Dimension of input features.
            Default: 256
        n_classes: Number of ordinal classes.
            Default: 3 (Low, Medium, High for TMB)
        hidden_dims: List of hidden layer dimensions.
            Default: [256, 128]
        dropout: Dropout probability.
            Default: 0.1
        activation: Activation function.
            Default: 'gelu'

    Example:
        >>> head = OrdinalClassificationHead(input_dim=256, n_classes=3)
        >>> features = torch.randn(32, 256)
        >>> logits = head(features)  # [32, 3]
        >>> probs = head.predict_proba(features)  # [32, 3], sums to 1
    """

    def __init__(
        self,
        input_dim: int = 256,
        n_classes: int = 3,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128]

        if n_classes < 2:
            raise ValueError("n_classes must be at least 2 for ordinal classification")

        self.input_dim = input_dim
        self.n_classes = n_classes
        self.n_thresholds = n_classes - 1  # K-1 thresholds for K classes

        # Get activation function
        activation_fn = self._get_activation(activation)

        # Feature extraction network
        layers: list[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                activation_fn,
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        # Shared feature projection
        self.shared_projection = nn.Linear(prev_dim, 1)

        # Learnable thresholds (ordered)
        # Initialize with evenly spaced values
        initial_thresholds = torch.linspace(-2, 2, self.n_thresholds)
        self.thresholds = nn.Parameter(initial_thresholds)

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "leaky_relu": nn.LeakyReLU(0.1),
        }
        return activations.get(activation.lower(), nn.GELU())

    def _enforce_threshold_ordering(self) -> Tensor:
        """
        Ensure thresholds are ordered (ascending).

        Returns:
            Ordered thresholds tensor.
        """
        # Use cumulative softplus to ensure ordering
        # threshold[i] = threshold[0] + sum(softplus(diff[j]) for j < i)
        diffs = F.softplus(self.thresholds[1:] - self.thresholds[:-1])
        ordered = torch.cat([
            self.thresholds[:1],
            self.thresholds[:1] + torch.cumsum(diffs, dim=0)
        ])
        return ordered

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute ordinal classification logits.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Logits for each class [batch_size, n_classes]
        """
        # Extract features
        h = self.feature_extractor(x)

        # Project to single value
        latent = self.shared_projection(h)  # [batch_size, 1]

        # Get ordered thresholds
        thresholds = self._enforce_threshold_ordering()

        # Compute cumulative logits: latent - threshold
        # Shape: [batch_size, n_thresholds]
        cumulative_logits = latent - thresholds.unsqueeze(0)

        # Convert cumulative logits to class logits
        # P(Y > k) = sigmoid(cumulative_logits[k])
        cumulative_probs = torch.sigmoid(cumulative_logits)

        # Compute class probabilities
        # P(Y = 1) = 1 - P(Y > 1)
        # P(Y = k) = P(Y > k-1) - P(Y > k)
        # P(Y = K) = P(Y > K-1)
        probs = self._cumulative_to_class_probs(cumulative_probs)

        # Return logits (log-odds)
        # Clamp for numerical stability
        probs = probs.clamp(min=1e-7, max=1 - 1e-7)
        logits = torch.log(probs / (1 - probs + 1e-7))

        return logits

    def _cumulative_to_class_probs(self, cumulative_probs: Tensor) -> Tensor:
        """
        Convert cumulative probabilities to class probabilities.

        Args:
            cumulative_probs: P(Y > k) for k = 1, ..., K-1
                Shape: [batch_size, n_thresholds]

        Returns:
            Class probabilities [batch_size, n_classes]
        """
        batch_size = cumulative_probs.shape[0]

        # P(Y = 1) = 1 - P(Y > 1)
        p_first = 1 - cumulative_probs[:, 0:1]

        # P(Y = k) = P(Y > k-1) - P(Y > k) for middle classes
        if self.n_classes > 2:
            p_middle = cumulative_probs[:, :-1] - cumulative_probs[:, 1:]
        else:
            p_middle = torch.empty(batch_size, 0, device=cumulative_probs.device)

        # P(Y = K) = P(Y > K-1)
        p_last = cumulative_probs[:, -1:]

        # Concatenate
        probs = torch.cat([p_first, p_middle, p_last], dim=-1)

        return probs

    def predict_proba(self, x: Tensor) -> Tensor:
        """
        Compute class probabilities.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Class probabilities [batch_size, n_classes], sums to 1.
        """
        h = self.feature_extractor(x)
        latent = self.shared_projection(h)
        thresholds = self._enforce_threshold_ordering()
        cumulative_logits = latent - thresholds.unsqueeze(0)
        cumulative_probs = torch.sigmoid(cumulative_logits)
        probs = self._cumulative_to_class_probs(cumulative_probs)

        # Ensure valid probability distribution
        probs = probs.clamp(min=1e-7)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        return probs

    def predict(self, x: Tensor) -> Tensor:
        """
        Predict ordinal class labels.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Predicted class labels [batch_size] (0 to n_classes-1)
        """
        probs = self.predict_proba(x)
        return probs.argmax(dim=-1)

    def predict_expected(self, x: Tensor) -> Tensor:
        """
        Predict expected ordinal value (useful for regression-like output).

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Expected ordinal value [batch_size]
        """
        probs = self.predict_proba(x)
        class_values = torch.arange(
            self.n_classes, dtype=probs.dtype, device=probs.device
        )
        return (probs * class_values).sum(dim=-1)

    @property
    def num_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"OrdinalClassificationHead(\n"
            f"  input_dim={self.input_dim},\n"
            f"  n_classes={self.n_classes},\n"
            f"  parameters={self.num_parameters:,}\n"
            f")"
        )


class MultiTaskOrdinalHead(nn.Module):
    """
    Multi-task ordinal classification head.

    Combines multiple ordinal classification tasks with shared feature extraction
    and task-specific output layers.

    Args:
        input_dim: Dimension of input features.
        task_configs: Dictionary mapping task names to number of ordinal classes.
            Example: {'tmb': 3, 'stage': 4}
        shared_hidden_dim: Hidden dimension for shared layers.
        task_hidden_dim: Hidden dimension for task-specific layers.
        dropout: Dropout probability.

    Example:
        >>> configs = {'tmb': 3, 'stage': 4}
        >>> head = MultiTaskOrdinalHead(input_dim=256, task_configs=configs)
        >>> features = torch.randn(32, 256)
        >>> outputs = head(features)
        >>> outputs['tmb'].shape  # [32, 3]
        >>> outputs['stage'].shape  # [32, 4]
    """

    def __init__(
        self,
        input_dim: int = 256,
        task_configs: dict[str, int] | None = None,
        shared_hidden_dim: int = 256,
        task_hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if task_configs is None:
            task_configs = {"tmb": 3}

        self.input_dim = input_dim
        self.task_configs = task_configs

        # Shared feature extraction
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, shared_hidden_dim),
            nn.LayerNorm(shared_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Task-specific ordinal heads
        self.task_heads = nn.ModuleDict()
        for task_name, n_classes in task_configs.items():
            self.task_heads[task_name] = OrdinalClassificationHead(
                input_dim=shared_hidden_dim,
                n_classes=n_classes,
                hidden_dims=[task_hidden_dim],
                dropout=dropout,
            )

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """
        Compute ordinal classification logits for all tasks.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Dictionary mapping task names to logits tensors.
        """
        # Shared feature extraction
        h = self.shared_layers(x)

        # Task-specific outputs
        outputs = {}
        for task_name, head in self.task_heads.items():
            outputs[task_name] = head(h)

        return outputs

    @property
    def num_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
