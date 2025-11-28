"""
Focal Loss Implementation.

Focal loss for handling class imbalance in multi-label classification.

Copyright 2024-2025 VTP Consortium
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification.

    Implements focal loss from "Focal Loss for Dense Object Detection"
    (Lin et al., 2017). Focal loss down-weights well-classified examples
    to focus learning on hard negatives.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where:
        p_t = p if y=1, else 1-p
        alpha_t = alpha if y=1, else 1-alpha

    The focusing parameter gamma reduces the loss contribution from easy
    examples and extends the range in which an example receives low loss.

    Args:
        alpha: Weighting factor for positive class [0, 1].
            Default: 0.25 (as in original paper)
        gamma: Focusing parameter. gamma=0 is equivalent to BCE.
            Default: 2.0 (as in original paper)
        reduction: Reduction method ('mean', 'sum', 'none').
            Default: 'mean'
        eps: Small constant for numerical stability.
            Default: 1e-7

    Example:
        >>> loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        >>> logits = torch.randn(32, 43)
        >>> targets = torch.randint(0, 2, (32, 43)).float()
        >>> loss = loss_fn(logits, targets)
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
        eps: float = 1e-7,
    ) -> None:
        super().__init__()

        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0.0, 1.0]")
        if gamma < 0:
            raise ValueError("gamma must be non-negative")

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Compute focal loss.

        Args:
            logits: Predicted logits [batch_size, n_targets]
            targets: Binary targets [batch_size, n_targets]

        Returns:
            Focal loss value.
        """
        # Compute probabilities
        probs = torch.sigmoid(logits)
        probs = probs.clamp(min=self.eps, max=1 - self.eps)

        # Compute p_t
        p_t = targets * probs + (1 - targets) * (1 - probs)

        # Compute alpha_t
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Compute cross-entropy
        ce = -torch.log(p_t)

        # Focal loss
        loss = alpha_t * focal_weight * ce

        # Reduce
        if self.reduction == "none":
            return loss
        elif self.reduction == "sum":
            return loss.sum()
        else:  # mean
            return loss.mean()


class AsymmetricFocalLoss(nn.Module):
    """
    Asymmetric Focal Loss for multi-label classification.

    Extends focal loss with asymmetric focusing to better handle
    extreme class imbalance common in genomic predictions where
    most genes are not mutated.

    AFL = -y * (1 - p)^gamma_pos * log(p) - (1-y) * p^gamma_neg * log(1-p)

    Using different gamma values for positive and negative samples
    allows more aggressive down-weighting of easy negatives while
    maintaining sensitivity to positives.

    Args:
        gamma_pos: Focusing parameter for positive samples.
            Default: 1.0
        gamma_neg: Focusing parameter for negative samples.
            Default: 4.0 (higher to suppress easy negatives more)
        clip: Probability margin for negative samples.
            Default: 0.05 (shifts negative probabilities down)
        reduction: Reduction method.
            Default: 'mean'
        eps: Numerical stability constant.
            Default: 1e-7

    Example:
        >>> loss_fn = AsymmetricFocalLoss(gamma_pos=1, gamma_neg=4)
        >>> loss = loss_fn(logits, targets)
    """

    def __init__(
        self,
        gamma_pos: float = 1.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        reduction: str = "mean",
        eps: float = 1e-7,
    ) -> None:
        super().__init__()

        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Compute asymmetric focal loss.

        Args:
            logits: Predicted logits [batch_size, n_targets]
            targets: Binary targets [batch_size, n_targets]

        Returns:
            Asymmetric focal loss value.
        """
        # Compute probabilities
        probs = torch.sigmoid(logits)
        probs = probs.clamp(min=self.eps, max=1 - self.eps)

        # Probability shifting (hard thresholding for negatives)
        probs_neg = (probs - self.clip).clamp(min=self.eps)

        # Positive samples: standard focal
        loss_pos = targets * (1 - probs) ** self.gamma_pos * torch.log(probs)

        # Negative samples: asymmetric focal with probability shifting
        loss_neg = (1 - targets) * probs_neg ** self.gamma_neg * torch.log(1 - probs_neg)

        # Combined loss
        loss = -(loss_pos + loss_neg)

        # Reduce
        if self.reduction == "none":
            return loss
        elif self.reduction == "sum":
            return loss.sum()
        else:  # mean
            return loss.mean()


class ClassBalancedFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss.

    Combines focal loss with class-balanced weighting based on effective
    number of samples (Cui et al., CVPR 2019).

    CB_FL = (1 - beta) / (1 - beta^n_y) * FL(p_t)

    where n_y is the number of samples for class y.

    Args:
        samples_per_class: Number of samples per class [n_targets].
        beta: Hyperparameter for effective number computation.
            Default: 0.9999
        gamma: Focal loss focusing parameter.
            Default: 2.0
        reduction: Reduction method.
            Default: 'mean'

    Example:
        >>> samples_per_class = torch.tensor([1000, 500, 100, ...])
        >>> loss_fn = ClassBalancedFocalLoss(samples_per_class)
        >>> loss = loss_fn(logits, targets)
    """

    def __init__(
        self,
        samples_per_class: Tensor,
        beta: float = 0.9999,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()

        self.gamma = gamma
        self.reduction = reduction

        # Compute effective number weights
        effective_num = 1.0 - torch.pow(torch.tensor(beta), samples_per_class.float())
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * len(weights)

        self.register_buffer("class_weights", weights)

        # Initialize focal loss without class weighting
        self.focal_loss = FocalLoss(alpha=0.5, gamma=gamma, reduction="none")

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Compute class-balanced focal loss.

        Args:
            logits: Predicted logits [batch_size, n_targets]
            targets: Binary targets [batch_size, n_targets]

        Returns:
            Class-balanced focal loss value.
        """
        # Compute per-element focal loss
        focal = self.focal_loss(logits, targets)

        # Apply class weights
        weights = self.class_weights.unsqueeze(0).expand_as(focal)
        loss = focal * weights

        # Reduce
        if self.reduction == "none":
            return loss
        elif self.reduction == "sum":
            return loss.sum()
        else:  # mean
            return loss.mean()
