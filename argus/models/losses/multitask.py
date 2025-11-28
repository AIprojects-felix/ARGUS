"""
Multi-Task Loss Functions.

Combined loss functions for multi-task learning in ARGUS.

Copyright 2024-2025 VTP Consortium
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss for ARGUS genomic predictions.

    Combines losses for different prediction tasks:
    - Gene mutation classification (multi-label BCE/Focal)
    - TMB level prediction (ordinal classification)
    - MSI status prediction (binary classification)
    - PD-L1 expression prediction (binary classification)

    Supports multiple loss weighting strategies:
    - Static: Fixed weights per task
    - Uncertainty: Homoscedastic uncertainty weighting (Kendall et al., 2018)
    - Dynamic: GradNorm-style adaptive weighting

    Args:
        task_losses: Dictionary mapping task names to loss modules.
        task_weights: Static weights per task (used if weighting='static').
            Default: equal weights
        weighting: Loss weighting strategy ('static', 'uncertainty', 'dynamic').
            Default: 'uncertainty'
        reduction: Reduction method ('mean', 'sum').
            Default: 'mean'

    Example:
        >>> task_losses = {
        ...     'genes': FocalLoss(),
        ...     'tmb': nn.CrossEntropyLoss(),
        ...     'msi': nn.BCEWithLogitsLoss(),
        ... }
        >>> task_weights = {'genes': 1.0, 'tmb': 0.5, 'msi': 0.5}
        >>> loss_fn = MultiTaskLoss(task_losses, task_weights)
        >>>
        >>> outputs = {'genes': gene_logits, 'tmb': tmb_logits, 'msi': msi_logits}
        >>> targets = {'genes': gene_targets, 'tmb': tmb_targets, 'msi': msi_targets}
        >>> loss, task_losses = loss_fn(outputs, targets)
    """

    def __init__(
        self,
        task_losses: dict[str, nn.Module] | None = None,
        task_weights: dict[str, float] | None = None,
        weighting: str = "uncertainty",
        reduction: str = "mean",
    ) -> None:
        super().__init__()

        if task_losses is None:
            # Default task losses
            task_losses = {
                "genes": nn.BCEWithLogitsLoss(),
                "tmb": nn.CrossEntropyLoss(),
                "msi": nn.BCEWithLogitsLoss(),
                "pdl1": nn.BCEWithLogitsLoss(),
            }

        self.task_names = list(task_losses.keys())
        self.task_losses = nn.ModuleDict(task_losses)
        self.weighting = weighting
        self.reduction = reduction

        # Initialize weighting parameters
        if weighting == "static":
            if task_weights is None:
                task_weights = {name: 1.0 for name in self.task_names}
            self.register_buffer(
                "task_weights",
                torch.tensor([task_weights[name] for name in self.task_names])
            )
        elif weighting == "uncertainty":
            # Learnable log variance parameters for uncertainty weighting
            self.log_vars = nn.ParameterDict({
                name: nn.Parameter(torch.zeros(1))
                for name in self.task_names
            })
        elif weighting == "dynamic":
            # Learnable task weights for dynamic weighting
            self.task_weight_params = nn.ParameterDict({
                name: nn.Parameter(torch.ones(1))
                for name in self.task_names
            })
            # Store initial losses for relative weighting
            self.register_buffer(
                "initial_losses",
                torch.ones(len(self.task_names))
            )
            self._initialized = False
        else:
            raise ValueError(f"Unknown weighting strategy: {weighting}")

    def forward(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor],
        return_task_losses: bool = True,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """
        Compute multi-task loss.

        Args:
            outputs: Dictionary of model outputs per task.
            targets: Dictionary of targets per task.
            return_task_losses: Whether to return individual task losses.

        Returns:
            Total loss (scalar).
            Optionally returns dictionary of per-task losses.
        """
        task_losses_dict = {}
        weighted_losses = []

        for i, task_name in enumerate(self.task_names):
            if task_name not in outputs or task_name not in targets:
                continue

            # Compute task loss
            task_output = outputs[task_name]
            task_target = targets[task_name]
            task_loss = self.task_losses[task_name](task_output, task_target)
            task_losses_dict[task_name] = task_loss

            # Apply weighting
            if self.weighting == "static":
                weighted_loss = self.task_weights[i] * task_loss
            elif self.weighting == "uncertainty":
                # Homoscedastic uncertainty weighting
                # L = sum(1/(2*sigma^2) * L_i + log(sigma))
                # = sum(exp(-log_var) * L_i + log_var / 2)
                log_var = self.log_vars[task_name]
                precision = torch.exp(-log_var)
                weighted_loss = precision * task_loss + log_var / 2
            else:  # dynamic
                weight = torch.relu(self.task_weight_params[task_name])
                weighted_loss = weight * task_loss

            weighted_losses.append(weighted_loss)

        # Combine losses
        total_loss = sum(weighted_losses)

        if self.reduction == "mean":
            total_loss = total_loss / len(weighted_losses)

        if return_task_losses:
            return total_loss, task_losses_dict
        return total_loss

    def get_task_weights(self) -> dict[str, float]:
        """
        Get current task weights.

        Returns:
            Dictionary of task weights.
        """
        weights = {}

        for i, task_name in enumerate(self.task_names):
            if self.weighting == "static":
                weights[task_name] = self.task_weights[i].item()
            elif self.weighting == "uncertainty":
                # Weight = 1 / (2 * sigma^2) = exp(-log_var) / 2
                log_var = self.log_vars[task_name].item()
                weights[task_name] = 0.5 * torch.exp(torch.tensor(-log_var)).item()
            else:  # dynamic
                weights[task_name] = torch.relu(
                    self.task_weight_params[task_name]
                ).item()

        return weights


class GradNormLoss(nn.Module):
    """
    GradNorm: Gradient Normalization for Adaptive Loss Balancing.

    Implements the GradNorm algorithm from "GradNorm: Gradient Normalization
    for Adaptive Loss Balancing in Deep Multitask Networks" (Chen et al., 2018).

    Dynamically adjusts task weights to balance gradient magnitudes across
    tasks, ensuring no single task dominates learning.

    Args:
        task_losses: Dictionary mapping task names to loss modules.
        alpha: Asymmetry hyperparameter controlling weight update rate.
            Default: 1.5
        shared_parameters: Shared model parameters for gradient computation.
            Required for gradient-based weighting.

    Example:
        >>> loss_fn = GradNormLoss(task_losses, alpha=1.5)
        >>> loss_fn.set_shared_parameters(model.shared_encoder.parameters())
        >>> loss, task_losses = loss_fn(outputs, targets)
    """

    def __init__(
        self,
        task_losses: dict[str, nn.Module],
        alpha: float = 1.5,
    ) -> None:
        super().__init__()

        self.task_names = list(task_losses.keys())
        self.task_losses = nn.ModuleDict(task_losses)
        self.alpha = alpha

        # Learnable task weights
        self.task_weights = nn.ParameterDict({
            name: nn.Parameter(torch.ones(1))
            for name in self.task_names
        })

        # Track initial losses for relative weighting
        self.register_buffer(
            "initial_losses",
            torch.ones(len(self.task_names))
        )

        self._shared_parameters: list[nn.Parameter] | None = None
        self._initialized = False

    def set_shared_parameters(self, parameters: Any) -> None:
        """Set shared parameters for gradient computation."""
        self._shared_parameters = list(parameters)

    def forward(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor],
        update_weights: bool = True,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Compute GradNorm loss.

        Args:
            outputs: Dictionary of model outputs per task.
            targets: Dictionary of targets per task.
            update_weights: Whether to update task weights this step.

        Returns:
            Total weighted loss and dictionary of per-task losses.
        """
        task_losses_dict = {}
        raw_losses = []

        for task_name in self.task_names:
            if task_name not in outputs or task_name not in targets:
                continue

            task_loss = self.task_losses[task_name](
                outputs[task_name],
                targets[task_name]
            )
            task_losses_dict[task_name] = task_loss
            raw_losses.append(task_loss)

        # Initialize reference losses
        if not self._initialized and len(raw_losses) == len(self.task_names):
            with torch.no_grad():
                self.initial_losses.copy_(torch.stack(raw_losses))
            self._initialized = True

        # Compute weighted loss
        weighted_loss = sum(
            torch.relu(self.task_weights[name]) * task_losses_dict[name]
            for name in task_losses_dict
        )

        return weighted_loss, task_losses_dict

    def compute_grad_norm_loss(
        self,
        task_losses: dict[str, Tensor],
    ) -> Tensor:
        """
        Compute gradient normalization loss for weight updates.

        This should be called separately to update task weights.

        Args:
            task_losses: Dictionary of per-task losses.

        Returns:
            GradNorm loss for weight optimization.
        """
        if self._shared_parameters is None:
            raise RuntimeError("Shared parameters not set. Call set_shared_parameters first.")

        grad_norms = []
        loss_ratios = []

        for i, task_name in enumerate(self.task_names):
            if task_name not in task_losses:
                continue

            task_loss = task_losses[task_name]
            weight = torch.relu(self.task_weights[task_name])

            # Compute gradient of weighted loss w.r.t. shared parameters
            weighted_loss = weight * task_loss
            grads = torch.autograd.grad(
                weighted_loss,
                self._shared_parameters,
                retain_graph=True,
                allow_unused=True,
            )

            # Compute gradient norm
            grad_norm = sum(
                g.norm() for g in grads if g is not None
            )
            grad_norms.append(grad_norm)

            # Compute loss ratio (inverse training rate)
            loss_ratio = task_loss / (self.initial_losses[i] + 1e-8)
            loss_ratios.append(loss_ratio)

        # Average gradient norm
        avg_grad_norm = sum(grad_norms) / len(grad_norms)

        # Compute relative inverse training rates
        loss_ratios = torch.stack(loss_ratios)
        avg_loss_ratio = loss_ratios.mean()
        relative_rates = loss_ratios / avg_loss_ratio

        # Target gradient norms
        target_grad_norms = avg_grad_norm * (relative_rates ** self.alpha)

        # GradNorm loss
        grad_norm_loss = sum(
            torch.abs(gn - tgn)
            for gn, tgn in zip(grad_norms, target_grad_norms)
        )

        return grad_norm_loss


class UncertaintyWeightedLoss(nn.Module):
    """
    Uncertainty-weighted multi-task loss.

    Simplified uncertainty weighting without the full GradNorm complexity.
    Learns task-specific uncertainty (log variance) to automatically
    balance task contributions.

    Loss = sum(exp(-s_i) * L_i + s_i)

    where s_i = log(sigma_i^2) is the learned log variance.

    Args:
        task_losses: Dictionary mapping task names to loss modules.
        initial_log_vars: Initial log variance values per task.
            Default: 0.0 for all tasks

    Example:
        >>> loss_fn = UncertaintyWeightedLoss(task_losses)
        >>> loss, task_losses = loss_fn(outputs, targets)
    """

    def __init__(
        self,
        task_losses: dict[str, nn.Module],
        initial_log_vars: dict[str, float] | None = None,
    ) -> None:
        super().__init__()

        self.task_names = list(task_losses.keys())
        self.task_losses = nn.ModuleDict(task_losses)

        if initial_log_vars is None:
            initial_log_vars = {name: 0.0 for name in self.task_names}

        # Learnable log variance parameters
        self.log_vars = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(initial_log_vars.get(name, 0.0)))
            for name in self.task_names
        })

    def forward(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Compute uncertainty-weighted loss.

        Args:
            outputs: Dictionary of model outputs per task.
            targets: Dictionary of targets per task.

        Returns:
            Total weighted loss and dictionary of per-task losses.
        """
        task_losses_dict = {}
        total_loss = torch.tensor(0.0, device=next(iter(outputs.values())).device)

        for task_name in self.task_names:
            if task_name not in outputs or task_name not in targets:
                continue

            # Compute task loss
            task_loss = self.task_losses[task_name](
                outputs[task_name],
                targets[task_name]
            )
            task_losses_dict[task_name] = task_loss

            # Apply uncertainty weighting
            log_var = self.log_vars[task_name]
            precision = torch.exp(-log_var)
            total_loss = total_loss + precision * task_loss + log_var

        return total_loss, task_losses_dict

    def get_uncertainties(self) -> dict[str, float]:
        """
        Get current uncertainty (standard deviation) for each task.

        Returns:
            Dictionary of task uncertainties.
        """
        return {
            name: torch.exp(0.5 * self.log_vars[name]).item()
            for name in self.task_names
        }
