"""
Learning Rate Schedulers.

Custom learning rate schedulers for ARGUS training.

Copyright 2024-2025 VTP Consortium
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import math
from typing import List

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmup(_LRScheduler):
    """
    Cosine annealing scheduler with linear warmup.

    This is the primary scheduler used in ARGUS training, following the
    approach from "Attention is All You Need" and later transformers.

    Learning rate schedule:
        - Warmup phase: Linear increase from 0 to base_lr
        - Annealing phase: Cosine decay from base_lr to min_lr

    LR = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * progress))

    where progress = (step - warmup_steps) / (total_steps - warmup_steps)

    Args:
        optimizer: Wrapped optimizer.
        warmup_steps: Number of warmup steps.
        total_steps: Total number of training steps.
        min_lr: Minimum learning rate after decay.
            Default: 1e-7
        last_epoch: The index of last epoch.
            Default: -1

    Example:
        >>> scheduler = CosineAnnealingWarmup(
        ...     optimizer=optimizer,
        ...     warmup_steps=1000,
        ...     total_steps=100000,
        ...     min_lr=1e-6
        ... )
        >>> for step in range(100000):
        ...     optimizer.step()
        ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-7,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Calculate learning rate for current step."""
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = self.last_epoch / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        # Cosine annealing
        progress = (self.last_epoch - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        progress = min(1.0, progress)

        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))

        return [
            self.min_lr + (base_lr - self.min_lr) * cosine_factor
            for base_lr in self.base_lrs
        ]


class LinearWarmup(_LRScheduler):
    """
    Linear warmup followed by constant learning rate.

    Args:
        optimizer: Wrapped optimizer.
        warmup_steps: Number of warmup steps.
        last_epoch: The index of last epoch.
            Default: -1

    Example:
        >>> scheduler = LinearWarmup(optimizer, warmup_steps=1000)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Calculate learning rate for current step."""
        if self.last_epoch < self.warmup_steps:
            warmup_factor = self.last_epoch / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        return self.base_lrs


class PolynomialDecay(_LRScheduler):
    """
    Polynomial learning rate decay with optional warmup.

    LR = (base_lr - end_lr) * (1 - progress)^power + end_lr

    Args:
        optimizer: Wrapped optimizer.
        total_steps: Total number of training steps.
        warmup_steps: Number of warmup steps.
            Default: 0
        end_lr: Final learning rate.
            Default: 0.0
        power: Polynomial power.
            Default: 1.0 (linear decay)
        last_epoch: The index of last epoch.
            Default: -1

    Example:
        >>> scheduler = PolynomialDecay(
        ...     optimizer=optimizer,
        ...     total_steps=100000,
        ...     warmup_steps=1000,
        ...     power=2.0
        ... )
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        end_lr: float = 0.0,
        power: float = 1.0,
        last_epoch: int = -1,
    ) -> None:
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.end_lr = end_lr
        self.power = power

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Calculate learning rate for current step."""
        if self.last_epoch < self.warmup_steps:
            warmup_factor = self.last_epoch / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        progress = (self.last_epoch - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        progress = min(1.0, progress)

        decay_factor = (1.0 - progress) ** self.power

        return [
            (base_lr - self.end_lr) * decay_factor + self.end_lr
            for base_lr in self.base_lrs
        ]


class OneCycleLR(_LRScheduler):
    """
    1Cycle learning rate policy.

    Implements the 1cycle learning rate policy from "Super-Convergence"
    (Smith & Topin, 2019). Learning rate increases from div_factor * max_lr
    to max_lr in first phase, then decreases to final_div_factor * max_lr.

    Args:
        optimizer: Wrapped optimizer.
        max_lr: Maximum learning rate.
        total_steps: Total number of training steps.
        pct_start: Percentage of cycle spent increasing learning rate.
            Default: 0.3
        div_factor: Initial learning rate = max_lr / div_factor.
            Default: 25.0
        final_div_factor: Final learning rate = max_lr / final_div_factor.
            Default: 10000.0
        anneal_strategy: Annealing strategy ('cos' or 'linear').
            Default: 'cos'
        last_epoch: The index of last epoch.
            Default: -1

    Example:
        >>> scheduler = OneCycleLR(
        ...     optimizer=optimizer,
        ...     max_lr=1e-3,
        ...     total_steps=100000
        ... )
    """

    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3,
        div_factor: float = 25.0,
        final_div_factor: float = 10000.0,
        anneal_strategy: str = "cos",
        last_epoch: int = -1,
    ) -> None:
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.anneal_strategy = anneal_strategy

        self.initial_lr = max_lr / div_factor
        self.final_lr = max_lr / final_div_factor
        self.step_up = int(total_steps * pct_start)
        self.step_down = total_steps - self.step_up

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Calculate learning rate for current step."""
        if self.last_epoch < self.step_up:
            # Increasing phase
            progress = self.last_epoch / max(1, self.step_up)
            lr = self._anneal(self.initial_lr, self.max_lr, progress)
        else:
            # Decreasing phase
            progress = (self.last_epoch - self.step_up) / max(1, self.step_down)
            lr = self._anneal(self.max_lr, self.final_lr, progress)

        return [lr for _ in self.base_lrs]

    def _anneal(self, start: float, end: float, progress: float) -> float:
        """Apply annealing function."""
        if self.anneal_strategy == "cos":
            return end + (start - end) * (1 + math.cos(math.pi * progress)) / 2
        else:  # linear
            return start + (end - start) * progress


class WarmupThenDecay(_LRScheduler):
    """
    Warmup followed by specified decay schedule.

    Combines linear warmup with various decay strategies.

    Args:
        optimizer: Wrapped optimizer.
        warmup_steps: Number of warmup steps.
        decay_steps: Number of decay steps after warmup.
        decay_type: Type of decay ('cosine', 'linear', 'exponential', 'step').
            Default: 'cosine'
        min_lr: Minimum learning rate.
            Default: 1e-7
        decay_rate: Decay rate for exponential/step decay.
            Default: 0.1
        step_size: Step size for step decay.
            Default: 1000
        last_epoch: The index of last epoch.
            Default: -1

    Example:
        >>> scheduler = WarmupThenDecay(
        ...     optimizer=optimizer,
        ...     warmup_steps=1000,
        ...     decay_steps=9000,
        ...     decay_type='cosine'
        ... )
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        decay_steps: int,
        decay_type: str = "cosine",
        min_lr: float = 1e-7,
        decay_rate: float = 0.1,
        step_size: int = 1000,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.decay_type = decay_type
        self.min_lr = min_lr
        self.decay_rate = decay_rate
        self.step_size = step_size

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Calculate learning rate for current step."""
        # Warmup phase
        if self.last_epoch < self.warmup_steps:
            warmup_factor = self.last_epoch / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        # Decay phase
        decay_progress = (self.last_epoch - self.warmup_steps) / max(1, self.decay_steps)
        decay_progress = min(1.0, decay_progress)

        lrs = []
        for base_lr in self.base_lrs:
            if self.decay_type == "cosine":
                factor = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
                lr = self.min_lr + (base_lr - self.min_lr) * factor

            elif self.decay_type == "linear":
                lr = base_lr * (1.0 - decay_progress) + self.min_lr * decay_progress

            elif self.decay_type == "exponential":
                lr = base_lr * (self.decay_rate ** decay_progress)
                lr = max(lr, self.min_lr)

            elif self.decay_type == "step":
                n_steps = (self.last_epoch - self.warmup_steps) // self.step_size
                lr = base_lr * (self.decay_rate ** n_steps)
                lr = max(lr, self.min_lr)

            else:
                raise ValueError(f"Unknown decay type: {self.decay_type}")

            lrs.append(lr)

        return lrs


class NoamScheduler(_LRScheduler):
    """
    Noam learning rate scheduler from "Attention is All You Need".

    LR = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))

    This scheduler increases the learning rate linearly for the first
    warmup_steps, then decreases it proportionally to the inverse
    square root of the step number.

    Args:
        optimizer: Wrapped optimizer.
        d_model: Model dimension (used for scaling).
        warmup_steps: Number of warmup steps.
        scale: Learning rate scale factor.
            Default: 1.0
        last_epoch: The index of last epoch.
            Default: -1

    Example:
        >>> scheduler = NoamScheduler(
        ...     optimizer=optimizer,
        ...     d_model=256,
        ...     warmup_steps=4000
        ... )
    """

    def __init__(
        self,
        optimizer: Optimizer,
        d_model: int,
        warmup_steps: int,
        scale: float = 1.0,
        last_epoch: int = -1,
    ) -> None:
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.scale = scale

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Calculate learning rate for current step."""
        step = max(1, self.last_epoch)
        warmup = self.warmup_steps

        lr = self.scale * (self.d_model ** -0.5) * min(
            step ** -0.5,
            step * warmup ** -1.5
        )

        return [lr for _ in self.base_lrs]


def get_scheduler(
    name: str,
    optimizer: Optimizer,
    **kwargs,
) -> _LRScheduler:
    """
    Factory function to create a learning rate scheduler.

    Args:
        name: Scheduler name.
        optimizer: Wrapped optimizer.
        **kwargs: Scheduler-specific arguments.

    Returns:
        Learning rate scheduler.

    Example:
        >>> scheduler = get_scheduler(
        ...     'cosine_warmup',
        ...     optimizer,
        ...     warmup_steps=1000,
        ...     total_steps=100000
        ... )
    """
    schedulers = {
        "cosine_warmup": CosineAnnealingWarmup,
        "linear_warmup": LinearWarmup,
        "polynomial": PolynomialDecay,
        "onecycle": OneCycleLR,
        "warmup_decay": WarmupThenDecay,
        "noam": NoamScheduler,
    }

    if name not in schedulers:
        raise ValueError(
            f"Unknown scheduler: {name}. Available: {list(schedulers.keys())}"
        )

    return schedulers[name](optimizer, **kwargs)
