"""
ARGUS Training Module.

This module contains components for training the ARGUS model:
- Trainer: Main training loop with support for various training strategies
- Callbacks: Training callbacks for logging, checkpointing, etc.
- Schedulers: Learning rate schedulers including cosine annealing with warmup
- Optimizers: Optimizer configurations
"""

from argus.training.trainer import ARGUSTrainer, TrainingConfig
from argus.training.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
    GradientClipping,
    MetricsLogger,
)
from argus.training.schedulers import (
    CosineAnnealingWarmup,
    LinearWarmup,
    PolynomialDecay,
    OneCycleLR,
)
from argus.training.optimizers import (
    create_optimizer,
    create_optimizer_groups,
)

__all__ = [
    # Trainer
    "ARGUSTrainer",
    "TrainingConfig",
    # Callbacks
    "EarlyStopping",
    "ModelCheckpoint",
    "LearningRateMonitor",
    "GradientClipping",
    "MetricsLogger",
    # Schedulers
    "CosineAnnealingWarmup",
    "LinearWarmup",
    "PolynomialDecay",
    "OneCycleLR",
    # Optimizers
    "create_optimizer",
    "create_optimizer_groups",
]
