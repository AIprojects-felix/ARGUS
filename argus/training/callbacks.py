"""
Training Callbacks.

Callback classes for extending training functionality.

Copyright 2024-2025 VTP Consortium
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import logging
from abc import ABC
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from argus.training.trainer import ARGUSTrainer

logger = logging.getLogger(__name__)


class Callback(ABC):
    """
    Base class for training callbacks.

    Callbacks allow extending the training loop with custom behavior
    at various points during training.

    Available hooks:
        - on_train_start: Called at the beginning of training
        - on_train_end: Called at the end of training
        - on_epoch_start: Called at the start of each epoch
        - on_epoch_end: Called at the end of each epoch
        - on_batch_start: Called before each batch
        - on_batch_end: Called after each batch
        - on_validation_start: Called before validation
        - on_validation_end: Called after validation
    """

    def on_train_start(self, trainer: "ARGUSTrainer") -> None:
        """Called at the beginning of training."""
        pass

    def on_train_end(self, trainer: "ARGUSTrainer") -> None:
        """Called at the end of training."""
        pass

    def on_epoch_start(self, trainer: "ARGUSTrainer", epoch: int) -> None:
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(
        self,
        trainer: "ARGUSTrainer",
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        """Called at the end of each epoch."""
        pass

    def on_batch_start(self, trainer: "ARGUSTrainer", batch_idx: int) -> None:
        """Called before each batch."""
        pass

    def on_batch_end(
        self,
        trainer: "ARGUSTrainer",
        batch_idx: int,
        loss: float,
    ) -> None:
        """Called after each batch."""
        pass

    def on_validation_start(self, trainer: "ARGUSTrainer") -> None:
        """Called before validation."""
        pass

    def on_validation_end(
        self,
        trainer: "ARGUSTrainer",
        metrics: dict[str, float],
    ) -> None:
        """Called after validation."""
        pass


class EarlyStopping(Callback):
    """
    Early stopping callback to stop training when a metric stops improving.

    Args:
        monitor: Metric to monitor.
            Default: 'val_loss'
        mode: 'min' or 'max' - whether to minimize or maximize the metric.
            Default: 'min'
        patience: Number of epochs with no improvement after which training will stop.
            Default: 10
        min_delta: Minimum change to qualify as an improvement.
            Default: 0.0
        verbose: Whether to print messages.
            Default: True

    Example:
        >>> callback = EarlyStopping(
        ...     monitor='val_auroc_mean',
        ...     mode='max',
        ...     patience=10
        ... )
        >>> trainer = ARGUSTrainer(..., callbacks=[callback])
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        mode: str = "min",
        patience: int = 10,
        min_delta: float = 0.0,
        verbose: bool = True,
    ) -> None:
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose

        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.best_epoch = 0

    def on_validation_end(
        self,
        trainer: "ARGUSTrainer",
        metrics: dict[str, float],
    ) -> None:
        """Check if training should stop."""
        if self.monitor not in metrics:
            return

        current = metrics[self.monitor]

        if self.mode == "min":
            is_improvement = current < (self.best_value - self.min_delta)
        else:
            is_improvement = current > (self.best_value + self.min_delta)

        if is_improvement:
            self.best_value = current
            self.counter = 0
            self.best_epoch = trainer.current_epoch
            if self.verbose:
                logger.info(
                    f"EarlyStopping: {self.monitor} improved to {current:.4f}"
                )
        else:
            self.counter += 1
            if self.verbose:
                logger.info(
                    f"EarlyStopping: {self.monitor} did not improve. "
                    f"Counter: {self.counter}/{self.patience}"
                )

            if self.counter >= self.patience:
                trainer.should_stop = True
                if self.verbose:
                    logger.info(
                        f"EarlyStopping: Stopping training. "
                        f"Best {self.monitor}: {self.best_value:.4f} at epoch {self.best_epoch}"
                    )


class ModelCheckpoint(Callback):
    """
    Save model checkpoints during training.

    Args:
        dirpath: Directory to save checkpoints.
        filename: Checkpoint filename pattern.
            Default: 'checkpoint-{epoch:02d}-{val_loss:.4f}'
        monitor: Metric to monitor for best checkpoint.
            Default: 'val_loss'
        mode: 'min' or 'max'.
            Default: 'min'
        save_top_k: Number of best checkpoints to keep.
            Default: 3
        save_last: Whether to save the last checkpoint.
            Default: True
        verbose: Whether to print messages.
            Default: True

    Example:
        >>> callback = ModelCheckpoint(
        ...     dirpath='checkpoints',
        ...     monitor='val_auroc_mean',
        ...     mode='max',
        ...     save_top_k=3
        ... )
    """

    def __init__(
        self,
        dirpath: str = "checkpoints",
        filename: str = "checkpoint-{epoch:02d}-{val_loss:.4f}",
        monitor: str = "val_loss",
        mode: str = "min",
        save_top_k: int = 3,
        save_last: bool = True,
        verbose: bool = True,
    ) -> None:
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.save_last = save_last
        self.verbose = verbose

        self.best_k_models: list[tuple[float, Path]] = []

    def on_train_start(self, trainer: "ARGUSTrainer") -> None:
        """Create checkpoint directory."""
        self.dirpath.mkdir(parents=True, exist_ok=True)

    def on_validation_end(
        self,
        trainer: "ARGUSTrainer",
        metrics: dict[str, float],
    ) -> None:
        """Save checkpoint if it's among the best."""
        if self.monitor not in metrics:
            return

        current = metrics[self.monitor]

        # Format filename
        format_dict = {"epoch": trainer.current_epoch}
        format_dict.update(metrics)

        try:
            filename = self.filename.format(**format_dict)
        except KeyError:
            filename = f"checkpoint-{trainer.current_epoch:02d}.pt"

        filepath = self.dirpath / filename

        # Check if this model should be saved
        if len(self.best_k_models) < self.save_top_k:
            self._save_checkpoint(trainer, filepath)
            self.best_k_models.append((current, filepath))
            self._sort_best_k()
        else:
            # Compare with worst of best_k
            worst_value, worst_path = self.best_k_models[-1]

            if self.mode == "min":
                is_better = current < worst_value
            else:
                is_better = current > worst_value

            if is_better:
                # Remove worst checkpoint
                if worst_path.exists():
                    worst_path.unlink()
                self.best_k_models.pop()

                # Save new checkpoint
                self._save_checkpoint(trainer, filepath)
                self.best_k_models.append((current, filepath))
                self._sort_best_k()

        # Save last checkpoint
        if self.save_last:
            last_path = self.dirpath / "last.pt"
            self._save_checkpoint(trainer, last_path)

    def _sort_best_k(self) -> None:
        """Sort best_k_models by metric value."""
        reverse = self.mode == "max"
        self.best_k_models.sort(key=lambda x: x[0], reverse=reverse)

    def _save_checkpoint(
        self,
        trainer: "ARGUSTrainer",
        filepath: Path,
    ) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scheduler_state_dict": trainer.scheduler.state_dict(),
        }

        torch.save(checkpoint, filepath)

        if self.verbose:
            logger.info(f"Saved checkpoint: {filepath}")


class LearningRateMonitor(Callback):
    """
    Log learning rate during training.

    Args:
        log_every_n_steps: How often to log the learning rate.
            Default: 100

    Example:
        >>> callback = LearningRateMonitor(log_every_n_steps=50)
    """

    def __init__(self, log_every_n_steps: int = 100) -> None:
        self.log_every_n_steps = log_every_n_steps
        self.lr_history: list[float] = []

    def on_batch_end(
        self,
        trainer: "ARGUSTrainer",
        batch_idx: int,
        loss: float,
    ) -> None:
        """Log current learning rate."""
        if trainer.global_step % self.log_every_n_steps == 0:
            lr = trainer.scheduler.get_last_lr()[0]
            self.lr_history.append(lr)
            logger.debug(f"Step {trainer.global_step}: LR = {lr:.2e}")

    def on_train_end(self, trainer: "ARGUSTrainer") -> None:
        """Log learning rate summary."""
        if self.lr_history:
            logger.info(
                f"Learning rate range: {min(self.lr_history):.2e} - {max(self.lr_history):.2e}"
            )


class GradientClipping(Callback):
    """
    Apply gradient clipping during training.

    Note: This is typically handled by the trainer itself,
    but this callback provides additional gradient monitoring.

    Args:
        max_norm: Maximum gradient norm.
            Default: 1.0
        log_gradient_norm: Whether to log gradient norms.
            Default: False
        log_every_n_steps: How often to log gradient norms.
            Default: 100

    Example:
        >>> callback = GradientClipping(max_norm=1.0, log_gradient_norm=True)
    """

    def __init__(
        self,
        max_norm: float = 1.0,
        log_gradient_norm: bool = False,
        log_every_n_steps: int = 100,
    ) -> None:
        self.max_norm = max_norm
        self.log_gradient_norm = log_gradient_norm
        self.log_every_n_steps = log_every_n_steps
        self.gradient_norms: list[float] = []

    def on_batch_end(
        self,
        trainer: "ARGUSTrainer",
        batch_idx: int,
        loss: float,
    ) -> None:
        """Monitor gradient norms."""
        if not self.log_gradient_norm:
            return

        if trainer.global_step % self.log_every_n_steps == 0:
            total_norm = 0.0
            for p in trainer.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            self.gradient_norms.append(total_norm)
            logger.debug(f"Step {trainer.global_step}: Gradient norm = {total_norm:.4f}")

    def on_train_end(self, trainer: "ARGUSTrainer") -> None:
        """Log gradient norm summary."""
        if self.gradient_norms:
            logger.info(
                f"Gradient norm range: {min(self.gradient_norms):.4f} - {max(self.gradient_norms):.4f}"
            )
            logger.info(f"Mean gradient norm: {np.mean(self.gradient_norms):.4f}")


class MetricsLogger(Callback):
    """
    Log metrics to various backends (console, file, wandb, tensorboard).

    Args:
        log_dir: Directory for log files.
            Default: 'logs'
        use_wandb: Whether to log to Weights & Biases.
            Default: False
        use_tensorboard: Whether to log to TensorBoard.
            Default: False
        wandb_project: W&B project name.
        wandb_config: Additional W&B config.

    Example:
        >>> callback = MetricsLogger(
        ...     use_tensorboard=True,
        ...     log_dir='logs/experiment_1'
        ... )
    """

    def __init__(
        self,
        log_dir: str = "logs",
        use_wandb: bool = False,
        use_tensorboard: bool = False,
        wandb_project: str | None = None,
        wandb_config: dict[str, Any] | None = None,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        self.wandb_project = wandb_project
        self.wandb_config = wandb_config or {}

        self.writer = None
        self.wandb_run = None

    def on_train_start(self, trainer: "ARGUSTrainer") -> None:
        """Initialize logging backends."""
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=str(self.log_dir))
                logger.info(f"TensorBoard logging to {self.log_dir}")
            except ImportError:
                logger.warning("TensorBoard not available")
                self.use_tensorboard = False

        if self.use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=self.wandb_project,
                    config={**self.wandb_config, **vars(trainer.config)},
                )
                logger.info("Initialized W&B logging")
            except ImportError:
                logger.warning("wandb not available")
                self.use_wandb = False

    def on_batch_end(
        self,
        trainer: "ARGUSTrainer",
        batch_idx: int,
        loss: float,
    ) -> None:
        """Log batch metrics."""
        step = trainer.global_step

        if self.use_tensorboard and self.writer is not None:
            self.writer.add_scalar("train/batch_loss", loss, step)

        if self.use_wandb:
            import wandb
            wandb.log({"train/batch_loss": loss}, step=step)

    def on_epoch_end(
        self,
        trainer: "ARGUSTrainer",
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        """Log epoch metrics."""
        if self.use_tensorboard and self.writer is not None:
            for name, value in metrics.items():
                self.writer.add_scalar(f"epoch/{name}", value, epoch)

        if self.use_wandb:
            import wandb
            wandb.log({f"epoch/{k}": v for k, v in metrics.items()}, step=epoch)

    def on_validation_end(
        self,
        trainer: "ARGUSTrainer",
        metrics: dict[str, float],
    ) -> None:
        """Log validation metrics."""
        epoch = trainer.current_epoch

        if self.use_tensorboard and self.writer is not None:
            for name, value in metrics.items():
                self.writer.add_scalar(f"val/{name}", value, epoch)

        if self.use_wandb:
            import wandb
            wandb.log({f"val/{k}": v for k, v in metrics.items()}, step=epoch)

    def on_train_end(self, trainer: "ARGUSTrainer") -> None:
        """Finalize logging."""
        if self.writer is not None:
            self.writer.close()

        if self.wandb_run is not None:
            import wandb
            wandb.finish()


class ProgressCallback(Callback):
    """
    Enhanced progress display during training.

    Args:
        refresh_rate: Progress bar refresh rate.
            Default: 1
        show_metrics: Metrics to show in progress bar.

    Example:
        >>> callback = ProgressCallback(show_metrics=['loss', 'accuracy'])
    """

    def __init__(
        self,
        refresh_rate: int = 1,
        show_metrics: list[str] | None = None,
    ) -> None:
        self.refresh_rate = refresh_rate
        self.show_metrics = show_metrics or ["loss"]

    def on_train_start(self, trainer: "ARGUSTrainer") -> None:
        """Display training start message."""
        n_params = sum(p.numel() for p in trainer.model.parameters())
        trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)

        print("\n" + "=" * 60)
        print("ARGUS Training")
        print("=" * 60)
        print(f"Total parameters: {n_params:,}")
        print(f"Trainable parameters: {trainable:,}")
        print(f"Device: {trainer.device}")
        print(f"Max epochs: {trainer.config.max_epochs}")
        print(f"Batch size: {trainer.config.batch_size}")
        print(f"Learning rate: {trainer.config.learning_rate}")
        print("=" * 60 + "\n")

    def on_train_end(self, trainer: "ARGUSTrainer") -> None:
        """Display training end message."""
        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"Best {trainer.config.early_stopping_metric}: {trainer.best_val_metric:.4f}")
        print(f"Total epochs: {trainer.current_epoch + 1}")
        print("=" * 60 + "\n")
