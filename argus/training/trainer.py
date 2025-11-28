"""
ARGUS Trainer Implementation.

Main training loop and utilities for training the ARGUS model.

Copyright 2024-2025 VTP Consortium
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from argus.training.schedulers import CosineAnnealingWarmup
from argus.training.callbacks import Callback

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """
    Configuration for ARGUS training.

    Args:
        max_epochs: Maximum number of training epochs.
        batch_size: Training batch size.
        learning_rate: Initial learning rate.
        weight_decay: L2 regularization weight.
        warmup_epochs: Number of warmup epochs.
        gradient_clip_val: Maximum gradient norm for clipping.
        accumulate_grad_batches: Number of batches to accumulate gradients.
        mixed_precision: Whether to use automatic mixed precision.
        checkpoint_dir: Directory for saving checkpoints.
        log_every_n_steps: Logging frequency.
        val_check_interval: Validation frequency (fraction of epoch or int).
        early_stopping_patience: Patience for early stopping.
        early_stopping_metric: Metric to monitor for early stopping.
        early_stopping_mode: 'min' or 'max' for the metric.
    """

    max_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    mixed_precision: bool = True
    checkpoint_dir: str = "checkpoints"
    log_every_n_steps: int = 50
    val_check_interval: float = 1.0
    early_stopping_patience: int = 10
    early_stopping_metric: str = "val_loss"
    early_stopping_mode: str = "min"
    seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Additional training options
    label_smoothing: float = 0.0
    pos_weight_factor: float = 1.0
    use_focal_loss: bool = False
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25

    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1
    stochastic_depth: float = 0.0

    # Optimizer
    optimizer: str = "adamw"
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    # Scheduler
    scheduler: str = "cosine_warmup"
    min_lr: float = 1e-6

    # Additional fields for tracking
    resume_from: str | None = None
    tags: list[str] = field(default_factory=list)


class ARGUSTrainer:
    """
    Main trainer for ARGUS model.

    Implements the training loop with support for:
    - Mixed precision training
    - Gradient accumulation
    - Learning rate scheduling with warmup
    - Early stopping
    - Checkpoint saving and loading
    - Comprehensive logging

    Args:
        model: ARGUS model to train.
        config: Training configuration.
        train_dataloader: Training data loader.
        val_dataloader: Validation data loader.
        loss_fn: Loss function.
        optimizer: Optimizer (created if not provided).
        scheduler: Learning rate scheduler (created if not provided).
        callbacks: List of training callbacks.

    Example:
        >>> config = TrainingConfig(max_epochs=100, learning_rate=1e-4)
        >>> trainer = ARGUSTrainer(
        ...     model=model,
        ...     config=config,
        ...     train_dataloader=train_loader,
        ...     val_dataloader=val_loader,
        ...     loss_fn=loss_fn,
        ... )
        >>> trainer.fit()
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None = None,
        loss_fn: nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        callbacks: list[Callback] | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.callbacks = callbacks or []

        # Setup device
        self.device = torch.device(config.device)
        self.model = self.model.to(self.device)

        # Setup loss function
        self.loss_fn = loss_fn or self._create_default_loss()

        # Setup optimizer
        self.optimizer = optimizer or self._create_optimizer()

        # Setup scheduler
        self.scheduler = scheduler or self._create_scheduler()

        # Setup mixed precision
        self.scaler = GradScaler() if config.mixed_precision else None

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_metric = float('inf') if config.early_stopping_mode == 'min' else float('-inf')
        self.patience_counter = 0
        self.should_stop = False

        # Metrics tracking
        self.train_metrics: dict[str, list[float]] = {}
        self.val_metrics: dict[str, list[float]] = {}

        # Create checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Set random seed
        self._set_seed(config.seed)

    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    def _create_default_loss(self) -> nn.Module:
        """Create default loss function."""
        from argus.models.losses import BCEWithLogitsLoss, FocalLoss

        if self.config.use_focal_loss:
            return FocalLoss(
                alpha=self.config.focal_alpha,
                gamma=self.config.focal_gamma,
            )
        else:
            return BCEWithLogitsLoss(
                label_smoothing=self.config.label_smoothing,
            )

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on config."""
        from argus.training.optimizers import create_optimizer

        return create_optimizer(
            model=self.model,
            optimizer_name=self.config.optimizer,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.eps,
        )

    def _create_scheduler(self) -> Any:
        """Create learning rate scheduler."""
        total_steps = len(self.train_dataloader) * self.config.max_epochs
        warmup_steps = len(self.train_dataloader) * self.config.warmup_epochs

        return CosineAnnealingWarmup(
            optimizer=self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=self.config.min_lr,
        )

    def fit(self) -> dict[str, Any]:
        """
        Run the full training loop.

        Returns:
            Dictionary with training history and final metrics.
        """
        logger.info(f"Starting training for {self.config.max_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Resume from checkpoint if specified
        if self.config.resume_from:
            self.load_checkpoint(self.config.resume_from)

        # Notify callbacks
        self._call_callbacks("on_train_start")

        start_time = time.time()

        try:
            for epoch in range(self.current_epoch, self.config.max_epochs):
                self.current_epoch = epoch

                # Train one epoch
                train_metrics = self._train_epoch()

                # Validation
                val_metrics = {}
                if self.val_dataloader is not None:
                    val_metrics = self._validate()

                # Log metrics
                self._log_epoch_metrics(train_metrics, val_metrics)

                # Check early stopping
                if self._check_early_stopping(val_metrics):
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break

                # Save checkpoint
                self._save_checkpoint(val_metrics)

                if self.should_stop:
                    break

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")

        # Finalize training
        self._call_callbacks("on_train_end")

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f}s")

        return {
            "train_metrics": self.train_metrics,
            "val_metrics": self.val_metrics,
            "best_val_metric": self.best_val_metric,
            "total_epochs": self.current_epoch + 1,
            "total_time": total_time,
        }

    def _train_epoch(self) -> dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of training metrics for this epoch.
        """
        self.model.train()
        self._call_callbacks("on_epoch_start", epoch=self.current_epoch)

        epoch_loss = 0.0
        epoch_samples = 0
        accumulation_counter = 0

        pbar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.current_epoch}",
            leave=False,
        )

        for batch_idx, batch in enumerate(pbar):
            self._call_callbacks("on_batch_start", batch_idx=batch_idx)

            # Move batch to device
            batch = self._move_batch_to_device(batch)

            # Forward pass with mixed precision
            if self.scaler is not None:
                with autocast():
                    loss, metrics = self._forward_step(batch)
                    loss = loss / self.config.accumulate_grad_batches

                # Backward pass
                self.scaler.scale(loss).backward()
            else:
                loss, metrics = self._forward_step(batch)
                loss = loss / self.config.accumulate_grad_batches
                loss.backward()

            accumulation_counter += 1

            # Optimizer step
            if accumulation_counter >= self.config.accumulate_grad_batches:
                if self.config.gradient_clip_val > 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_val,
                    )

                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.scheduler.step()
                accumulation_counter = 0

            # Update metrics
            batch_size = batch["targets"].shape[0]
            epoch_loss += loss.item() * self.config.accumulate_grad_batches * batch_size
            epoch_samples += batch_size
            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item() * self.config.accumulate_grad_batches:.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
            })

            self._call_callbacks(
                "on_batch_end",
                batch_idx=batch_idx,
                loss=loss.item() * self.config.accumulate_grad_batches,
            )

            # Periodic logging
            if self.global_step % self.config.log_every_n_steps == 0:
                self._log_step_metrics(metrics)

        # Compute epoch metrics
        avg_loss = epoch_loss / epoch_samples if epoch_samples > 0 else 0.0

        epoch_metrics = {"train_loss": avg_loss}
        self._call_callbacks("on_epoch_end", epoch=self.current_epoch, metrics=epoch_metrics)

        return epoch_metrics

    def _forward_step(
        self, batch: dict[str, Tensor]
    ) -> tuple[Tensor, dict[str, float]]:
        """
        Forward pass for a single batch.

        Args:
            batch: Batch dictionary.

        Returns:
            Tuple of (loss, metrics_dict).
        """
        # Extract inputs
        static = batch["static"]
        temporal = batch["temporal"]
        targets = batch["targets"]

        temporal_mask = batch.get("temporal_mask")
        feature_mask = batch.get("feature_mask")
        time_deltas = batch.get("time_deltas")
        target_mask = batch.get("target_mask")

        # Forward pass
        outputs = self.model(
            static_features=static,
            temporal_features=temporal,
            temporal_mask=temporal_mask,
            feature_mask=feature_mask,
            time_deltas=time_deltas,
        )

        logits = outputs["logits"]

        # Compute loss
        if target_mask is not None:
            # Masked loss for missing labels
            loss = self._compute_masked_loss(logits, targets, target_mask)
        else:
            loss = self.loss_fn(logits, targets)

        # Compute additional metrics
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            # Accuracy (for valid targets only)
            if target_mask is not None:
                correct = ((preds == targets) * target_mask).sum()
                total = target_mask.sum()
            else:
                correct = (preds == targets).sum()
                total = targets.numel()

            accuracy = correct / total if total > 0 else 0.0

        metrics = {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
        }

        return loss, metrics

    def _compute_masked_loss(
        self,
        logits: Tensor,
        targets: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Compute loss with masking for missing labels."""
        # Compute per-element loss
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        # Apply mask
        masked_loss = loss * mask.float()

        # Mean over valid elements
        return masked_loss.sum() / mask.sum().clamp(min=1)

    def _validate(self) -> dict[str, float]:
        """
        Run validation.

        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        self._call_callbacks("on_validation_start")

        total_loss = 0.0
        total_samples = 0
        all_preds = []
        all_targets = []
        all_masks = []

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation", leave=False):
                batch = self._move_batch_to_device(batch)

                if self.scaler is not None:
                    with autocast():
                        loss, _ = self._forward_step(batch)
                else:
                    loss, _ = self._forward_step(batch)

                batch_size = batch["targets"].shape[0]
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                # Collect predictions for metrics
                outputs = self.model(
                    static_features=batch["static"],
                    temporal_features=batch["temporal"],
                    temporal_mask=batch.get("temporal_mask"),
                    feature_mask=batch.get("feature_mask"),
                    time_deltas=batch.get("time_deltas"),
                )

                probs = torch.sigmoid(outputs["logits"])
                all_preds.append(probs.cpu())
                all_targets.append(batch["targets"].cpu())
                if "target_mask" in batch:
                    all_masks.append(batch["target_mask"].cpu())

        # Compute metrics
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        if all_masks:
            all_masks = torch.cat(all_masks, dim=0)
            metrics = self._compute_validation_metrics(all_preds, all_targets, all_masks)
        else:
            metrics = self._compute_validation_metrics(all_preds, all_targets)

        metrics["val_loss"] = avg_loss

        self._call_callbacks("on_validation_end", metrics=metrics)

        return metrics

    def _compute_validation_metrics(
        self,
        preds: Tensor,
        targets: Tensor,
        mask: Tensor | None = None,
    ) -> dict[str, float]:
        """Compute comprehensive validation metrics."""
        from sklearn.metrics import roc_auc_score, average_precision_score

        preds_np = preds.numpy()
        targets_np = targets.numpy()

        if mask is not None:
            mask_np = mask.numpy()
        else:
            mask_np = np.ones_like(targets_np, dtype=bool)

        # Per-target metrics
        aucs = []
        aps = []

        for i in range(targets_np.shape[1]):
            target_mask = mask_np[:, i] if mask_np.ndim > 1 else mask_np
            valid_targets = targets_np[target_mask, i]
            valid_preds = preds_np[target_mask, i]

            # Need both classes present
            if len(np.unique(valid_targets)) < 2:
                continue

            try:
                auc = roc_auc_score(valid_targets, valid_preds)
                ap = average_precision_score(valid_targets, valid_preds)
                aucs.append(auc)
                aps.append(ap)
            except ValueError:
                continue

        return {
            "val_auroc_mean": np.mean(aucs) if aucs else 0.0,
            "val_auprc_mean": np.mean(aps) if aps else 0.0,
            "val_auroc_std": np.std(aucs) if aucs else 0.0,
        }

    def _move_batch_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move batch tensors to device."""
        moved = {}
        for key, value in batch.items():
            if isinstance(value, Tensor):
                moved[key] = value.to(self.device)
            else:
                moved[key] = value
        return moved

    def _check_early_stopping(self, val_metrics: dict[str, float]) -> bool:
        """Check if early stopping criteria is met."""
        if not val_metrics or self.config.early_stopping_metric not in val_metrics:
            return False

        current_metric = val_metrics[self.config.early_stopping_metric]

        if self.config.early_stopping_mode == "min":
            is_better = current_metric < self.best_val_metric
        else:
            is_better = current_metric > self.best_val_metric

        if is_better:
            self.best_val_metric = current_metric
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.config.early_stopping_patience

    def _save_checkpoint(self, val_metrics: dict[str, float]) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
            "val_metrics": val_metrics,
            "best_val_metric": self.best_val_metric,
        }

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        # Save latest checkpoint
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)

        # Save best checkpoint
        if self.patience_counter == 0:  # Just improved
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model with {self.config.early_stopping_metric}: {self.best_val_metric:.4f}")

        # Save periodic checkpoint
        if self.current_epoch % 10 == 0:
            epoch_path = self.checkpoint_dir / f"epoch_{self.current_epoch}.pt"
            torch.save(checkpoint, epoch_path)

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        logger.info(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"] + 1
        self.global_step = checkpoint["global_step"]
        self.best_val_metric = checkpoint["best_val_metric"]

        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

    def _log_epoch_metrics(
        self,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
    ) -> None:
        """Log epoch metrics."""
        # Store metrics
        for key, value in train_metrics.items():
            if key not in self.train_metrics:
                self.train_metrics[key] = []
            self.train_metrics[key].append(value)

        for key, value in val_metrics.items():
            if key not in self.val_metrics:
                self.val_metrics[key] = []
            self.val_metrics[key].append(value)

        # Log to console
        log_str = f"Epoch {self.current_epoch}"
        for key, value in train_metrics.items():
            log_str += f" | {key}: {value:.4f}"
        for key, value in val_metrics.items():
            log_str += f" | {key}: {value:.4f}"

        logger.info(log_str)

    def _log_step_metrics(self, metrics: dict[str, float]) -> None:
        """Log step-level metrics."""
        # Can be extended for tensorboard, wandb, etc.
        pass

    def _call_callbacks(self, hook_name: str, **kwargs: Any) -> None:
        """Call all callbacks for a given hook."""
        for callback in self.callbacks:
            hook = getattr(callback, hook_name, None)
            if hook is not None:
                hook(trainer=self, **kwargs)
