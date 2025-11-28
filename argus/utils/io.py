"""
I/O Utilities.

File I/O utilities for checkpoints, configs, and data.

Copyright 2024-2025 VTP Consortium
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: _LRScheduler | None = None,
    epoch: int = 0,
    global_step: int = 0,
    metrics: dict[str, float] | None = None,
    config: Any = None,
    path: str | Path = "checkpoint.pt",
    target_names: list[str] | None = None,
) -> None:
    """
    Save model checkpoint.

    Args:
        model: Model to save.
        optimizer: Optimizer state.
        scheduler: Learning rate scheduler (optional).
        epoch: Current epoch.
        global_step: Current global step.
        metrics: Validation metrics.
        config: Model/training configuration.
        path: Output path.
        target_names: Names of prediction targets.

    Example:
        >>> save_checkpoint(
        ...     model=model,
        ...     optimizer=optimizer,
        ...     epoch=10,
        ...     metrics={'val_auroc': 0.85},
        ...     path='best_model.pt'
        ... )
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if metrics is not None:
        checkpoint["metrics"] = metrics

    if config is not None:
        if hasattr(config, "__dict__"):
            checkpoint["config"] = vars(config)
        elif isinstance(config, dict):
            checkpoint["config"] = config

    if target_names is not None:
        checkpoint["target_names"] = target_names

    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved: {path}")


def load_checkpoint(
    path: str | Path,
    model: nn.Module | None = None,
    optimizer: Optimizer | None = None,
    scheduler: _LRScheduler | None = None,
    device: str = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """
    Load model checkpoint.

    Args:
        path: Checkpoint path.
        model: Model to load weights into (optional).
        optimizer: Optimizer to load state into (optional).
        scheduler: Scheduler to load state into (optional).
        device: Device to map checkpoint to.
        strict: Whether to strictly enforce state dict keys match.

    Returns:
        Checkpoint dictionary with all saved data.

    Example:
        >>> checkpoint = load_checkpoint(
        ...     'best_model.pt',
        ...     model=model,
        ...     optimizer=optimizer
        ... )
        >>> print(f"Loaded from epoch {checkpoint['epoch']}")
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device)

    if model is not None:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        logger.info("Model weights loaded")

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info("Optimizer state loaded")

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        logger.info("Scheduler state loaded")

    logger.info(f"Checkpoint loaded from: {path}")
    return checkpoint


def save_config(
    config: Any,
    path: str | Path,
    format: str = "yaml",
) -> None:
    """
    Save configuration to file.

    Args:
        config: Configuration object or dictionary.
        path: Output path.
        format: Output format ('yaml' or 'json').

    Example:
        >>> save_config(training_config, 'config.yaml')
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(config, "__dict__"):
        config_dict = vars(config)
    elif isinstance(config, dict):
        config_dict = config
    else:
        config_dict = {"value": config}

    # Convert non-serializable objects
    config_dict = _make_serializable(config_dict)

    if format == "yaml":
        try:
            import yaml
            with open(path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=True)
        except ImportError:
            # Fall back to JSON
            format = "json"
            path = path.with_suffix(".json")

    if format == "json":
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2, sort_keys=True)

    logger.info(f"Config saved: {path}")


def load_config(
    path: str | Path,
) -> dict[str, Any]:
    """
    Load configuration from file.

    Args:
        path: Config file path (YAML or JSON).

    Returns:
        Configuration dictionary.

    Example:
        >>> config = load_config('config.yaml')
        >>> print(config['learning_rate'])
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()

    if suffix in [".yaml", ".yml"]:
        try:
            import yaml
            with open(path, "r") as f:
                config = yaml.safe_load(f)
        except ImportError:
            raise ImportError("PyYAML is required to load YAML configs")

    elif suffix == ".json":
        with open(path, "r") as f:
            config = json.load(f)

    else:
        raise ValueError(f"Unsupported config format: {suffix}")

    logger.info(f"Config loaded: {path}")
    return config


def _make_serializable(obj: Any) -> Any:
    """Convert object to JSON-serializable format."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif hasattr(obj, "__dict__"):
        return _make_serializable(vars(obj))
    else:
        return str(obj)


def export_onnx(
    model: nn.Module,
    path: str | Path,
    static_input_shape: tuple[int, ...] = (1, 63),
    temporal_input_shape: tuple[int, ...] = (1, 20, 117),
    opset_version: int = 14,
) -> None:
    """
    Export model to ONNX format.

    Args:
        model: Model to export.
        path: Output path.
        static_input_shape: Shape of static input.
        temporal_input_shape: Shape of temporal input.
        opset_version: ONNX opset version.

    Example:
        >>> export_onnx(model, 'model.onnx')
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()

    # Create dummy inputs
    static_input = torch.randn(*static_input_shape)
    temporal_input = torch.randn(*temporal_input_shape)

    # Export
    torch.onnx.export(
        model,
        (static_input, temporal_input),
        path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["static_features", "temporal_features"],
        output_names=["predictions"],
        dynamic_axes={
            "static_features": {0: "batch_size"},
            "temporal_features": {0: "batch_size", 1: "seq_len"},
            "predictions": {0: "batch_size"},
        },
    )

    logger.info(f"ONNX model exported: {path}")


def export_torchscript(
    model: nn.Module,
    path: str | Path,
    method: str = "trace",
    static_input_shape: tuple[int, ...] = (1, 63),
    temporal_input_shape: tuple[int, ...] = (1, 20, 117),
) -> None:
    """
    Export model to TorchScript format.

    Args:
        model: Model to export.
        path: Output path.
        method: Export method ('trace' or 'script').
        static_input_shape: Shape of static input.
        temporal_input_shape: Shape of temporal input.

    Example:
        >>> export_torchscript(model, 'model.pt', method='trace')
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()

    if method == "trace":
        static_input = torch.randn(*static_input_shape)
        temporal_input = torch.randn(*temporal_input_shape)

        scripted_model = torch.jit.trace(
            model,
            (static_input, temporal_input),
        )
    elif method == "script":
        scripted_model = torch.jit.script(model)
    else:
        raise ValueError(f"Unknown method: {method}")

    scripted_model.save(path)
    logger.info(f"TorchScript model exported: {path}")


class CheckpointManager:
    """
    Manager for handling multiple checkpoints.

    Automatically manages checkpoint saving with rotation and best model tracking.

    Args:
        save_dir: Directory to save checkpoints.
        max_checkpoints: Maximum number of checkpoints to keep.
        monitor: Metric to monitor for best checkpoint.
        mode: 'min' or 'max' for metric comparison.

    Example:
        >>> manager = CheckpointManager(
        ...     save_dir='checkpoints',
        ...     max_checkpoints=5,
        ...     monitor='val_auroc',
        ...     mode='max'
        ... )
        >>> manager.save(model, optimizer, epoch=10, metrics={'val_auroc': 0.85})
    """

    def __init__(
        self,
        save_dir: str | Path,
        max_checkpoints: int = 5,
        monitor: str = "val_loss",
        mode: str = "min",
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.monitor = monitor
        self.mode = mode

        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.checkpoints: list[tuple[float, Path]] = []

    def save(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        epoch: int,
        metrics: dict[str, float],
        **kwargs: Any,
    ) -> bool:
        """
        Save checkpoint with automatic management.

        Args:
            model: Model to save.
            optimizer: Optimizer state.
            epoch: Current epoch.
            metrics: Validation metrics.
            **kwargs: Additional arguments for save_checkpoint.

        Returns:
            True if this is a new best checkpoint.
        """
        current_value = metrics.get(self.monitor, 0.0)

        # Check if this is the best model
        is_best = False
        if self.mode == "min" and current_value < self.best_value:
            self.best_value = current_value
            is_best = True
        elif self.mode == "max" and current_value > self.best_value:
            self.best_value = current_value
            is_best = True

        # Save checkpoint
        filename = f"checkpoint_epoch_{epoch:04d}_{self.monitor}_{current_value:.4f}.pt"
        filepath = self.save_dir / filename

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=metrics,
            path=filepath,
            **kwargs,
        )

        self.checkpoints.append((current_value, filepath))

        # Save best model separately
        if is_best:
            best_path = self.save_dir / "best_model.pt"
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=metrics,
                path=best_path,
                **kwargs,
            )
            logger.info(f"New best model saved! {self.monitor}: {current_value:.4f}")

        # Clean up old checkpoints
        self._cleanup()

        return is_best

    def _cleanup(self) -> None:
        """Remove old checkpoints to maintain max_checkpoints limit."""
        if len(self.checkpoints) <= self.max_checkpoints:
            return

        # Sort by metric value
        reverse = self.mode == "max"
        self.checkpoints.sort(key=lambda x: x[0], reverse=reverse)

        # Remove worst checkpoints
        while len(self.checkpoints) > self.max_checkpoints:
            _, filepath = self.checkpoints.pop()
            if filepath.exists():
                filepath.unlink()
                logger.debug(f"Removed old checkpoint: {filepath}")

    def get_best_checkpoint(self) -> Path | None:
        """Get path to best checkpoint."""
        best_path = self.save_dir / "best_model.pt"
        return best_path if best_path.exists() else None

    def get_latest_checkpoint(self) -> Path | None:
        """Get path to latest checkpoint."""
        if not self.checkpoints:
            return None
        return self.checkpoints[-1][1]
