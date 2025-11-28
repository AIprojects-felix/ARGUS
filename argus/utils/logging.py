"""
Logging Utilities.

Logging configuration and utilities for ARGUS.

Copyright 2024-2025 VTP Consortium
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any


def setup_logging(
    level: int | str = logging.INFO,
    log_file: str | Path | None = None,
    format_string: str | None = None,
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    name: str = "argus",
) -> logging.Logger:
    """
    Setup logging configuration for ARGUS.

    Args:
        level: Logging level.
            Default: logging.INFO
        log_file: Optional file path for logging.
        format_string: Custom format string.
        datefmt: Date format string.
            Default: '%Y-%m-%d %H:%M:%S'
        name: Logger name.
            Default: 'argus'

    Returns:
        Configured logger instance.

    Example:
        >>> logger = setup_logging(level=logging.DEBUG, log_file='train.log')
        >>> logger.info("Training started")
    """
    if format_string is None:
        format_string = (
            "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
        )

    # Create formatter
    formatter = logging.Formatter(format_string, datefmt=datefmt)

    # Get logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "argus") -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name.
            Default: 'argus'

    Returns:
        Logger instance.

    Example:
        >>> logger = get_logger("argus.training")
        >>> logger.info("Starting training")
    """
    return logging.getLogger(name)


def log_config(config: Any, logger: logging.Logger | None = None) -> None:
    """
    Log configuration details.

    Args:
        config: Configuration object or dictionary.
        logger: Logger instance. Uses default if None.

    Example:
        >>> log_config(training_config)
    """
    if logger is None:
        logger = get_logger()

    logger.info("=" * 60)
    logger.info("Configuration:")
    logger.info("=" * 60)

    if hasattr(config, "__dict__"):
        config_dict = vars(config)
    elif isinstance(config, dict):
        config_dict = config
    else:
        config_dict = {"value": str(config)}

    for key, value in sorted(config_dict.items()):
        if not key.startswith("_"):
            logger.info(f"  {key}: {value}")

    logger.info("=" * 60)


class TqdmLoggingHandler(logging.Handler):
    """
    Logging handler that works with tqdm progress bars.

    Ensures log messages don't interfere with progress bar display.
    """

    def __init__(self, level: int = logging.NOTSET) -> None:
        super().__init__(level)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            from tqdm import tqdm
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


class MetricsLogger:
    """
    Logger for training metrics with automatic formatting.

    Provides convenient methods for logging metrics during training.

    Example:
        >>> metrics_logger = MetricsLogger()
        >>> metrics_logger.log_epoch(1, {'loss': 0.5, 'auroc': 0.85})
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or get_logger("argus.metrics")

    def log_epoch(self, epoch: int, metrics: dict[str, float]) -> None:
        """Log epoch metrics."""
        metrics_str = " | ".join([
            f"{k}: {v:.4f}" for k, v in sorted(metrics.items())
        ])
        self.logger.info(f"Epoch {epoch:3d} | {metrics_str}")

    def log_batch(
        self,
        epoch: int,
        batch: int,
        total_batches: int,
        loss: float,
    ) -> None:
        """Log batch metrics."""
        self.logger.debug(
            f"Epoch {epoch:3d} | Batch {batch:4d}/{total_batches} | Loss: {loss:.4f}"
        )

    def log_validation(self, epoch: int, metrics: dict[str, float]) -> None:
        """Log validation metrics."""
        metrics_str = " | ".join([
            f"{k}: {v:.4f}" for k, v in sorted(metrics.items())
        ])
        self.logger.info(f"Validation @ Epoch {epoch} | {metrics_str}")

    def log_best_model(self, metric_name: str, metric_value: float) -> None:
        """Log best model checkpoint."""
        self.logger.info(f"New best model! {metric_name}: {metric_value:.4f}")
