"""
ARGUS Utility Module.

This module provides common utilities for ARGUS:
- Logging: Logging configuration and utilities
- IO: File I/O utilities for checkpoints and configs
- Visualization: Plotting and visualization utilities
- Registry: Component registration system

Copyright 2024-2025 VTP Consortium
Licensed under the Apache License, Version 2.0
"""

from argus.utils.logging import (
    setup_logging,
    get_logger,
    log_config,
)
from argus.utils.io import (
    save_checkpoint,
    load_checkpoint,
    save_config,
    load_config,
)
from argus.utils.visualization import (
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_confusion_matrix,
    plot_calibration_curve,
    plot_feature_importance,
)
from argus.utils.registry import (
    Registry,
    MODEL_REGISTRY,
    LOSS_REGISTRY,
    OPTIMIZER_REGISTRY,
    SCHEDULER_REGISTRY,
)

__all__ = [
    # Logging
    "setup_logging",
    "get_logger",
    "log_config",
    # IO
    "save_checkpoint",
    "load_checkpoint",
    "save_config",
    "load_config",
    # Visualization
    "plot_roc_curve",
    "plot_precision_recall_curve",
    "plot_confusion_matrix",
    "plot_calibration_curve",
    "plot_feature_importance",
    # Registry
    "Registry",
    "MODEL_REGISTRY",
    "LOSS_REGISTRY",
    "OPTIMIZER_REGISTRY",
    "SCHEDULER_REGISTRY",
]
