#!/usr/bin/env python
"""
ARGUS Training Script.

Train the ARGUS model for genomic mutation prediction from EHR data.

Usage:
    python scripts/train.py --config configs/training/default.yaml
    python scripts/train.py --config configs/training/default.yaml --gpus 2

Copyright 2024-2025 VTP Consortium
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from argus.data import ARGUSDataModule
from argus.models import ARGUS
from argus.training import ARGUSTrainer
from argus.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train ARGUS model for genomic mutation prediction"
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/default.yaml",
        help="Path to training configuration file",
    )

    # Data
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override data directory from config",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default=None,
        help="Path to training data file",
    )
    parser.add_argument(
        "--val-file",
        type=str,
        default=None,
        help="Path to validation data file",
    )

    # Training
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs from config",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size from config",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate from config",
    )

    # Hardware
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (default: all available)",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable mixed precision training",
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )

    # Logging
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="argus_training",
        help="Name for this experiment run",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for training logs",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def merge_args_with_config(args: argparse.Namespace, config: dict) -> dict:
    """Merge command line arguments with config file settings."""
    # Command line arguments override config file
    if args.epochs is not None:
        config["training"]["max_epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
    if args.gpus is not None:
        config["training"]["gpus"] = args.gpus
    if args.mixed_precision:
        config["training"]["mixed_precision"] = True
    if args.seed is not None:
        config["training"]["seed"] = args.seed

    # Override paths
    if args.data_dir is not None:
        config["data"]["data_dir"] = args.data_dir
    if args.train_file is not None:
        config["data"]["train_file"] = args.train_file
    if args.val_file is not None:
        config["data"]["val_file"] = args.val_file

    config["checkpoint_dir"] = args.checkpoint_dir
    config["log_dir"] = args.log_dir
    config["experiment_name"] = args.experiment_name

    return config


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    """Main training function."""
    args = parse_args()

    # Setup logging
    setup_logging(
        log_dir=args.log_dir,
        log_level=args.log_level,
        experiment_name=args.experiment_name,
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("ARGUS Training Pipeline")
    logger.info("=" * 60)

    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    config = merge_args_with_config(args, config)

    # Set random seed
    seed = config.get("training", {}).get("seed", 42)
    set_seed(seed)
    logger.info(f"Random seed set to: {seed}")

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"CUDA devices available: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # Create data module
    logger.info("Creating data module...")
    data_config = config.get("data", {})

    # Placeholder: In production, load actual data files
    # For now, create placeholder to demonstrate structure
    logger.info("Note: This script requires actual data files to be provided.")
    logger.info("Expected data format: NumPy arrays or PyTorch tensors")
    logger.info("  - static_features: (N, 63)")
    logger.info("  - temporal_features: (N, T, 117)")
    logger.info("  - temporal_mask: (N, T)")
    logger.info("  - labels: (N, 43)")

    # Create model
    logger.info("Creating ARGUS model...")
    model_config = config.get("model", {})
    model = ARGUS(**model_config)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {n_params:,}")
    logger.info(f"Trainable parameters: {n_trainable:,}")

    # Create trainer
    logger.info("Creating trainer...")
    training_config = config.get("training", {})

    trainer = ARGUSTrainer(
        model=model,
        config=training_config,
        checkpoint_dir=config.get("checkpoint_dir", "checkpoints"),
        experiment_name=config.get("experiment_name", "argus_training"),
    )

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming training from: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Training would happen here with actual data
    logger.info("Training setup complete.")
    logger.info(
        "To run actual training, provide data files using --train-file and --val-file"
    )

    logger.info("=" * 60)
    logger.info("Training pipeline ready")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
