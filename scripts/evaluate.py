#!/usr/bin/env python
"""
ARGUS Evaluation Script.

Evaluate trained ARGUS model performance on test data.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --data data/test.npz
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --data data/test.npz --bootstrap

Copyright 2024-2025 VTP Consortium
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from argus.evaluation.bootstrap import BootstrapEvaluator
from argus.evaluation.calibration import CalibrationAnalyzer
from argus.evaluation.metrics import MultilabelMetrics
from argus.inference import ARGUSPredictor
from argus.models import ARGUS
from argus.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate ARGUS model on test data"
    )

    # Model
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to model configuration file (if not in checkpoint)",
    )

    # Data
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to test data file (NPZ format)",
    )

    # Evaluation options
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Compute bootstrap confidence intervals",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap samples",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Confidence level for intervals",
    )

    # Calibration
    parser.add_argument(
        "--calibration",
        action="store_true",
        help="Compute calibration metrics",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=10,
        help="Number of bins for calibration analysis",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save model predictions to file",
    )

    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cuda, cpu, or auto)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    return parser.parse_args()


def load_test_data(data_path: str) -> dict:
    """Load test data from file."""
    logger = logging.getLogger(__name__)
    logger.info(f"Loading test data from: {data_path}")

    data = np.load(data_path)

    required_keys = ["static_features", "temporal_features", "labels"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key in data file: {key}")

    loaded_data = {
        "static_features": data["static_features"],
        "temporal_features": data["temporal_features"],
        "labels": data["labels"],
    }

    if "temporal_mask" in data:
        loaded_data["temporal_mask"] = data["temporal_mask"]
    else:
        # Create full mask if not provided
        n_samples, seq_len = data["temporal_features"].shape[:2]
        loaded_data["temporal_mask"] = np.ones((n_samples, seq_len), dtype=np.int32)

    logger.info(f"Loaded {len(loaded_data['labels'])} samples")

    return loaded_data


def main():
    """Main evaluation function."""
    args = parse_args()

    # Setup logging
    setup_logging(log_level=args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("ARGUS Evaluation Pipeline")
    logger.info("=" * 60)

    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Load model
    logger.info(f"Loading model from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    if "model_config" in checkpoint:
        model_config = checkpoint["model_config"]
    elif args.config:
        import yaml
        with open(args.config) as f:
            model_config = yaml.safe_load(f)["model"]
    else:
        raise ValueError("Model config not found. Provide --config argument.")

    model = ARGUS(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info("Model loaded successfully")

    # Create predictor
    predictor = ARGUSPredictor(
        model=model,
        device=device,
        threshold=0.5,
    )

    # Load test data
    test_data = load_test_data(args.data)

    # Convert to tensors
    static = torch.tensor(test_data["static_features"], dtype=torch.float32)
    temporal = torch.tensor(test_data["temporal_features"], dtype=torch.float32)
    mask = torch.tensor(test_data["temporal_mask"], dtype=torch.bool)
    labels = test_data["labels"]

    # Run inference
    logger.info("Running inference...")
    n_samples = len(labels)
    all_probs = []

    with torch.no_grad():
        for i in range(0, n_samples, args.batch_size):
            batch_static = static[i:i + args.batch_size].to(device)
            batch_temporal = temporal[i:i + args.batch_size].to(device)
            batch_mask = mask[i:i + args.batch_size].to(device)

            results = predictor.predict_batch(batch_static, batch_temporal, batch_mask)
            all_probs.append(results["probabilities"].cpu().numpy())

    predictions = np.concatenate(all_probs, axis=0)
    logger.info(f"Inference complete. Shape: {predictions.shape}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute metrics
    logger.info("Computing evaluation metrics...")
    metrics = MultilabelMetrics(n_targets=predictions.shape[1])
    results = metrics.compute_all(
        targets=labels,
        predictions=predictions,
        threshold=0.5,
    )

    # Per-target metrics
    per_target_auc = metrics.compute_per_target_auc(labels, predictions)

    # Log results
    logger.info("\n" + "=" * 40)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 40)
    logger.info(f"Macro AUC-ROC: {results['macro_auc_roc']:.4f}")
    logger.info(f"Micro AUC-ROC: {results['micro_auc_roc']:.4f}")
    logger.info(f"Macro AUC-PR:  {results['macro_auc_pr']:.4f}")
    logger.info(f"Macro F1:      {results['macro_f1']:.4f}")
    logger.info(f"Micro F1:      {results['micro_f1']:.4f}")

    # Bootstrap confidence intervals
    if args.bootstrap:
        logger.info("\nComputing bootstrap confidence intervals...")
        bootstrap_eval = BootstrapEvaluator(
            n_bootstrap=args.n_bootstrap,
            confidence_level=args.confidence_level,
            method="percentile",
        )

        from argus.evaluation.metrics import calculate_auc_roc

        ci_results = bootstrap_eval.compute_metric_ci(
            metric_fn=lambda t, p: calculate_auc_roc(t, p),
            targets=labels,
            predictions=predictions,
        )

        logger.info(
            f"Macro AUC-ROC: {ci_results['mean']:.4f} "
            f"({args.confidence_level*100:.0f}% CI: "
            f"[{ci_results['ci_lower']:.4f}, {ci_results['ci_upper']:.4f}])"
        )
        results["bootstrap_ci"] = ci_results

    # Calibration analysis
    if args.calibration:
        logger.info("\nComputing calibration metrics...")
        calibration_analyzer = CalibrationAnalyzer(n_bins=args.n_bins)

        # Flatten for calibration analysis
        flat_labels = labels.flatten()
        flat_preds = predictions.flatten()

        cal_results = calibration_analyzer.analyze(flat_labels, flat_preds)

        logger.info(f"Expected Calibration Error (ECE): {cal_results['ece']:.4f}")
        logger.info(f"Maximum Calibration Error (MCE): {cal_results['mce']:.4f}")
        logger.info(f"Brier Score: {cal_results['brier_score']:.4f}")

        results["calibration"] = cal_results

    # Save results
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in results.items()
        }
        json.dump(json_results, f, indent=2)
    logger.info(f"\nResults saved to: {results_path}")

    # Save per-target metrics
    per_target_path = output_dir / "per_target_auc.json"
    with open(per_target_path, "w") as f:
        json.dump(per_target_auc, f, indent=2)
    logger.info(f"Per-target AUC saved to: {per_target_path}")

    # Save predictions
    if args.save_predictions:
        predictions_path = output_dir / "predictions.npz"
        np.savez(
            predictions_path,
            probabilities=predictions,
            predictions=(predictions > 0.5).astype(int),
            labels=labels,
        )
        logger.info(f"Predictions saved to: {predictions_path}")

    logger.info("\n" + "=" * 60)
    logger.info("Evaluation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
