#!/usr/bin/env python
"""
ARGUS Prediction Script.

Generate predictions for new patient data using trained ARGUS model.

Usage:
    python scripts/predict.py --checkpoint checkpoints/best.pt --input patient_data.npz
    python scripts/predict.py --checkpoint checkpoints/best.pt --input patient_data.npz --output predictions.json

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

from argus.inference import ARGUSPredictor, InferencePipeline
from argus.models import ARGUS
from argus.utils.logging import setup_logging


# Gene names for the 43 targets
GENE_NAMES = [
    "EGFR", "KRAS", "TP53", "PIK3CA", "BRAF", "ALK", "ROS1", "MET", "RET",
    "ERBB2", "NTRK1", "NTRK2", "NTRK3", "BRCA1", "BRCA2", "ATM", "PALB2",
    "CHEK2", "APC", "MLH1", "MSH2", "MSH6", "PMS2", "PTEN", "STK11",
    "CDKN2A", "NF1", "RB1", "SMAD4", "SMARCA4", "KEAP1", "NFE2L2", "ARID1A",
    "FGFR1", "FGFR2", "FGFR3", "IDH1", "IDH2", "DNMT3A", "JAK2",
    "TMB_High", "MSI_High", "PD-L1_High"
]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate genomic predictions for patient data"
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
        help="Path to model configuration file",
    )

    # Input/Output
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input data file (NPZ format)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.json",
        help="Path to output predictions file",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="json",
        choices=["json", "csv", "npz"],
        help="Output format for predictions",
    )

    # Prediction options
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold",
    )
    parser.add_argument(
        "--uncertainty",
        action="store_true",
        help="Estimate prediction uncertainty using MC dropout",
    )
    parser.add_argument(
        "--n-mc-samples",
        type=int,
        default=30,
        help="Number of MC samples for uncertainty estimation",
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

    # Report
    parser.add_argument(
        "--clinical-report",
        action="store_true",
        help="Generate clinical report format",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output messages",
    )

    return parser.parse_args()


def load_input_data(input_path: str) -> dict:
    """Load input data from file."""
    logger = logging.getLogger(__name__)
    logger.info(f"Loading input data from: {input_path}")

    data = np.load(input_path)

    loaded_data = {
        "static_features": data["static_features"],
        "temporal_features": data["temporal_features"],
    }

    if "temporal_mask" in data:
        loaded_data["temporal_mask"] = data["temporal_mask"]
    else:
        # Create full mask
        n_samples, seq_len = data["temporal_features"].shape[:2]
        loaded_data["temporal_mask"] = np.ones((n_samples, seq_len), dtype=np.int32)

    if "patient_ids" in data:
        loaded_data["patient_ids"] = data["patient_ids"]
    else:
        loaded_data["patient_ids"] = [f"patient_{i}" for i in range(len(data["static_features"]))]

    logger.info(f"Loaded {len(loaded_data['static_features'])} samples")

    return loaded_data


def format_clinical_report(
    patient_id: str,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    gene_names: list,
    uncertainty: np.ndarray | None = None,
) -> dict:
    """Format predictions as a clinical report."""
    report = {
        "patient_id": patient_id,
        "summary": {
            "total_predicted_mutations": int(predictions.sum()),
            "high_confidence_predictions": int((probabilities > 0.8).sum()),
        },
        "gene_mutations": {},
        "biomarkers": {},
    }

    # Gene predictions (first 40)
    for i, gene in enumerate(gene_names[:40]):
        if predictions[i] == 1:
            entry = {
                "probability": float(probabilities[i]),
                "status": "POSITIVE" if probabilities[i] > 0.8 else "LIKELY POSITIVE",
            }
            if uncertainty is not None:
                entry["uncertainty"] = float(uncertainty[i])
            report["gene_mutations"][gene] = entry

    # Biomarkers (last 3)
    biomarker_names = ["TMB", "MSI", "PD-L1"]
    for i, name in enumerate(biomarker_names):
        idx = 40 + i
        report["biomarkers"][name] = {
            "probability": float(probabilities[idx]),
            "status": "HIGH" if predictions[idx] == 1 else "LOW/UNKNOWN",
            "confidence": "HIGH" if abs(probabilities[idx] - 0.5) > 0.3 else "MODERATE",
        }
        if uncertainty is not None:
            report["biomarkers"][name]["uncertainty"] = float(uncertainty[idx])

    return report


def main():
    """Main prediction function."""
    args = parse_args()

    # Setup logging
    if not args.quiet:
        setup_logging(log_level=args.log_level)
    logger = logging.getLogger(__name__)

    if not args.quiet:
        logger.info("=" * 60)
        logger.info("ARGUS Prediction Pipeline")
        logger.info("=" * 60)

    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if not args.quiet:
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

    # Create predictor
    predictor = ARGUSPredictor(
        model=model,
        device=device,
        threshold=args.threshold,
        enable_uncertainty=args.uncertainty,
        n_mc_samples=args.n_mc_samples,
    )

    # Load input data
    input_data = load_input_data(args.input)

    # Convert to tensors
    static = torch.tensor(input_data["static_features"], dtype=torch.float32)
    temporal = torch.tensor(input_data["temporal_features"], dtype=torch.float32)
    mask = torch.tensor(input_data["temporal_mask"], dtype=torch.bool)
    patient_ids = input_data["patient_ids"]

    # Run predictions
    logger.info("Running predictions...")
    n_samples = len(static)
    all_probs = []
    all_preds = []
    all_uncert = []

    with torch.no_grad():
        for i in range(0, n_samples, args.batch_size):
            batch_static = static[i:i + args.batch_size].to(device)
            batch_temporal = temporal[i:i + args.batch_size].to(device)
            batch_mask = mask[i:i + args.batch_size].to(device)

            if args.uncertainty:
                results = predictor.predict_with_uncertainty(
                    batch_static, batch_temporal, batch_mask
                )
                all_uncert.append(results["uncertainty"].cpu().numpy())
            else:
                results = predictor.predict_batch(batch_static, batch_temporal, batch_mask)

            all_probs.append(results["probabilities"].cpu().numpy())
            all_preds.append(results["predictions"].cpu().numpy())

    probabilities = np.concatenate(all_probs, axis=0)
    predictions = np.concatenate(all_preds, axis=0)
    uncertainty = np.concatenate(all_uncert, axis=0) if args.uncertainty else None

    logger.info(f"Predictions complete for {n_samples} samples")

    # Format output
    if args.clinical_report:
        output_data = {
            "reports": [],
            "metadata": {
                "model_checkpoint": args.checkpoint,
                "threshold": args.threshold,
                "n_samples": n_samples,
            }
        }

        for i in range(n_samples):
            report = format_clinical_report(
                patient_id=patient_ids[i] if isinstance(patient_ids[i], str)
                          else patient_ids[i].decode() if hasattr(patient_ids[i], 'decode')
                          else str(patient_ids[i]),
                probabilities=probabilities[i],
                predictions=predictions[i],
                gene_names=GENE_NAMES,
                uncertainty=uncertainty[i] if uncertainty is not None else None,
            )
            output_data["reports"].append(report)

    else:
        output_data = {
            "patient_ids": [
                p if isinstance(p, str) else p.decode() if hasattr(p, 'decode') else str(p)
                for p in patient_ids
            ],
            "probabilities": probabilities.tolist(),
            "predictions": predictions.tolist(),
            "gene_names": GENE_NAMES,
        }
        if uncertainty is not None:
            output_data["uncertainty"] = uncertainty.tolist()

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.output_format == "json":
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

    elif args.output_format == "csv":
        import csv
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            # Header
            header = ["patient_id"] + [f"{g}_prob" for g in GENE_NAMES] + [f"{g}_pred" for g in GENE_NAMES]
            writer.writerow(header)
            # Data
            for i in range(n_samples):
                row = [output_data["patient_ids"][i]]
                row.extend(probabilities[i].tolist())
                row.extend(predictions[i].tolist())
                writer.writerow(row)

    elif args.output_format == "npz":
        np.savez(
            output_path,
            patient_ids=patient_ids,
            probabilities=probabilities,
            predictions=predictions,
            gene_names=GENE_NAMES,
            **({"uncertainty": uncertainty} if uncertainty is not None else {}),
        )

    logger.info(f"Predictions saved to: {output_path}")

    if not args.quiet:
        logger.info("\n" + "=" * 60)
        logger.info("Prediction Summary")
        logger.info("=" * 60)
        logger.info(f"Total samples: {n_samples}")
        logger.info(f"Average mutations predicted per sample: {predictions.sum(axis=1).mean():.2f}")
        logger.info(f"Most common mutations:")

        mutation_freq = predictions.sum(axis=0)
        top_indices = np.argsort(mutation_freq)[::-1][:5]
        for idx in top_indices:
            logger.info(f"  {GENE_NAMES[idx]}: {int(mutation_freq[idx])} ({100*mutation_freq[idx]/n_samples:.1f}%)")


if __name__ == "__main__":
    main()
