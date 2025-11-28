#!/usr/bin/env python
"""
ARGUS Interpretation Script.

Generate model interpretability analysis for patient predictions.

Usage:
    python scripts/interpret.py --checkpoint checkpoints/best.pt --input patient.npz
    python scripts/interpret.py --checkpoint checkpoints/best.pt --input patient.npz --method integrated_gradients

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

from argus.interpretation import AttentionAnalyzer, FeatureAttributor, ClinicalInsights
from argus.models import ARGUS
from argus.utils.logging import setup_logging


# Feature names (example - should be customized for actual features)
STATIC_FEATURE_NAMES = [
    "Age", "Gender", "BMI", "Smoking_Status", "ECOG_PS",
    # ... add all 63 static feature names
] + [f"Static_{i}" for i in range(58)]

TEMPORAL_FEATURE_NAMES = [
    "WBC", "RBC", "Hemoglobin", "Platelet", "Neutrophil",
    # ... add all 117 temporal feature names
] + [f"Temporal_{i}" for i in range(112)]

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
        description="Generate interpretability analysis for ARGUS predictions"
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
        "--output-dir",
        type=str,
        default="interpretations",
        help="Directory to save interpretation results",
    )

    # Interpretation methods
    parser.add_argument(
        "--method",
        type=str,
        default="all",
        choices=["attention", "integrated_gradients", "occlusion", "all"],
        help="Interpretation method to use",
    )
    parser.add_argument(
        "--target-genes",
        type=str,
        nargs="+",
        default=None,
        help="Specific genes to analyze (default: top predicted)",
    )
    parser.add_argument(
        "--top-k-genes",
        type=int,
        default=5,
        help="Number of top predicted genes to analyze",
    )

    # Integrated gradients options
    parser.add_argument(
        "--n-steps",
        type=int,
        default=50,
        help="Number of steps for integrated gradients",
    )

    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cuda, cpu, or auto)",
    )

    # Output options
    parser.add_argument(
        "--save-figures",
        action="store_true",
        help="Save visualization figures",
    )
    parser.add_argument(
        "--figure-format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Figure output format",
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

    if "patient_ids" in data:
        loaded_data["patient_ids"] = data["patient_ids"]

    return loaded_data


def analyze_attention(
    analyzer: AttentionAnalyzer,
    static: torch.Tensor,
    temporal: torch.Tensor,
    output_dir: Path,
    save_figures: bool = False,
) -> dict:
    """Run attention analysis."""
    logger = logging.getLogger(__name__)
    logger.info("Running attention analysis...")

    results = {}

    # Extract raw attention weights
    attention_weights = analyzer.extract_attention(static, temporal)
    if attention_weights is not None:
        results["raw_attention"] = attention_weights.cpu().numpy().tolist()

    # Compute attention rollout
    rollout = analyzer.compute_attention_rollout(static, temporal)
    if rollout is not None:
        results["attention_rollout"] = rollout.cpu().numpy().tolist()

    # Get attention heatmap data
    heatmap_data = analyzer.get_attention_heatmap_data(static, temporal)
    results["heatmap_data"] = heatmap_data

    logger.info("Attention analysis complete")

    return results


def analyze_attributions(
    attributor: FeatureAttributor,
    static: torch.Tensor,
    temporal: torch.Tensor,
    target_indices: list,
    n_steps: int,
    output_dir: Path,
) -> dict:
    """Run feature attribution analysis."""
    logger = logging.getLogger(__name__)
    logger.info("Running feature attribution analysis...")

    results = {}

    for target_idx in target_indices:
        gene_name = GENE_NAMES[target_idx]
        logger.info(f"Analyzing attributions for {gene_name} (idx={target_idx})")

        gene_results = {}

        # Static feature attributions
        static_attr = attributor.integrated_gradients(
            static, temporal,
            target_idx=target_idx,
            feature_type="static",
            n_steps=n_steps,
        )
        gene_results["static_attributions"] = static_attr.cpu().numpy().tolist()

        # Temporal feature attributions
        temporal_attr = attributor.integrated_gradients(
            static, temporal,
            target_idx=target_idx,
            feature_type="temporal",
            n_steps=n_steps,
        )
        gene_results["temporal_attributions"] = temporal_attr.cpu().numpy().tolist()

        # Top contributing features
        static_importance = static_attr.abs().mean(dim=0).cpu().numpy()
        top_static = np.argsort(static_importance)[::-1][:10]
        gene_results["top_static_features"] = [
            {"feature": STATIC_FEATURE_NAMES[i], "importance": float(static_importance[i])}
            for i in top_static
        ]

        temporal_importance = temporal_attr.abs().mean(dim=(0, 1)).cpu().numpy()
        top_temporal = np.argsort(temporal_importance)[::-1][:10]
        gene_results["top_temporal_features"] = [
            {"feature": TEMPORAL_FEATURE_NAMES[i], "importance": float(temporal_importance[i])}
            for i in top_temporal
        ]

        results[gene_name] = gene_results

    logger.info("Feature attribution analysis complete")

    return results


def analyze_occlusion(
    attributor: FeatureAttributor,
    static: torch.Tensor,
    temporal: torch.Tensor,
    target_indices: list,
    output_dir: Path,
) -> dict:
    """Run occlusion analysis."""
    logger = logging.getLogger(__name__)
    logger.info("Running occlusion analysis...")

    results = {}

    for target_idx in target_indices:
        gene_name = GENE_NAMES[target_idx]
        logger.info(f"Analyzing occlusion for {gene_name} (idx={target_idx})")

        importance = attributor.occlusion_analysis(
            static, temporal, target_idx=target_idx
        )

        results[gene_name] = {
            "static_importance": importance["static"].cpu().numpy().tolist(),
            "temporal_importance": importance["temporal"].cpu().numpy().tolist(),
        }

    logger.info("Occlusion analysis complete")

    return results


def main():
    """Main interpretation function."""
    args = parse_args()

    # Setup logging
    setup_logging(log_level=args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("ARGUS Interpretation Pipeline")
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

    # Load input data
    input_data = load_input_data(args.input)

    # Convert to tensors (use first sample for interpretation)
    static = torch.tensor(input_data["static_features"][:1], dtype=torch.float32).to(device)
    temporal = torch.tensor(input_data["temporal_features"][:1], dtype=torch.float32).to(device)

    # Get model predictions first
    with torch.no_grad():
        output = model(static, temporal)
        if isinstance(output, dict):
            logits = output.get("logits", output.get("predictions"))
        else:
            logits = output
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    # Determine target genes to analyze
    if args.target_genes:
        target_indices = [GENE_NAMES.index(g) for g in args.target_genes]
    else:
        # Use top predicted genes
        top_indices = np.argsort(probs)[::-1][:args.top_k_genes]
        target_indices = top_indices.tolist()

    logger.info(f"Analyzing genes: {[GENE_NAMES[i] for i in target_indices]}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize analyzers
    attention_analyzer = AttentionAnalyzer(model=model, device=device)
    feature_attributor = FeatureAttributor(model=model, device=device)

    # Run analyses
    all_results = {
        "patient_id": input_data.get("patient_ids", ["unknown"])[0],
        "predictions": {
            gene: float(probs[i]) for i, gene in enumerate(GENE_NAMES)
        },
        "analyzed_genes": [GENE_NAMES[i] for i in target_indices],
    }

    if args.method in ["attention", "all"]:
        all_results["attention"] = analyze_attention(
            attention_analyzer, static, temporal, output_dir, args.save_figures
        )

    if args.method in ["integrated_gradients", "all"]:
        all_results["attributions"] = analyze_attributions(
            feature_attributor, static, temporal, target_indices, args.n_steps, output_dir
        )

    if args.method in ["occlusion", "all"]:
        all_results["occlusion"] = analyze_occlusion(
            feature_attributor, static, temporal, target_indices, output_dir
        )

    # Generate clinical insights
    clinical_insights = ClinicalInsights(
        model=model,
        device=device,
        gene_names=GENE_NAMES,
        feature_names={
            "static": STATIC_FEATURE_NAMES,
            "temporal": TEMPORAL_FEATURE_NAMES,
        },
    )

    explanation = clinical_insights.explain_patient(static[0], temporal[0])
    all_results["clinical_explanation"] = explanation

    # Save results
    results_path = output_dir / "interpretation_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Results saved to: {results_path}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Interpretation Summary")
    logger.info("=" * 60)

    for gene_name in all_results["analyzed_genes"]:
        logger.info(f"\n{gene_name}:")
        logger.info(f"  Prediction probability: {probs[GENE_NAMES.index(gene_name)]:.4f}")

        if "attributions" in all_results and gene_name in all_results["attributions"]:
            top_features = all_results["attributions"][gene_name]["top_static_features"][:3]
            logger.info("  Top contributing static features:")
            for feat in top_features:
                logger.info(f"    - {feat['feature']}: {feat['importance']:.4f}")

    logger.info("\n" + "=" * 60)
    logger.info("Interpretation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
