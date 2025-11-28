"""
ARGUS: AI-based Routine Genomic Understanding System.

A pan-cancer deep learning framework for non-invasive genomic profiling
from longitudinal electronic health records (EHRs).

This framework provides:
- Dual-encoder architecture (Static + Temporal Transformer)
- Multi-label prediction of 40+ actionable driver genes
- Biomarker inference (TMB, MSI, PD-L1)
- Model interpretability with SHAP analysis
- Survival analysis for therapeutic stratification

Example:
    >>> from argus.models import ARGUS
    >>> from argus.data import ARGUSDataset
    >>>
    >>> model = ARGUS(
    ...     static_dim=18,
    ...     temporal_dim=180,
    ...     d_model=256,
    ...     n_heads=8,
    ...     n_layers=6,
    ...     n_targets=43
    ... )
    >>> output = model(static_features, temporal_features)

For more information, see:
- Documentation: https://github.com/AIprojects-felix/ARGUS
- Paper: "A Pan-Cancer AI Framework for Non-Invasive Genomic Profiling"

Copyright 2024-2025 VTP Consortium
Licensed under the Apache License, Version 2.0
"""

from argus.__version__ import __version__, __version_info__

__author__ = "VTP Consortium"
__email__ = "liufei_2359@163.com"
__license__ = "Apache-2.0"

# Expose main classes at package level
from argus.models import ARGUS
from argus.data import ARGUSDataset, ARGUSDataModule
from argus.training import ARGUSTrainer
from argus.inference import ARGUSPredictor

__all__ = [
    # Version
    "__version__",
    "__version_info__",
    # Models
    "ARGUS",
    # Data
    "ARGUSDataset",
    "ARGUSDataModule",
    # Training
    "ARGUSTrainer",
    # Inference
    "ARGUSPredictor",
]
