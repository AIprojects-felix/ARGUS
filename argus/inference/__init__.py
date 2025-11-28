"""
ARGUS Inference Module.

This module provides inference and clinical deployment utilities:
- Predictor: Batch and single-sample prediction interfaces
- Pipeline: End-to-end inference pipeline with preprocessing
- Server: Model serving utilities for deployment

Copyright 2024-2025 VTP Consortium
Licensed under the Apache License, Version 2.0
"""

from argus.inference.predictor import (
    ARGUSPredictor,
    PredictionResult,
    BatchPredictionResult,
)
from argus.inference.pipeline import (
    InferencePipeline,
    PipelineConfig,
    ClinicalPipeline,
)
from argus.inference.server import (
    ModelServer,
    create_prediction_endpoint,
    health_check,
)

__all__ = [
    # Predictor
    "ARGUSPredictor",
    "PredictionResult",
    "BatchPredictionResult",
    # Pipeline
    "InferencePipeline",
    "PipelineConfig",
    "ClinicalPipeline",
    # Server
    "ModelServer",
    "create_prediction_endpoint",
    "health_check",
]
