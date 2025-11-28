"""
Model Serving Utilities.

HTTP server and API endpoints for ARGUS model deployment.

Copyright 2024-2025 VTP Consortium
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    """
    Configuration for model server.

    Attributes:
        model_path: Path to model checkpoint.
        host: Server host address.
        port: Server port.
        workers: Number of worker processes.
        max_batch_size: Maximum batch size for inference.
        timeout: Request timeout in seconds.
        enable_cors: Whether to enable CORS.
        api_key: Optional API key for authentication.
    """
    model_path: str
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    max_batch_size: int = 32
    timeout: float = 30.0
    enable_cors: bool = True
    api_key: str | None = None


@dataclass
class RequestMetrics:
    """
    Request metrics for monitoring.

    Attributes:
        total_requests: Total number of requests.
        successful_requests: Number of successful requests.
        failed_requests: Number of failed requests.
        total_inference_time: Total inference time in seconds.
        avg_latency: Average request latency.
    """
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_inference_time: float = 0.0
    avg_latency: float = 0.0

    def record_request(self, success: bool, latency: float) -> None:
        """Record a request."""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        self.total_inference_time += latency
        self.avg_latency = self.total_inference_time / self.total_requests

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": (
                self.successful_requests / self.total_requests
                if self.total_requests > 0
                else 0.0
            ),
            "avg_latency_ms": self.avg_latency * 1000,
            "total_inference_time_s": self.total_inference_time,
        }


class ModelServer:
    """
    HTTP server for ARGUS model serving.

    Provides REST API endpoints for model inference with support for
    batching, authentication, and monitoring.

    Args:
        config: Server configuration.

    Example:
        >>> config = ServerConfig(
        ...     model_path='model.pt',
        ...     port=8000
        ... )
        >>> server = ModelServer(config)
        >>> server.start()
    """

    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self.predictor = None
        self.metrics = RequestMetrics()
        self._started = False

    def initialize(self) -> None:
        """Initialize model predictor."""
        from argus.inference.predictor import ARGUSPredictor

        logger.info(f"Loading model from {self.config.model_path}")
        self.predictor = ARGUSPredictor.from_checkpoint(
            checkpoint_path=self.config.model_path,
            device="auto",
            batch_size=self.config.max_batch_size,
        )
        logger.info("Model loaded successfully")

    def create_app(self) -> Any:
        """
        Create ASGI application.

        Returns:
            FastAPI application instance.
        """
        try:
            from fastapi import FastAPI, HTTPException, Depends, Header
            from fastapi.middleware.cors import CORSMiddleware
            from pydantic import BaseModel
        except ImportError:
            raise ImportError(
                "FastAPI is required for server functionality. "
                "Install with: pip install fastapi uvicorn"
            )

        app = FastAPI(
            title="ARGUS Prediction API",
            description="AI-based Routine Genomic Understanding System",
            version="1.0.0",
        )

        # CORS middleware
        if self.config.enable_cors:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

        # Request models
        class PredictionRequest(BaseModel):
            static_features: list[float]
            temporal_features: list[list[float]]
            temporal_mask: list[int] | None = None
            patient_id: str | None = None

        class BatchPredictionRequest(BaseModel):
            samples: list[PredictionRequest]

        class PredictionResponse(BaseModel):
            patient_id: str
            predictions: dict[str, float]
            predicted_classes: dict[str, int]
            processing_time_ms: float

        # Authentication dependency
        async def verify_api_key(x_api_key: str = Header(None)) -> None:
            if self.config.api_key is not None:
                if x_api_key != self.config.api_key:
                    raise HTTPException(status_code=401, detail="Invalid API key")

        # Initialize on startup
        @app.on_event("startup")
        async def startup_event() -> None:
            self.initialize()

        # Health check endpoint
        @app.get("/health")
        async def health_check_endpoint() -> dict[str, Any]:
            return health_check(self.predictor)

        # Metrics endpoint
        @app.get("/metrics")
        async def metrics_endpoint() -> dict[str, Any]:
            return self.metrics.to_dict()

        # Single prediction endpoint
        @app.post("/predict", response_model=PredictionResponse)
        async def predict_endpoint(
            request: PredictionRequest,
            _: None = Depends(verify_api_key),
        ) -> PredictionResponse:
            start_time = time.time()

            try:
                # Convert to numpy arrays
                static = np.array(request.static_features, dtype=np.float32)
                temporal = np.array(request.temporal_features, dtype=np.float32)
                mask = (
                    np.array(request.temporal_mask, dtype=np.int32)
                    if request.temporal_mask
                    else None
                )

                # Run prediction
                result = self.predictor.predict_single(
                    static_features=static,
                    temporal_features=temporal,
                    temporal_mask=mask,
                    patient_id=request.patient_id or "unknown",
                )

                latency = time.time() - start_time
                self.metrics.record_request(success=True, latency=latency)

                return PredictionResponse(
                    patient_id=result.patient_id,
                    predictions={
                        name: float(result.predictions[i])
                        for i, name in enumerate(result.target_names)
                    },
                    predicted_classes={
                        name: int(result.predicted_classes[i])
                        for i, name in enumerate(result.target_names)
                    },
                    processing_time_ms=latency * 1000,
                )

            except Exception as e:
                latency = time.time() - start_time
                self.metrics.record_request(success=False, latency=latency)
                logger.error(f"Prediction error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # Batch prediction endpoint
        @app.post("/predict/batch")
        async def predict_batch_endpoint(
            request: BatchPredictionRequest,
            _: None = Depends(verify_api_key),
        ) -> dict[str, Any]:
            start_time = time.time()

            try:
                # Convert all samples
                static_batch = np.array([
                    s.static_features for s in request.samples
                ], dtype=np.float32)
                temporal_batch = np.array([
                    s.temporal_features for s in request.samples
                ], dtype=np.float32)

                patient_ids = [
                    s.patient_id or f"sample_{i}"
                    for i, s in enumerate(request.samples)
                ]

                # Run batch prediction
                batch_result = self.predictor.predict_batch(
                    static_features=static_batch,
                    temporal_features=temporal_batch,
                    patient_ids=patient_ids,
                )

                latency = time.time() - start_time
                self.metrics.record_request(success=True, latency=latency)

                return {
                    "results": [r.to_dict() for r in batch_result.results],
                    "summary": batch_result.summary,
                    "processing_time_ms": latency * 1000,
                }

            except Exception as e:
                latency = time.time() - start_time
                self.metrics.record_request(success=False, latency=latency)
                logger.error(f"Batch prediction error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # Model info endpoint
        @app.get("/info")
        async def info_endpoint() -> dict[str, Any]:
            return self.predictor.get_model_info()

        return app

    def start(self) -> None:
        """Start the server."""
        try:
            import uvicorn
        except ImportError:
            raise ImportError(
                "Uvicorn is required for server functionality. "
                "Install with: pip install uvicorn"
            )

        app = self.create_app()
        uvicorn.run(
            app,
            host=self.config.host,
            port=self.config.port,
            workers=self.config.workers,
        )


def create_prediction_endpoint(
    model_path: str,
    device: str = "auto",
) -> Callable:
    """
    Create a standalone prediction function for serverless deployment.

    Args:
        model_path: Path to model checkpoint.
        device: Device for inference.

    Returns:
        Prediction function that can be used as a serverless handler.

    Example:
        >>> handler = create_prediction_endpoint('model.pt')
        >>> result = handler({
        ...     'static_features': [...],
        ...     'temporal_features': [[...]],
        ... })
    """
    from argus.inference.predictor import ARGUSPredictor

    # Lazy load predictor
    _predictor = None

    def get_predictor() -> ARGUSPredictor:
        nonlocal _predictor
        if _predictor is None:
            _predictor = ARGUSPredictor.from_checkpoint(
                checkpoint_path=model_path,
                device=device,
            )
        return _predictor

    def predict_handler(event: dict[str, Any]) -> dict[str, Any]:
        """
        Serverless prediction handler.

        Args:
            event: Request event with input data.

        Returns:
            Prediction results.
        """
        try:
            predictor = get_predictor()

            # Parse input
            static = np.array(event["static_features"], dtype=np.float32)
            temporal = np.array(event["temporal_features"], dtype=np.float32)
            mask = (
                np.array(event.get("temporal_mask"), dtype=np.int32)
                if event.get("temporal_mask")
                else None
            )

            # Run prediction
            result = predictor.predict_single(
                static_features=static,
                temporal_features=temporal,
                temporal_mask=mask,
                patient_id=event.get("patient_id", "unknown"),
            )

            return {
                "statusCode": 200,
                "body": json.dumps(result.to_dict()),
            }

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                "statusCode": 500,
                "body": json.dumps({"error": str(e)}),
            }

    return predict_handler


def health_check(predictor: Any | None = None) -> dict[str, Any]:
    """
    Perform health check on model server.

    Args:
        predictor: Optional predictor instance to check.

    Returns:
        Health check results.

    Example:
        >>> status = health_check(predictor)
        >>> print(status['status'])  # 'healthy' or 'unhealthy'
    """
    import sys
    import platform

    status = {
        "status": "healthy",
        "timestamp": time.time(),
        "python_version": sys.version,
        "platform": platform.platform(),
    }

    # Check predictor
    if predictor is not None:
        try:
            # Try a simple inference
            n_static = 63
            n_temporal = 117
            seq_len = 10

            test_static = np.zeros(n_static, dtype=np.float32)
            test_temporal = np.zeros((seq_len, n_temporal), dtype=np.float32)

            result = predictor.predict_single(
                static_features=test_static,
                temporal_features=test_temporal,
                patient_id="health_check",
            )

            status["model_status"] = "ready"
            status["model_device"] = str(predictor.device)
            status["n_targets"] = len(result.target_names)

        except Exception as e:
            status["status"] = "unhealthy"
            status["model_status"] = "error"
            status["error"] = str(e)
    else:
        status["model_status"] = "not_loaded"

    # Check GPU availability
    try:
        import torch
        status["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            status["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception:
        status["cuda_available"] = False

    return status


class GRPCModelServer:
    """
    gRPC server for high-performance model serving.

    Provides gRPC endpoints for model inference with support for
    streaming and bidirectional communication.

    Note: Requires grpcio and grpcio-tools packages.
    """

    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self.predictor = None

    def initialize(self) -> None:
        """Initialize model predictor."""
        from argus.inference.predictor import ARGUSPredictor

        self.predictor = ARGUSPredictor.from_checkpoint(
            checkpoint_path=self.config.model_path,
            device="auto",
        )

    def start(self) -> None:
        """Start gRPC server."""
        try:
            import grpc
            from concurrent import futures
        except ImportError:
            raise ImportError(
                "gRPC is required. Install with: pip install grpcio grpcio-tools"
            )

        # Note: Actual gRPC implementation would require protobuf definitions
        # This is a placeholder showing the structure
        logger.info(
            f"gRPC server would start on {self.config.host}:{self.config.port}"
        )
        logger.warning("gRPC implementation requires protobuf definitions")


def create_docker_deployment(
    model_path: str,
    output_dir: str | Path = "deployment",
    port: int = 8000,
) -> None:
    """
    Create Docker deployment files for model serving.

    Args:
        model_path: Path to model checkpoint.
        output_dir: Output directory for deployment files.
        port: Server port.

    Creates:
        - Dockerfile
        - docker-compose.yml
        - requirements.txt
        - serve.py
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dockerfile
    dockerfile = f'''FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE {port}

# Run server
CMD ["python", "serve.py"]
'''

    # docker-compose.yml
    compose = f'''version: '3.8'

services:
  argus-server:
    build: .
    ports:
      - "{port}:{port}"
    environment:
      - MODEL_PATH=/app/model.pt
      - HOST=0.0.0.0
      - PORT={port}
    volumes:
      - ./model.pt:/app/model.pt:ro
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:{port}/health"]
      interval: 30s
      timeout: 10s
      retries: 3
'''

    # requirements.txt
    requirements = '''torch>=2.0.0
numpy>=1.24.0
fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0.0
'''

    # serve.py
    serve_script = f'''#!/usr/bin/env python3
"""ARGUS Model Server."""

import os
from argus.inference.server import ModelServer, ServerConfig

if __name__ == "__main__":
    config = ServerConfig(
        model_path=os.environ.get("MODEL_PATH", "model.pt"),
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", {port})),
    )
    server = ModelServer(config)
    server.start()
'''

    # Write files
    (output_dir / "Dockerfile").write_text(dockerfile)
    (output_dir / "docker-compose.yml").write_text(compose)
    (output_dir / "requirements.txt").write_text(requirements)
    (output_dir / "serve.py").write_text(serve_script)

    logger.info(f"Docker deployment files created in {output_dir}")
