"""
Optimizer Utilities.

Optimizer creation and configuration for ARGUS training.

Copyright 2024-2025 VTP Consortium
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

from typing import Any, Iterator

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer


def create_optimizer(
    model: nn.Module,
    optimizer_name: str = "adamw",
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    **kwargs: Any,
) -> Optimizer:
    """
    Create an optimizer for the model.

    Args:
        model: Model to optimize.
        optimizer_name: Optimizer name ('adam', 'adamw', 'sgd', 'lamb', 'adafactor').
            Default: 'adamw'
        lr: Learning rate.
            Default: 1e-4
        weight_decay: Weight decay (L2 regularization).
            Default: 0.01
        betas: Adam beta parameters.
            Default: (0.9, 0.999)
        eps: Adam epsilon for numerical stability.
            Default: 1e-8
        **kwargs: Additional optimizer-specific arguments.

    Returns:
        Configured optimizer.

    Example:
        >>> optimizer = create_optimizer(
        ...     model=model,
        ...     optimizer_name='adamw',
        ...     lr=1e-4,
        ...     weight_decay=0.01
        ... )
    """
    # Get parameter groups with weight decay handling
    param_groups = create_optimizer_groups(
        model=model,
        weight_decay=weight_decay,
        lr=lr,
    )

    optimizer_name = optimizer_name.lower()

    if optimizer_name == "adam":
        return torch.optim.Adam(
            param_groups,
            lr=lr,
            betas=betas,
            eps=eps,
        )

    elif optimizer_name == "adamw":
        return torch.optim.AdamW(
            param_groups,
            lr=lr,
            betas=betas,
            eps=eps,
        )

    elif optimizer_name == "sgd":
        momentum = kwargs.get("momentum", 0.9)
        nesterov = kwargs.get("nesterov", True)
        return torch.optim.SGD(
            param_groups,
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
        )

    elif optimizer_name == "lamb":
        # LAMB optimizer for large batch training
        try:
            from torch_optimizer import Lamb
            return Lamb(
                param_groups,
                lr=lr,
                betas=betas,
                eps=eps,
            )
        except ImportError:
            raise ImportError(
                "LAMB optimizer requires torch-optimizer: pip install torch-optimizer"
            )

    elif optimizer_name == "adafactor":
        # Adafactor for memory-efficient training
        try:
            from transformers import Adafactor
            return Adafactor(
                param_groups,
                lr=lr,
                scale_parameter=True,
                relative_step=False,
            )
        except ImportError:
            raise ImportError(
                "Adafactor requires transformers: pip install transformers"
            )

    elif optimizer_name == "radam":
        # RAdam for improved Adam convergence
        try:
            from torch.optim import RAdam
            return RAdam(
                param_groups,
                lr=lr,
                betas=betas,
                eps=eps,
            )
        except ImportError:
            raise ImportError("RAdam requires PyTorch >= 2.0")

    else:
        raise ValueError(
            f"Unknown optimizer: {optimizer_name}. "
            f"Available: adam, adamw, sgd, lamb, adafactor, radam"
        )


def create_optimizer_groups(
    model: nn.Module,
    weight_decay: float = 0.01,
    lr: float = 1e-4,
    no_decay_keywords: list[str] | None = None,
    layer_decay: float | None = None,
) -> list[dict[str, Any]]:
    """
    Create parameter groups with differentiated weight decay and learning rates.

    Implements best practices:
    - No weight decay on bias terms and LayerNorm parameters
    - Optional layer-wise learning rate decay (LLRD) for fine-tuning

    Args:
        model: Model to optimize.
        weight_decay: Weight decay for parameters that should be regularized.
            Default: 0.01
        lr: Base learning rate.
            Default: 1e-4
        no_decay_keywords: Keywords for parameters that shouldn't have weight decay.
            Default: ['bias', 'LayerNorm', 'layer_norm', 'ln_']
        layer_decay: Layer-wise learning rate decay factor.
            If None, all layers use the same learning rate.

    Returns:
        List of parameter group dictionaries.

    Example:
        >>> param_groups = create_optimizer_groups(
        ...     model=model,
        ...     weight_decay=0.01,
        ...     layer_decay=0.9  # Earlier layers get lower LR
        ... )
    """
    if no_decay_keywords is None:
        no_decay_keywords = [
            "bias",
            "LayerNorm",
            "layer_norm",
            "ln_",
            "norm",
            "embedding",
        ]

    # Separate parameters into decay and no-decay groups
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Check if parameter should have no weight decay
        if any(nd in name for nd in no_decay_keywords):
            no_decay_params.append((name, param))
        else:
            decay_params.append((name, param))

    # Apply layer-wise learning rate decay if specified
    if layer_decay is not None:
        param_groups = _create_llrd_groups(
            decay_params=decay_params,
            no_decay_params=no_decay_params,
            lr=lr,
            weight_decay=weight_decay,
            layer_decay=layer_decay,
        )
    else:
        param_groups = [
            {
                "params": [p for _, p in decay_params],
                "weight_decay": weight_decay,
                "lr": lr,
            },
            {
                "params": [p for _, p in no_decay_params],
                "weight_decay": 0.0,
                "lr": lr,
            },
        ]

    return param_groups


def _create_llrd_groups(
    decay_params: list[tuple[str, Tensor]],
    no_decay_params: list[tuple[str, Tensor]],
    lr: float,
    weight_decay: float,
    layer_decay: float,
) -> list[dict[str, Any]]:
    """
    Create parameter groups with layer-wise learning rate decay.

    Earlier (deeper in architecture) layers get lower learning rates.

    Args:
        decay_params: Parameters with weight decay.
        no_decay_params: Parameters without weight decay.
        lr: Base learning rate for the last layer.
        weight_decay: Weight decay value.
        layer_decay: Decay factor per layer.

    Returns:
        List of parameter group dictionaries.
    """
    # Identify layer indices from parameter names
    def get_layer_id(name: str) -> int:
        """Extract layer number from parameter name."""
        # Common patterns: layer.0., layers.5., encoder.layer.3.
        import re
        matches = re.findall(r'layer[s]?\.?(\d+)', name)
        if matches:
            return int(matches[-1])
        # Embedding and first layers get lowest LR
        if 'embedding' in name or 'input' in name:
            return 0
        # Output layers get highest LR
        if 'head' in name or 'output' in name or 'classifier' in name:
            return 999
        return 500  # Default: middle

    # Group parameters by layer
    layer_params: dict[int, dict[str, list]] = {}

    for name, param in decay_params:
        layer_id = get_layer_id(name)
        if layer_id not in layer_params:
            layer_params[layer_id] = {"decay": [], "no_decay": []}
        layer_params[layer_id]["decay"].append(param)

    for name, param in no_decay_params:
        layer_id = get_layer_id(name)
        if layer_id not in layer_params:
            layer_params[layer_id] = {"decay": [], "no_decay": []}
        layer_params[layer_id]["no_decay"].append(param)

    # Sort layers and compute LR per layer
    sorted_layers = sorted(layer_params.keys())
    max_layer = max(sorted_layers) if sorted_layers else 0

    param_groups = []
    for layer_id in sorted_layers:
        # LR scales with layer_decay^(max_layer - layer_id)
        # So later layers have higher LR
        layer_lr = lr * (layer_decay ** (max_layer - layer_id))

        if layer_params[layer_id]["decay"]:
            param_groups.append({
                "params": layer_params[layer_id]["decay"],
                "weight_decay": weight_decay,
                "lr": layer_lr,
            })

        if layer_params[layer_id]["no_decay"]:
            param_groups.append({
                "params": layer_params[layer_id]["no_decay"],
                "weight_decay": 0.0,
                "lr": layer_lr,
            })

    return param_groups


def count_parameters(model: nn.Module) -> dict[str, int]:
    """
    Count model parameters.

    Args:
        model: Model to count parameters for.

    Returns:
        Dictionary with parameter counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable

    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": non_trainable,
    }


def freeze_parameters(
    model: nn.Module,
    freeze_patterns: list[str] | None = None,
    unfreeze_patterns: list[str] | None = None,
) -> None:
    """
    Freeze or unfreeze model parameters by name patterns.

    Args:
        model: Model to modify.
        freeze_patterns: Patterns for parameters to freeze.
        unfreeze_patterns: Patterns for parameters to unfreeze.

    Example:
        >>> # Freeze encoder, train only classifier
        >>> freeze_parameters(model, freeze_patterns=['encoder'])

        >>> # Unfreeze last layers
        >>> freeze_parameters(model, unfreeze_patterns=['layer.5', 'head'])
    """
    for name, param in model.named_parameters():
        if freeze_patterns:
            if any(pattern in name for pattern in freeze_patterns):
                param.requires_grad = False

        if unfreeze_patterns:
            if any(pattern in name for pattern in unfreeze_patterns):
                param.requires_grad = True


def get_optimizer_state_summary(optimizer: Optimizer) -> dict[str, Any]:
    """
    Get summary of optimizer state.

    Args:
        optimizer: Optimizer to summarize.

    Returns:
        Dictionary with optimizer state summary.
    """
    state = optimizer.state_dict()

    summary = {
        "n_param_groups": len(state["param_groups"]),
        "param_groups": [],
    }

    for i, group in enumerate(state["param_groups"]):
        group_info = {
            "index": i,
            "lr": group.get("lr"),
            "weight_decay": group.get("weight_decay"),
            "n_params": len(group["params"]),
        }
        summary["param_groups"].append(group_info)

    return summary
