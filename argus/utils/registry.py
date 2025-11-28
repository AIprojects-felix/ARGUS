"""
Component Registry System.

Registry for model components with plugin support.

Copyright 2024-2025 VTP Consortium
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import logging
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Registry:
    """
    Generic registry for components.

    Provides a centralized system for registering and retrieving
    components (models, losses, optimizers, etc.) by name.

    Args:
        name: Registry name.

    Example:
        >>> model_registry = Registry("models")
        >>> @model_registry.register("my_model")
        ... class MyModel(nn.Module):
        ...     pass
        >>> model_cls = model_registry.get("my_model")
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._registry: dict[str, Any] = {}

    def register(
        self,
        name: str | None = None,
    ) -> Callable[[type[T]], type[T]]:
        """
        Register a component.

        Can be used as a decorator with or without arguments.

        Args:
            name: Optional name for the component. If None, uses class name.

        Returns:
            Decorator function.

        Example:
            >>> @registry.register("custom_name")
            ... class MyComponent:
            ...     pass

            >>> @registry.register()
            ... class AnotherComponent:  # Uses "AnotherComponent" as name
            ...     pass
        """
        def decorator(cls: type[T]) -> type[T]:
            key = name if name is not None else cls.__name__
            self._registry[key] = cls
            logger.debug(f"Registered {self.name}/{key}")
            return cls

        return decorator

    def register_module(self, name: str, module: Any) -> None:
        """
        Register a component directly (non-decorator).

        Args:
            name: Component name.
            module: Component class or function.

        Example:
            >>> registry.register_module("my_component", MyComponent)
        """
        self._registry[name] = module
        logger.debug(f"Registered {self.name}/{name}")

    def get(self, name: str) -> Any:
        """
        Get a registered component by name.

        Args:
            name: Component name.

        Returns:
            Registered component.

        Raises:
            KeyError: If component not found.

        Example:
            >>> model_cls = model_registry.get("my_model")
            >>> model = model_cls(**config)
        """
        if name not in self._registry:
            available = ", ".join(self._registry.keys())
            raise KeyError(
                f"'{name}' not found in {self.name} registry. "
                f"Available: {available}"
            )
        return self._registry[name]

    def build(self, name: str, **kwargs: Any) -> Any:
        """
        Build a component by name with arguments.

        Args:
            name: Component name.
            **kwargs: Arguments to pass to component constructor.

        Returns:
            Instantiated component.

        Example:
            >>> model = model_registry.build("my_model", hidden_dim=256)
        """
        cls = self.get(name)
        return cls(**kwargs)

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __len__(self) -> int:
        return len(self._registry)

    def keys(self) -> list[str]:
        """List registered component names."""
        return list(self._registry.keys())

    def items(self) -> list[tuple[str, Any]]:
        """List registered components as (name, class) pairs."""
        return list(self._registry.items())

    def __repr__(self) -> str:
        return f"Registry(name={self.name}, components={list(self._registry.keys())})"


# Global registries
MODEL_REGISTRY = Registry("models")
LOSS_REGISTRY = Registry("losses")
OPTIMIZER_REGISTRY = Registry("optimizers")
SCHEDULER_REGISTRY = Registry("schedulers")
ENCODER_REGISTRY = Registry("encoders")
FUSION_REGISTRY = Registry("fusion")
HEAD_REGISTRY = Registry("heads")


def register_defaults() -> None:
    """Register default components in all registries."""
    _register_default_models()
    _register_default_losses()
    _register_default_optimizers()
    _register_default_schedulers()
    _register_default_encoders()
    _register_default_fusion()
    _register_default_heads()


def _register_default_models() -> None:
    """Register default models."""
    try:
        from argus.models import ARGUS
        MODEL_REGISTRY.register_module("argus", ARGUS)
    except ImportError:
        pass


def _register_default_losses() -> None:
    """Register default loss functions."""
    try:
        from argus.models.losses import (
            BCEWithLogitsLoss,
            FocalLoss,
            WeightedBCELoss,
            MultiTaskLoss,
        )
        LOSS_REGISTRY.register_module("bce", BCEWithLogitsLoss)
        LOSS_REGISTRY.register_module("focal", FocalLoss)
        LOSS_REGISTRY.register_module("weighted_bce", WeightedBCELoss)
        LOSS_REGISTRY.register_module("multitask", MultiTaskLoss)
    except ImportError:
        pass


def _register_default_optimizers() -> None:
    """Register default optimizers."""
    import torch.optim as optim

    OPTIMIZER_REGISTRY.register_module("adam", optim.Adam)
    OPTIMIZER_REGISTRY.register_module("adamw", optim.AdamW)
    OPTIMIZER_REGISTRY.register_module("sgd", optim.SGD)

    try:
        OPTIMIZER_REGISTRY.register_module("radam", optim.RAdam)
    except AttributeError:
        pass


def _register_default_schedulers() -> None:
    """Register default learning rate schedulers."""
    try:
        from argus.training.schedulers import (
            CosineAnnealingWarmup,
            LinearWarmup,
            PolynomialDecay,
            OneCycleLR,
            NoamScheduler,
        )
        SCHEDULER_REGISTRY.register_module("cosine_warmup", CosineAnnealingWarmup)
        SCHEDULER_REGISTRY.register_module("linear_warmup", LinearWarmup)
        SCHEDULER_REGISTRY.register_module("polynomial", PolynomialDecay)
        SCHEDULER_REGISTRY.register_module("onecycle", OneCycleLR)
        SCHEDULER_REGISTRY.register_module("noam", NoamScheduler)
    except ImportError:
        pass


def _register_default_encoders() -> None:
    """Register default encoders."""
    try:
        from argus.models.encoders import StaticEncoder, TemporalEncoder
        ENCODER_REGISTRY.register_module("static", StaticEncoder)
        ENCODER_REGISTRY.register_module("temporal", TemporalEncoder)
    except ImportError:
        pass


def _register_default_fusion() -> None:
    """Register default fusion modules."""
    try:
        from argus.models.fusion import (
            ConcatFusion,
            CrossAttentionFusion,
            GatedFusion,
        )
        FUSION_REGISTRY.register_module("concat", ConcatFusion)
        FUSION_REGISTRY.register_module("cross_attention", CrossAttentionFusion)
        FUSION_REGISTRY.register_module("gated", GatedFusion)
    except ImportError:
        pass


def _register_default_heads() -> None:
    """Register default prediction heads."""
    try:
        from argus.models.heads import (
            MultiLabelClassificationHead,
            OrdinalClassificationHead,
        )
        HEAD_REGISTRY.register_module("multilabel", MultiLabelClassificationHead)
        HEAD_REGISTRY.register_module("ordinal", OrdinalClassificationHead)
    except ImportError:
        pass


def build_model(config: dict[str, Any]) -> Any:
    """
    Build a model from configuration.

    Args:
        config: Model configuration dictionary.
            Must contain 'name' key specifying model type.

    Returns:
        Instantiated model.

    Example:
        >>> config = {
        ...     'name': 'argus',
        ...     'd_model': 256,
        ...     'n_heads': 8,
        ... }
        >>> model = build_model(config)
    """
    name = config.pop("name", "argus")
    return MODEL_REGISTRY.build(name, **config)


def build_loss(config: dict[str, Any]) -> Any:
    """
    Build a loss function from configuration.

    Args:
        config: Loss configuration dictionary.

    Returns:
        Instantiated loss function.
    """
    name = config.pop("name", "bce")
    return LOSS_REGISTRY.build(name, **config)


def build_optimizer(model_params: Any, config: dict[str, Any]) -> Any:
    """
    Build an optimizer from configuration.

    Args:
        model_params: Model parameters to optimize.
        config: Optimizer configuration dictionary.

    Returns:
        Instantiated optimizer.
    """
    name = config.pop("name", "adamw")
    return OPTIMIZER_REGISTRY.build(name, params=model_params, **config)


def build_scheduler(optimizer: Any, config: dict[str, Any]) -> Any:
    """
    Build a learning rate scheduler from configuration.

    Args:
        optimizer: Optimizer instance.
        config: Scheduler configuration dictionary.

    Returns:
        Instantiated scheduler.
    """
    name = config.pop("name", "cosine_warmup")
    return SCHEDULER_REGISTRY.build(name, optimizer=optimizer, **config)


# Auto-register defaults on import
register_defaults()
