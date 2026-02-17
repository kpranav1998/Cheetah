from __future__ import annotations

from strategies.base import BaseStrategy

_REGISTRY: dict[str, type[BaseStrategy]] = {}


def register(cls: type[BaseStrategy]) -> type[BaseStrategy]:
    _REGISTRY[cls.name] = cls
    return cls


def get_strategy(name: str) -> BaseStrategy:
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown strategy: {name}. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]()


def list_strategies() -> list[str]:
    return list(_REGISTRY.keys())
