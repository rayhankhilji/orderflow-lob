"""Plugin registries used by agent branches to register competing implementations."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

_EXECUTION: dict[str, type] = {}
_PREDICTOR: dict[str, type] = {}
_FLOW: dict[str, type] = {}
_ADVERSARY: dict[str, type] = {}


def _make_register(registry: dict[str, type]) -> Callable[[str], Callable[[T], T]]:
    def register(name: str) -> Callable[[T], T]:
        def decorator(cls: T) -> T:
            if name in registry:
                raise KeyError(f"{name!r} already registered")
            registry[name] = cls  # type: ignore[assignment]
            return cls

        return decorator

    return register


def _make_get(registry: dict[str, type]) -> Callable[[str], type]:
    def get(name: str) -> type:
        try:
            return registry[name]
        except KeyError:
            raise KeyError(f"{name!r} not registered; have: {sorted(registry)}") from None

    return get


register_execution = _make_register(_EXECUTION)
register_predictor = _make_register(_PREDICTOR)
register_flow = _make_register(_FLOW)
register_adversary = _make_register(_ADVERSARY)

get_execution = _make_get(_EXECUTION)
get_predictor = _make_get(_PREDICTOR)
get_flow = _make_get(_FLOW)
get_adversary = _make_get(_ADVERSARY)


def list_execution() -> list[str]:
    return sorted(_EXECUTION)


def list_predictor() -> list[str]:
    return sorted(_PREDICTOR)


def list_flow() -> list[str]:
    return sorted(_FLOW)


def list_adversary() -> list[str]:
    return sorted(_ADVERSARY)
