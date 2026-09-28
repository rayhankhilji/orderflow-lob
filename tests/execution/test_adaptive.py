"""Adaptive-execution variants: the re-planner must complete like TWAP."""

from __future__ import annotations

import numpy as np

import orderflow.execution  # noqa: F401
from orderflow.book.types import Side
from orderflow.eval.execution_bench import run_episode
from orderflow.execution.adaptive import AdaptiveAC
from orderflow.execution.types import ExecutionTask
from orderflow.registry import get_execution, list_execution
from orderflow.sim.regimes import get_regime

TASK = ExecutionTask(side=Side.SELL, total_qty=20_000, horizon=600.0, step=10.0)


class _FakeBook:
    """Minimal mid()-provider for the EWMA-vol unit test."""

    def __init__(self, mid: float) -> None:
        self._mid = mid

    def mid(self) -> float:
        return self._mid


def _episode(name: str, seed: int):
    flow_cls, params, seed_kwargs = get_regime("exec", flow="hawkes")
    return run_episode(
        get_execution(name)(), TASK, flow_cls, params, seed_kwargs, sim_seed=seed
    )


def test_variants_registered() -> None:
    assert {"ac_adaptive", "twap_capped"} <= set(list_execution())


def test_adaptive_ac_completes() -> None:
    c = _episode("ac_adaptive", seed=21)
    assert c.fill_rate >= 0.999


def test_capped_twap_completes() -> None:
    c = _episode("twap_capped", seed=22)
    assert c.fill_rate >= 0.999


def test_adaptive_sigma_tracks_vol() -> None:
    # alternating +/-2 tick mid moves must read as hot vol vs a flat book
    hot_algo = AdaptiveAC()
    hot_algo.reset(TASK, _FakeBook(100.0), np.random.default_rng(0))
    t = 0.0
    hot = 0.0
    for k in range(60):
        t += 10.0
        hot = hot_algo._sigma_now(t, _FakeBook(100.0 + (2.0 if k % 2 else -2.0)))

    calm_algo = AdaptiveAC()
    calm_algo.reset(TASK, _FakeBook(100.0), np.random.default_rng(0))
    t = 0.0
    calm = 0.0
    for _ in range(60):
        t += 10.0
        calm = calm_algo._sigma_now(t, _FakeBook(100.0))

    assert hot > 2 * calm
