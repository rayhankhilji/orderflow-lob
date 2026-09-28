"""LearnedPolicy: an Almgren-Chriss spine tilted by live model signal.

The base trajectory is the closed-form AC inventory path x_j. At each decision
the live 9-dim state vector is fed to a predictor's ``mid_move`` head and the
schedule is tilted:

    x_j' = x_j * (1 + kappa_tilt * s_j),   s_j = P(favourable) - P(adverse)

then re-clipped to a monotone non-increasing inventory path. For a SELL parent
adverse = predicted mid-down; for a BUY parent adverse = mid-up. When the
predictor thinks the near-term drift is against us the remaining inventory
shrinks faster (front-load); favourable drift lets the schedule relax.

``predictor`` is anything with ``signal(state_vec) -> float``; ``TorchSignal``
adapts a fitted LogisticBaseline/MLPBaseline (they only need ``batch["state"]``).
With no predictor the signal is 0 and the policy reduces to plain AC.
"""

from __future__ import annotations

import numpy as np

from orderflow.book.book import LimitOrderBook
from orderflow.book.types import Side
from orderflow.execution.almgren_chriss import (
    DEFAULT_ETA,
    DEFAULT_GAMMA,
    DEFAULT_SIGMA,
    ac_inventory,
    ac_kappa,
)
from orderflow.execution.live import LiveFeatures
from orderflow.execution.types import ChildOrder, ExecState, ExecutionTask
from orderflow.registry import register_execution


class TorchSignal:
    """Signal adapter for fitted state-head baselines."""

    def __init__(self, predictor, side: Side) -> None:
        self.predictor = predictor
        self.side = side

    def signal(self, state_vec: np.ndarray) -> float:
        import torch

        x = torch.tensor(state_vec, dtype=torch.float32).reshape(1, 1, -1)
        out = self.predictor.predict({"state": x})
        probs = np.asarray(out.get("mid_move")).reshape(-1)
        if len(probs) < 3:
            return 0.0
        p_down, p_up = float(probs[0]), float(probs[2])
        if self.side is Side.SELL:
            return p_up - p_down  # favourable = mid rises while we still hold
        return p_down - p_up


@register_execution("learned")
class LearnedPolicy:
    name = "learned"

    def __init__(
        self,
        predictor=None,
        kappa_tilt: float = 1.0,
        sigma: float = DEFAULT_SIGMA,
        eta: float = DEFAULT_ETA,
        gamma: float = DEFAULT_GAMMA,
        lam: float = 1e-5,
    ) -> None:
        self.predictor = predictor
        self.kappa_tilt = kappa_tilt
        self.sigma, self.eta, self.gamma, self.lam = sigma, eta, gamma, lam
        self.live = LiveFeatures()

    def reset(
        self, task: ExecutionTask, book: LimitOrderBook, rng: np.random.Generator
    ) -> None:
        self.task = task
        self.live = LiveFeatures(tick_size=book.tick_size)
        self.n_inner = 0
        while (self.n_inner + 1) * task.step < task.horizon - 1e-9:
            self.n_inner += 1
        kappa = ac_kappa(self.sigma, self.eta, self.gamma, self.lam, task.step)
        self.base_x = ac_inventory(task.total_qty, task.horizon, task.step, kappa)
        if self.predictor is not None and not hasattr(self.predictor, "signal"):
            self.predictor = TorchSignal(self.predictor, task.side)

    def decide(
        self, t: float, book: LimitOrderBook, state: ExecState
    ) -> list[ChildOrder]:
        self.live.update(book, t)
        j = min(round(state.elapsed / self.task.step), self.n_inner)
        x_j = float(self.base_x[min(j, len(self.base_x) - 2)])
        s = 0.0
        if self.predictor is not None:
            time_frac = (state.elapsed / self.task.horizon) if self.task.horizon else 0.0
            s = float(np.clip(self.predictor.signal(self.live.state(time_frac)), -1, 1))
        # tilt the remaining inventory, then derive this step's cumulative target
        x_tilted = max(0.0, x_j * (1.0 + self.kappa_tilt * s))
        target = self.task.total_qty - x_tilted
        qty = min(round(target) - state.filled_qty, state.remaining_qty)
        if qty <= 0:
            return []
        return [ChildOrder(side=self.task.side, qty=qty, kind="market")]
