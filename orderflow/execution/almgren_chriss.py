"""Almgren-Chriss optimal execution (discrete-time, linear impact).

Model (classical AC, Almgren & Chriss 2001): trading parent size X over T in
N slices n_j at times t_j = j*tau. Mid evolves as

    S_k = S_0 + sigma * sum sqrt(tau) xi_j - gamma * sum n_j   (permanent)
    fill_k = S_k - eta * (n_k / tau)                           (temporary)

so E[cost] = gamma X^2 / 2 + eta~ * sum_j n_j^2 / tau,
Var[cost] = sigma^2 * tau * sum_j x_j^2  (x_j = inventory during interval j),
eta~ = eta - gamma * tau / 2. The mean-variance optimum for risk aversion
lambda is

    x_j = X * sinh(kappa (T - t_j)) / sinh(kappa T),
    kappa = arccosh(1 + lambda sigma^2 tau^2 / (2 eta~)) / tau.

Units: sigma in $/(share sqrt(s)), eta in $ s/share^2, gamma in $/share^2,
n_j in shares -> E[C] and Var[C] come out in dollars. ``calibrate_almgren_chriss``
in stats/impact.py estimates exactly these coefficients.

A negative gamma estimate (possible in our sim because the book mean-reverts
after a sweep) is clamped to zero for scheduling: trading as if impact
*reverted* would front-load on a phantom drift.
"""

from __future__ import annotations

import math

import numpy as np

from orderflow.book.book import LimitOrderBook
from orderflow.execution.types import ChildOrder, ExecState, ExecutionTask
from orderflow.registry import register_execution


def ac_kappa(sigma: float, eta: float, gamma: float, lam: float, tau: float) -> float:
    eta_tilde = max(eta - max(gamma, 0.0) * tau / 2.0, 1e-12)
    arg = 1.0 + lam * sigma * sigma * tau * tau / (2.0 * eta_tilde)
    return math.acosh(max(arg, 1.0)) / tau


def ac_inventory(
    total_qty: int, horizon: float, step: float, kappa: float
) -> np.ndarray:
    """x_j for j = 0..N on the decision grid (t_j = j*step, N = ceil(T/step))."""
    n = max(1, math.ceil(horizon / step))
    t = np.arange(n + 1) * step
    if kappa <= 1e-12:
        x = total_qty * (1.0 - t / t[-1])
    else:
        x = total_qty * np.sinh(kappa * (t[-1] - t)) / np.sinh(kappa * t[-1])
    x[-1] = 0.0
    return x


def ac_expected_cost(
    inventory: np.ndarray, step: float, sigma: float, eta: float, gamma: float
) -> float:
    """E[C] in dollars for an inventory trajectory (x_0 = X ... x_N = 0)."""
    g = max(gamma, 0.0)
    eta_tilde = max(eta - g * step / 2.0, 1e-12)
    x = np.asarray(inventory, dtype=float)
    n = np.diff(-x)  # shares executed per interval (positive)
    return float(g * x[0] ** 2 / 2.0 + eta_tilde * np.sum(n**2) / step)


def ac_cost_variance(inventory: np.ndarray, step: float, sigma: float) -> float:
    """Var[C] in dollars^2: each interval's held inventory sees sigma*sqrt(tau)."""
    x = np.asarray(inventory, dtype=float)
    return float(sigma * sigma * step * np.sum(x[1:] ** 2))


def ac_frontier(
    total_qty: int,
    horizon: float,
    step: float,
    sigma: float,
    eta: float,
    gamma: float,
    lambdas: np.ndarray | list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """(expected_cost, std_cost) over a grid of risk aversions."""
    e, s = [], []
    for lam in lambdas:
        x = ac_inventory(
            total_qty, horizon, step, ac_kappa(sigma, eta, gamma, float(lam), step)
        )
        e.append(ac_expected_cost(x, step, sigma, eta, gamma))
        s.append(math.sqrt(ac_cost_variance(x, step, sigma)))
    return np.asarray(e), np.asarray(s)


# sim-calibrated defaults from scripts/calibrate.py on the "exec" regime
DEFAULT_SIGMA = 0.0174  # $/(share sqrt(s))
DEFAULT_ETA = 5.2e-4  # $ s / share^2
DEFAULT_GAMMA = 0.0  # permanent impact is ~unmeasurable in our sim


@register_execution("ac")
class AlmgrenChriss:
    """Closed-form AC schedule; aggressive (market) slices by default."""

    name = "ac"

    def __init__(
        self,
        sigma: float = DEFAULT_SIGMA,
        eta: float = DEFAULT_ETA,
        gamma: float = DEFAULT_GAMMA,
        lam: float = 1e-5,
        aggressive: bool = True,
    ) -> None:
        self.sigma, self.eta, self.gamma, self.lam = sigma, eta, gamma, lam
        self.aggressive = aggressive

    @classmethod
    def from_params(cls, params, lam: float = 1e-5, aggressive: bool = True):
        return cls(
            sigma=params.sigma,
            eta=params.eta,
            gamma=params.gamma,
            lam=lam,
            aggressive=aggressive,
        )

    def reset(
        self, task: ExecutionTask, book: LimitOrderBook, rng: np.random.Generator
    ) -> None:
        self.task = task
        self.n_inner = 0
        while (self.n_inner + 1) * task.step < task.horizon - 1e-9:
            self.n_inner += 1
        kappa = ac_kappa(self.sigma, self.eta, self.gamma, self.lam, task.step)
        # inventory after each interval j = 0..n_inner+1
        x = ac_inventory(task.total_qty, task.horizon, task.step, kappa)
        self.cum_target = np.maximum.accumulate(task.total_qty - x)  # c_j = X - x_j

    def _target(self, elapsed: float) -> int:
        j = min(round(elapsed / self.task.step), self.n_inner)
        idx = min(j, len(self.cum_target) - 2)  # keep last point for the deadline
        return round(float(self.cum_target[idx]))

    def decide(
        self, t: float, book: LimitOrderBook, state: ExecState
    ) -> list[ChildOrder]:
        qty = min(self._target(state.elapsed) - state.filled_qty, state.remaining_qty)
        if qty <= 0:
            return []
        return [
            ChildOrder(
                side=self.task.side,
                qty=qty,
                kind="market" if self.aggressive else "limit",
                price_offset_ticks=0,
            )
        ]
