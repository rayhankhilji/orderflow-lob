"""Generate every figure in docs/figures/ — deterministic, short sims only.

    python scripts/make_figures.py

Produces: architecture.png, stylized_facts.png, hawkes_intensity.png,
ofi_impact.png, ac_frontier.png, exec_costs.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import orderflow.execution  # noqa: F401
from orderflow.book.book import LimitOrderBook
from orderflow.sim.flow.hawkes import HawkesParams, simulate_hawkes
from orderflow.sim.regimes import get_regime
from orderflow.sim.simulator import Simulator, seed_book
from orderflow.stats.ofi import compute_ofi_series, ofi_regression

OUT = Path("docs/figures")
FIG = {"figure.facecolor": "#0d1117", "axes.facecolor": "#0d1117",
       "axes.edgecolor": "#30363d", "axes.labelcolor": "#8b949e",
       "text.color": "#e6edf3", "xtick.color": "#8b949e", "ytick.color": "#8b949e",
       "grid.color": "#21262d", "font.size": 9}
plt.rcParams.update(FIG)
BID, ASK, ACC = "#2ea043", "#f85149", "#58a6ff"


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    print("wrote", OUT / name)


def fig_stylized(rng):
    cls, params, skw = get_regime("normal", flow="hawkes")
    book = LimitOrderBook()
    seed_book(book, rng=rng, **skw)
    res = Simulator(book, cls(params), seed=0).run(900.0, snapshot_every=1.0)
    mids = np.array([
        (s.bid_prices[0] + s.ask_prices[0]) / 2
        if len(s.bid_prices) and len(s.ask_prices) else np.nan
        for s in res.snapshots
    ])
    spreads = np.array([
        s.ask_prices[0] - s.bid_prices[0]
        if len(s.bid_prices) and len(s.ask_prices) else np.nan
        for s in res.snapshots
    ])
    dm = np.diff(mids)
    # only intervals where the book was healthy at both ends — a collapsed
    # side makes "mid" a bookkeeping artifact, not a tradeable price
    ok = np.isfinite(dm) & np.isfinite(spreads[:-1]) & np.isfinite(spreads[1:]) \
        & (spreads[:-1] <= 8) & (spreads[1:] <= 8)
    r = dm[ok] / mids[:-1][ok] * 1e4  # bps
    from scipy.stats import kurtosis

    # display window: rare book-collapse jumps reach ±1e4 bps; they are real
    # sim events (counted in the title) but would render the body invisible
    tail = np.abs(r) > 400
    rd = r[~tail]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3))
    ax[0].hist(rd, bins=60, density=True, color=ACC, alpha=0.8)
    xs = np.linspace(rd.min(), rd.max(), 200)
    ax[0].plot(xs, np.exp(-xs**2 / (2 * rd.std()**2)) / (rd.std() * np.sqrt(2 * np.pi)),
               color="#e6edf3", lw=1, ls="--", label="normal fit")
    ax[0].set_yscale("log")
    ax[0].set_title(
        f"mid returns, |r|≤400bps ({tail.sum()} collapse events excluded)\n"
        f"excess kurtosis {kurtosis(r):.0f}"
    )
    ax[0].legend()

    def acf(x, lags=40):
        x = x - x.mean()
        return np.array([1.0] + [np.corrcoef(x[:-k], x[k:])[0, 1] for k in range(1, lags)])

    ax[1].bar(range(40), acf(np.abs(r)), color=ACC)
    ax[1].set_title("|r| autocorrelation — vol clustering")
    ax[1].set_xlabel("lag (s)")
    ax[2].bar(range(40), acf(np.sign(r[np.abs(r) > 0])), color=ASK)
    ax[2].set_title("sign(r) autocorrelation")
    ax[2].set_xlabel("lag (s)")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "stylized_facts.png")


def fig_hawkes(rng):
    _, p, _ = get_regime("normal", flow="hawkes")
    truth = HawkesParams(mu=np.array(p["mu"]), alpha=np.array(p["alpha"]), beta=float(p["beta"]))
    tapes = simulate_hawkes(truth, T=120.0, rng=rng)
    ti = tapes[0]  # buy_limit intensity trace

    beta = truth.beta[0, 0]
    ts = np.linspace(0, 120, 2400)
    # replay the tape: R_j(t) = sum over type-j events of alpha_0j e^{-b(t-t_l)}
    fig, ax = plt.subplots(2, 1, figsize=(9, 4.5), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1]})
    R = np.zeros(6)
    prev = 0.0
    ptr = np.zeros(6, dtype=int)
    lam = np.empty(len(ts))
    for k, t in enumerate(ts):
        R *= np.exp(-beta * (t - prev))
        for j in range(6):
            tj = tapes[j]
            while ptr[j] < len(tj) and tj[ptr[j]] <= t:
                R[j] += truth.alpha[0, j]
                ptr[j] += 1
        lam[k] = truth.mu[0] + R.sum()
        prev = t
    ax[0].plot(ts, lam, color=ACC, lw=1)
    ax[0].set_ylabel(r"$\lambda_0(t)$")
    ax[0].set_title("Hawkes intensity, buy-limit dim — self-exciting bursts")
    ax[0].grid(alpha=0.3)
    ax[1].vlines(ti, 0, 1, color=ASK, lw=0.6)
    ax[1].set_yticks([])
    ax[1].set_xlabel("t (s)")
    ax[1].set_title("buy_limit events")
    save(fig, "hawkes_intensity.png")


def fig_ofi(rng):
    cls, params, skw = get_regime("normal", flow="hawkes")
    book = LimitOrderBook()
    seed_book(book, rng=rng, **skw)
    res = Simulator(book, cls(params), seed=1).run(600.0, snapshot_every=1.0)
    ofi = compute_ofi_series(res.snapshots)
    mids = np.array([
        (s.bid_prices[0] + s.ask_prices[0]) / 2
        if len(s.bid_prices) and len(s.ask_prices) else np.nan
        for s in res.snapshots
    ])
    dm = np.diff(mids, prepend=mids[0])
    w = 10
    n = (len(ofi) // w) * w
    x = ofi[:n].reshape(-1, w).sum(1)
    y = np.nansum(dm[:n].reshape(-1, w), axis=1)
    m = np.isfinite(x) & np.isfinite(y)
    fit = ofi_regression(ofi, dm, window=w)
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.scatter(x[m], y[m], s=6, alpha=0.5, color=ACC)
    xs = np.linspace(x[m].min(), x[m].max(), 50)
    ax.plot(xs, fit.intercept + fit.beta * xs, color=ASK, lw=1.4,
            label=f"OLS β={fit.beta:.3f}, $r^2$={fit.r2:.2f}")
    ax.set_xlabel("OFI (10 s)")
    ax.set_ylabel("Δ mid (ticks)")
    ax.set_title("OFI → mid move (Cont–Kukanov–Stoikov)")
    ax.legend()
    ax.grid(alpha=0.3)
    save(fig, "ofi_impact.png")


def fig_frontier():
    from orderflow.execution.almgren_chriss import (
        DEFAULT_ETA,
        DEFAULT_GAMMA,
        DEFAULT_SIGMA,
        ac_frontier,
        ac_inventory,
        ac_kappa,
    )
    lams = np.logspace(-7, -3, 25)
    e, s = ac_frontier(100_000, 1800.0, 10.0, DEFAULT_SIGMA, DEFAULT_ETA, DEFAULT_GAMMA, lams)
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
    ax[0].plot(s / 1e3, e / 1e3, color=ACC, marker="o", ms=3)
    ax[0].set_xlabel("sd(C)  [$1k]")
    ax[0].set_ylabel("E[C]  [$1k]")
    ax[0].set_title("Almgren–Chriss efficient frontier")
    tt = np.arange(181) * 10.0
    for lam, c in zip([1e-7, 1e-5, 1e-4, 1e-3], ["#8b949e", ACC, "#d29922", ASK]):
        x = ac_inventory(100_000, 1800.0, 10.0,
                         ac_kappa(DEFAULT_SIGMA, DEFAULT_ETA, DEFAULT_GAMMA, lam, 10.0))
        ax[1].plot(tt / 60, x / 1e3, color=c, lw=1.4, label=f"λ={lam:.0e}")
    ax[1].set_xlabel("t (min)")
    ax[1].set_ylabel("inventory (1k sh)")
    ax[1].set_title("optimal trajectories")
    ax[1].legend()
    for a in ax:
        a.grid(alpha=0.3)
    save(fig, "ac_frontier.png")


def fig_exec_costs():
    src = Path("research/execution/results.json")
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    if src.exists():
        rows = json.loads(src.read_text())["study"]["algos"]
    else:
        rows = json.loads(Path("research/benchmark/results.json").read_text())["regimes"]["exec"]
    names = [k for k in rows if "@" not in k]
    means = [rows[k]["mean_is_bps"] for k in names]
    stds = [rows[k]["std_is_bps"] for k in names]
    cvars = [rows[k]["cvar95_is_bps"] for k in names]
    x = np.arange(len(names))
    ax.bar(x - 0.2, means, width=0.4, color=ACC, label="mean IS")
    ax.bar(x + 0.2, cvars, width=0.4, color=ASK, label="CVaR95")
    ax.errorbar(x - 0.2, means, yerr=stds, fmt="none", color="#e6edf3", capsize=3, lw=1)
    ax.set_xticks(x, names, rotation=20)
    ax.set_ylabel("IS (bps)")
    ax.axhline(0, color="#30363d", lw=0.8)
    ax.set_title("execution cost by algo — exec task ($10M / 30 min)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    save(fig, "exec_costs.png")


def fig_architecture():
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.axis("off")

    def box(x, y, w, h, label, sub="", color="#161b22"):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=color, ec="#30363d", lw=1.2,
                                   joinstyle="round"))
        ax.text(x + w / 2, y + h * 0.62 if sub else y + h / 2, label,
                ha="center", va="center", fontsize=10, weight="bold", color="#e6edf3")
        if sub:
            ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center",
                    fontsize=7.5, color="#8b949e")

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops={"arrowstyle": "-|>", "color": "#8b949e", "lw": 1.2})

    box(0.02, 0.62, 0.20, 0.30, "Flow models", "ZI · Hawkes · queue-reactive")
    box(0.02, 0.08, 0.20, 0.30, "Participants", "exec adapter · adversaries\n"
        "spoof · ignite · withdraw · frontrun · pennyjump")
    box(0.30, 0.36, 0.18, 0.26, "LimitOrderBook", "L1/L2/L3 · price-time FIFO\n"
        "hidden + iceberg · event tape")
    box(0.56, 0.62, 0.19, 0.30, "Stats layer", "Hawkes MLE + GOF\nOFI · Kyle λ · vol")
    box(0.56, 0.08, 0.19, 0.30, "Models", "transformer · MLP\nlogistic · markov")
    box(0.80, 0.62, 0.18, 0.30, "Execution", "TWAP · VWAP · AC\nadaptive · learned · PPO")
    box(0.80, 0.08, 0.18, 0.30, "Evaluation", "bench + CRN · selection\nLEADERBOARD · survivors")
    arrow(0.22, 0.77, 0.30, 0.55)
    arrow(0.22, 0.24, 0.30, 0.42)
    arrow(0.48, 0.49, 0.56, 0.72)
    arrow(0.48, 0.49, 0.56, 0.28)
    arrow(0.75, 0.77, 0.80, 0.77)
    arrow(0.75, 0.23, 0.80, 0.23)
    arrow(0.89, 0.62, 0.89, 0.38)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("OrderFlow architecture", color="#e6edf3")
    save(fig, "architecture.png")


def main() -> None:
    rng = np.random.default_rng(0)
    fig_architecture()
    fig_stylized(rng)
    fig_hawkes(rng)
    fig_ofi(rng)
    fig_frontier()
    fig_exec_costs()


if __name__ == "__main__":
    main()
