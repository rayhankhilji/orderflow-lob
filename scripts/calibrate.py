"""Calibrate model parameters from simulations -> artifacts/params/*.json.

Outputs:
  hawkes.json       - MLE fit of the 6-type Hawkes to a normal-regime run
  garch.json        - GARCH(1,1) on resampled mid diffs
  ac.json           - (sigma, eta, gamma) from controlled metaorder experiments
  vwap_profile.npy  - cumulative volume profile for the VWAP algo

Usage: python scripts/calibrate.py --horizon 1800 --runs 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from orderflow.book.book import LimitOrderBook
from orderflow.execution.vwap import VolumeProfile
from orderflow.sim.regimes import get_regime
from orderflow.sim.simulator import EVENT_TYPES, Simulator, seed_book
from orderflow.stats.hawkes import fit_hawkes_exp
from orderflow.stats.impact import calibrate_almgren_chriss
from orderflow.stats.volatility import GARCH11


def run_plain(flow: str, regime: str, seed: int, horizon: float, tick: float):
    flow_cls, params, seed_kwargs = get_regime(regime, flow=flow)
    rng = np.random.default_rng(seed)
    book = LimitOrderBook(tick_size=tick)
    seed_book(book, rng=rng, **seed_kwargs)
    sim = Simulator(book, flow_cls(dict(params)), seed=seed)
    return sim.run(horizon)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow", default="hawkes")
    ap.add_argument("--regime", default="normal", help="regime for hawkes/garch")
    ap.add_argument("--exec-regime", default="exec", help="regime for AC/vwap")
    ap.add_argument("--horizon", type=float, default=1800.0)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tick-size", type=float, default=0.01)
    ap.add_argument("--out", default="artifacts/params")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # --- Hawkes MLE on one normal-regime run ---
    res = run_plain(args.flow, args.regime, args.seed, args.horizon, args.tick_size)
    times = [res.event_times_by_type[k] for k in EVENT_TYPES]
    fit = fit_hawkes_exp(times, T=args.horizon, beta=5.0)
    hp = fit.params
    (out / "hawkes.json").write_text(
        json.dumps(
            {
                "mu": np.asarray(hp.mu).tolist(),
                "alpha": np.asarray(hp.alpha).tolist(),
                "beta": np.asarray(hp.beta).tolist(),
                "loglik": fit.loglik,
                "n_obs": fit.n_obs,
                "converged": fit.converged,
                "branching": float(np.max(np.linalg.eigvals(np.asarray(hp.alpha) / np.asarray(hp.beta)).real)),
            },
            indent=2,
        )
    )
    print(f"hawkes: loglik={fit.loglik:.1f} n={fit.n_obs} -> {out/'hawkes.json'}")

    # --- GARCH on 1-second-resampled mid diffs ---
    ms = res.mid_series
    if len(ms) > 2:
        grid = np.arange(0, ms[-1, 0], 1.0)
        idx = np.clip(np.searchsorted(ms[:, 0], grid, side="right") - 1, 0, len(ms) - 1)
        diffs = np.diff(ms[idx, 1] * args.tick_size)
        g = GARCH11.fit(diffs)
        (out / "garch.json").write_text(
            json.dumps(
                {"omega": g.omega, "alpha": g.alpha, "beta": g.beta,
                 "loglik": g.loglik, "n_obs": g.n_obs},
                indent=2,
            )
        )
        print(f"garch: alpha={g.alpha:.3f} beta={g.beta:.3f} -> {out/'garch.json'}")

    # --- AC params on the exec regime + VWAP volume profile ---
    flow_cls, params, seed_kwargs = get_regime(args.exec_regime, flow=args.flow)
    ac = calibrate_almgren_chriss(
        flow_factory=lambda: flow_cls(dict(params)),
        seed_kwargs=seed_kwargs,
        n_runs=args.runs,
        horizon=min(args.horizon, 600.0),
        seed=args.seed,
        tick_size=args.tick_size,
    )
    (out / "ac.json").write_text(
        json.dumps(
            {"sigma": ac.sigma, "eta": ac.eta, "gamma": ac.gamma, "n_obs": ac.n_obs},
            indent=2,
        )
    )
    print(f"ac: sigma={ac.sigma:.5f} eta={ac.eta:.6f} gamma={ac.gamma:.6f}")

    results = [
        run_plain(args.flow, args.exec_regime, args.seed + 100 + i, args.horizon, args.tick_size)
        for i in range(args.runs)
    ]
    prof = VolumeProfile.fit(results, horizon=args.horizon, step=10.0)
    np.save(out / "vwap_profile.npy", prof.cum)
    print(f"vwap profile: {len(prof.cum)} buckets -> {out/'vwap_profile.npy'}")


if __name__ == "__main__":
    main()
