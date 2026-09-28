"""Statistical-modeler pipeline — writes results.json next to this script.

A. MLE recovery: simulate a known 6-dim Hawkes tape, refit with
   fit_hawkes_exp, compare mu/alpha against truth.
B. Goodness-of-fit under the TRUE params on a long tape: compensator
   residuals must be i.i.d. Exp(1) (random time change theorem) —
   validates both the model class and the O(N) residual implementation.
C. Market microstructure on a book sim: OFI->mid-change regression
   (Cont-Kukanov-Stoikov), Kyle lambda on bucketed signed flow.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from orderflow.book.book import LimitOrderBook
from orderflow.sim.flow.hawkes import HawkesParams, simulate_hawkes
from orderflow.sim.regimes import get_regime
from orderflow.sim.simulator import Simulator, seed_book
from orderflow.stats.goodness import hawkes_gof
from orderflow.stats.hawkes import fit_hawkes_exp
from orderflow.stats.impact import kyle_lambda
from orderflow.stats.ofi import compute_ofi_series, ofi_regression

OUT = Path(__file__).parent / "results.json"


def hawkes_experiment(rng: np.random.Generator) -> dict:
    _, params, _ = get_regime("normal", flow="hawkes")
    truth = HawkesParams(
        mu=np.array(params["mu"], float),
        alpha=np.array(params["alpha"], float),
        beta=float(params["beta"]),
    )

    t0 = time.time()
    tapes = simulate_hawkes(truth, T=240.0, rng=rng)
    fit = fit_hawkes_exp(tapes, T=240.0, maxiter=150, rng=rng)
    fit_seconds = time.time() - t0

    # long tape for GOF under truth (no fitting cost)
    long_tape = simulate_hawkes(truth, T=900.0, rng=rng)
    gof_truth = hawkes_gof(truth, long_tape, T=900.0)
    gof_fitted = hawkes_gof(fit.params, tapes, T=240.0)

    return {
        "truth": {
            "mu": truth.mu.tolist(),
            "alpha": truth.alpha.tolist(),
            "beta": float(truth.beta[0, 0]),
            "branching_ratio": truth.branching_ratio(),
        },
        "fit": {
            "mu": fit.params.mu.tolist(),
            "alpha": fit.params.alpha.tolist(),
            "beta": float(fit.params.beta[0, 0]),
            "branching_ratio": fit.params.branching_ratio(),
            "loglik": fit.loglik,
            "converged": fit.converged,
            "n_obs": fit.n_obs,
            "seconds": round(fit_seconds, 1),
        },
        "param_errors": {
            "mu_rel": float(
                np.abs(fit.params.mu - truth.mu).mean() / truth.mu.mean()
            ),
            "alpha_rel": float(
                np.abs(fit.params.alpha - truth.alpha).mean()
                / truth.alpha.mean()
            ),
            "beta_rel": float(
                abs(fit.params.beta[0, 0] - truth.beta[0, 0]) / truth.beta[0, 0]
            ),
        },
        "gof_truth": gof_truth,
        "gof_fitted": gof_fitted,
    }


def book_experiment(seed: int) -> dict:
    flow_cls, params, seed_kwargs = get_regime("normal", flow="hawkes")
    rng = np.random.default_rng(seed)
    book = LimitOrderBook(tick_size=0.01)
    seed_book(book, rng=rng, **seed_kwargs)
    sim = Simulator(book, flow_cls(params), seed=seed)
    res = sim.run(600.0, snapshot_every=1.0)

    # OFI -> mid change at 1 s cadence, raw and 10 s-aggregated
    ofi = compute_ofi_series(res.snapshots)
    mids = np.array(
        [
            (s.bid_prices[0] + s.ask_prices[0]) / 2
            if s.bid_prices[0] and s.ask_prices[0]
            else np.nan
            for s in res.snapshots
        ]
    )
    dm = np.diff(mids, prepend=mids[0])
    ofi_fit = ofi_regression(ofi, dm)
    ofi_fit_w = ofi_regression(ofi, dm, window=10)

    # Kyle lambda: signed taker volume vs mid change in 5 s buckets
    fills = res.fills
    if fills:
        t_end = fills[-1].timestamp
        nb = max(1, int(t_end // 5))
        vol = np.zeros(nb)
        for f in fills:
            b = min(int(f.timestamp // 5), nb - 1)
            vol[b] += f.taker_side.value * f.qty
        ts = np.array([s.timestamp for s in res.snapshots])
        mid_at = np.array(mids)
        edges = np.arange(nb + 1) * 5.0
        idx = np.clip(np.searchsorted(ts, edges) - 1, 0, len(ts) - 1)
        dp = np.diff(mid_at[idx])
        mask = np.isfinite(dp) & np.isfinite(vol[: len(dp)])
        lam = (
            kyle_lambda(vol[: len(dp)][mask], dp[mask])
            if mask.sum() > 2
            else float("nan")
        )
        n_buckets = int(mask.sum())
    else:
        lam, n_buckets = float("nan"), 0

    return {
        "n_snapshots": len(res.snapshots),
        "n_fills": len(fills),
        "ofi": {
            "beta": ofi_fit.beta,
            "r2": ofi_fit.r2,
            "t_stat": ofi_fit.t_stat,
            "n_obs": ofi_fit.n_obs,
        },
        "ofi_windowed": {
            "beta": ofi_fit_w.beta,
            "r2": ofi_fit_w.r2,
            "t_stat": ofi_fit_w.t_stat,
            "n_obs": ofi_fit_w.n_obs,
        },
        "kyle_lambda_ticks_per_1k_shares": lam * 1000.0,
        "kyle_n_buckets": n_buckets,
        "events_per_dim": {k: len(v) for k, v in res.event_times_by_type.items()},
    }


def main() -> None:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    rng = np.random.default_rng(seed)
    print(f"[stats] seed={seed}")

    print("[stats] hawkes: simulate + MLE ...", flush=True)
    haw = hawkes_experiment(rng)
    e = haw["param_errors"]
    print(
        f"  fit done in {haw['fit']['seconds']}s  converged={haw['fit']['converged']}  "
        f"rel err mu={e['mu_rel']:.1%} alpha={e['alpha_rel']:.1%} beta={e['beta_rel']:.1%}"
    )
    gt = haw["gof_truth"]
    print(
        f"  GOF(truth): pooled KS p={gt['pooled_ks_p']:.3f} on {gt['n_events']} residuals"
    )

    print("[stats] book sim: OFI + Kyle lambda ...", flush=True)
    bk = book_experiment(seed)
    print(
        f"  OFI r2={bk['ofi']['r2']:.3f} (w10: {bk['ofi_windowed']['r2']:.3f})  "
        f"kyle={bk['kyle_lambda_ticks_per_1k_shares']:.3f} ticks/1k"
    )

    results = {"seed": seed, "hawkes": haw, "book": bk}
    OUT.write_text(json.dumps(results, indent=2))
    print(f"[stats] wrote {OUT}")


if __name__ == "__main__":
    main()
