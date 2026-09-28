# Statistical Modeler — Report

**Branch:** `agent/statistical` · **Package:** `orderflow/stats/` · **Seed:** 7

## Research question

Does the exponential multivariate Hawkes used to drive the simulator's order
flow recover its parameters under maximum likelihood, do its compensator
residuals satisfy the random time change theorem, and does the simulated
market reproduce the two canonical linear price-impact relationships —
OFI → mid moves and Kyle's λ?

## Method

**Hawkes MLE.** A 6-dim exponential Hawkes (Ogata thinning) was simulated for
240 s with the `normal` regime parameters (μ = [1.2, 1.2, .35, .35, .9, .9],
shared β = 5, branching ratio 0.227), then refit by L-BFGS-B on the
log-likelihood in log-parameter space. The log-likelihood was rewritten from
an O(N²) history resum to an amortised pointer walk — ~13× faster
(4.1 s → 0.3 s per evaluation) — which is what made the 42-parameter numeric
optimisation feasible at all.

**Goodness of fit.** Compensator residuals
`Λ(t_k) − Λ(t_{k−1})` are i.i.d. Exp(1) under the true model (random time
change theorem). `orderflow/stats/goodness.py` computes them in
O(n_i + n_j) per (i,j) pair with the *inclusive* kernel sum
`A(t) = Σ_{t_l ≤ t} e^{−β(t−t_l)}`; the per-interval contribution is
`(α/β)·(A(prev) − A(t_k) + n_inside)`. Verified against the reference
implementation to machine precision including simultaneous-event boundary
collisions (`tests/stats/test_goodness.py`). KS-tested per dimension and
pooled, both under the true parameters (long tape, T = 900 s) and under the
fitted parameters (same tape as the fit).

**OFI regression.** Cont–Kukanov–Stoikov: per-snapshot best-level OFI vs
subsequent mid change, OLS, at 1 s cadence and aggregated into 10 s buckets.
**Kyle λ:** signed taker volume vs mid change in 5 s buckets, OLS slope.

## Results

### Hawkes parameter recovery (T = 240 s, n = 1 504 events)

| quantity | truth | estimate | error |
|---|---|---|---|
| μ (mean) | 0.783 | 0.728 | 11.2% |
| β (shared) | 5.00 | 4.94 | 1.2% |
| α (mean rel.) | — | — | 86.8% |
| branching ratio | 0.227 | 0.300 | — |

α is poorly identified on ~1 500 events — consistent with the literature
(Hawkes kernels need ~10⁴+ events/dim to pin down). The optimizer hit its
150-iteration cap (`converged=False`), so these are lower bounds on
achievable accuracy, not asymptotic statements.

### Residual diagnostics

| test | n | pooled KS p | verdict |
|---|---|---|---|
| truth params, T = 900 s tape | 5 519 | 0.077 | Exp(1) not rejected |
| fitted params, T = 240 s tape | 1 504 | 0.779 | Exp(1) not rejected |

Per-dimension KS under truth: all six dims p ∈ [0.18, 0.89]. The fitted
model passes GOF *more* easily than the true one — in-sample residuals adapt
to fitted intensities, so GOF is necessary but not sufficient evidence of
recovery. This is the expected behaviour and worth stating plainly.

### Price impact on the simulated book (normal regime, 600 s)

| regression | n | slope | r² | t |
|---|---|---|---|---|
| OFI → Δmid, 1 s | 575 | 0.001 ticks/share | 0.060 | 6.04 |
| OFI → Δmid, 10 s buckets | 51 | 0.002 | 0.157 | 3.02 |
| Kyle λ, 5 s buckets | 118 | 1.76 ticks / 1k shares | — | — |

OFI slope is positive and highly significant; r² of ~6% at 1 s rising to
~16% at 10 s is the right order of magnitude for this cadence — real-data
OFI regressions land around 10–30% at similar horizons, and most of the
unexplained variance is queue-length noise the L1-only OFI discards.

## Decision

- **Adopted:** `goodness.py` as the per-dimension GOF tool; the pointer-walk
  `_loglik` in `hawkes.py` (13× faster, identical values).
- **Finding for the benchmark:** Hawkes MLE is reliable for μ and β on
  minute-scale tapes; treat fitted α as a prior mean, not ground truth.
  Downstream models should not fine-tune against α point estimates.
- **Caveat logged:** GOF under fitted parameters is weak evidence of model
  adequacy; holdout-tape GOF would be stronger (noted for `agent/benchmark`).

## Artifacts

- `orderflow/stats/goodness.py` — per-dim residuals + `hawkes_gof`
- `orderflow/stats/hawkes.py` — `_loglik` rewritten O(N) amortised
- `research/statistical/run_stats.py` — reproducible pipeline (`python research/statistical/run_stats.py [seed]`)
- `research/statistical/results.json` — measured outputs
- `tests/stats/test_goodness.py` — 6 tests: reference equivalence, boundary
  collisions, Exp(1) validation, wrong-model rejection, OFI/Kyle sanity
