# Microstructure Researcher — REPORT

**Question.** Does the event-driven simulator reproduce the canonical
stylised facts of equity limit-order books, and which flow model
(ZI vs Hawkes) earns which facts?

**Method.** `orderflow/stats/stylized.py` measures four things on a
`SimResult`: (1) return moments on 1 s mid resamples, (2) the |return|
autocorrelation (volatility clustering), (3) the trade-sign autocorrelation
(order-splitting memory), (4) the spread–imbalance relation. Runs: `normal`
regime, 600 s, seed 0, tick 0.01.

## Findings (measured, not assumed)

| fact | hawkes | zi | real markets |
|---|---|---|---|
| excess kurtosis (fat tails) | **+26.7** | +13.2 | large positive |
| vol clustering \|r\| acf[1] | +0.335 | +0.432 | positive, slow-decay |
| sign acf[1] | +0.545 | +0.547 | positive, very long memory |
| spread (ticks) | 2.55 | 1.06 | ~1–3 |
| corr(spread, \|imb\|) | −0.04 | −0.01 | negative (stronger) |

## What the Hawkes flow gets right — and what it doesn't

* **Fat tails & bursts.** Self-excitation produces clustered activity and
  heavy-tailed returns (xkurt +27 vs +13 for ZI). Mid vol: 0.66 ticks/s
  vs 0.17 for ZI.
* **Honest limitation: memory has one timescale.** Exponential kernels give
  e-folding ≈ 1/β = 0.2 s, so |r| acf collapses to noise by lag 5–10. Real
  equity volatility clusters for minutes/hours (power-law memory). A
  superposition of Hawkes kernels at multiple timescales, or a rough/Bouchaud
  kernel, would be needed to reproduce that; noted as future work.
* **Trade-sign persistence is generic.** Both models give sign acf[1] ≈
  +0.55 — most of it comes from *book mechanics* (sweeping a thin queue
  moves the price, attracting the same direction next event) not from the
  arrival process. Fact documented; long-lag persistence still absent.
* **Spread–imbalance coupling is too weak** (−0.04 vs strongly negative in
  real data). Our flow prices limits relative to the same-side best rather
  than crossing the spread strategically, so imbalance rarely squeezes the
  spread. Candidate fix for a future flow model.

## Deliverables

* `orderflow/stats/stylized.py` — resample/moments/acf/sign/spread stats.
* `tests/stats/test_stylized.py` — moment recovery on synthetic AR(1),
  resampling, spread stats.
* Calibration note: the Hawkes MLE in `stats/hawkes.py` recovers the
  generating mu/alpha/beta on our own simulated runs (sanity-checked in
  `stats` tests); the calibrated parameters land in
  `artifacts/params/hawkes.json` via `scripts/calibrate.py`.

**Decision.** Keep both flows. Hawkes for regimes/tests needing bursts and
fat tails; ZI as the cheap neutral backdrop and for fast tests. The missing
long-memory and weak spread–imbalance coupling are documented limitations,
not silently patched.
