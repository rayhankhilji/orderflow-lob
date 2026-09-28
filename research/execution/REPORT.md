# Execution Researcher — Report

**Branch:** `agent/execution` · **Package:** `orderflow/execution/` · **Seeds:** 311–316 (common random numbers)

## Research question

On the flagship task — liquidate 100 000 shares (~$10M at the $100 reference)
over 30 minutes on a 10 s decision grid — do the competing variants
(`ac_adaptive`, a vol-adaptive re-planner; `twap_capped`, a displayed-depth
participation cap) beat the static baselines, and does the closed-form AC
frontier line up with realized sim costs?

## Method

- Six algorithms × six market seeds on the `exec` regime, **common random
  numbers**: every algo replays the same order-flow realisation, so IS
  differences are paired rather than environmental.
- `ac_adaptive` re-derives κ each step from an EWMA estimate of mid
  volatility: `n_j = R_j·(1 − sinh κ(T_rem−τ)/sinh κT_rem)`, floored at the
  TWAP-on-remaining increment so the re-plan can never fall behind schedule.
- `twap_capped` clips children to 35% of the top-8 displayed opposing depth,
  floored identically.
- Frontier: `ac_frontier` over λ ∈ [1e-7, 1e-3] with the sim-calibrated
  σ = 0.0174 $/(sh·√s), η = 5.2e-4 $·s/sh², γ = 0.

## Results

### Head-to-head (6 paired seeds, sell 100k / 1800 s)

| algo | mean IS (bps) | std | CVaR95 | robust S* |
|---|---|---|---|---|
| **ac** | **−28.02** | 12.37 | −16.08 | **−36.06** |
| twap_capped | −18.08 | 18.41 | 9.98 | −13.09 |
| ac_adaptive | −9.33 | **6.52** | −2.44 | −10.55 |
| twap | −8.58 | 22.39 | 14.50 | −1.33 |
| vwap | −8.58 | 22.39 | 14.50 | −1.33 |
| twap_passive | +12.85 | 6.64 | +24.51 | +25.11 |

\* S = mean_IS + 0.5·CVaR95 — the repo's survivor score. All algos filled
100%; participation ≈ 6.5% of episode volume.

Paired vs TWAP: `ac` −19.4 bps mean, `twap_capped` −9.5 bps,
`ac_adaptive` −0.7 bps, `vwap` identical (the exec regime's volume profile
is flat — VWAP correctly degenerates to TWAP rather than inventing a shape).

### AC efficient frontier (closed form)

| λ | E[C] | sd(C) |
|---|---|---|
| 1e-7 | $2 891 | $41 916 |
| 1e-5 (deployed) | $6 294 | $24 713 |
| 1e-4 | $19 825 | $13 553 |
| 1e-3 | $62 285 | $6 997 |

The frontier is monotone and convex — the textbook tradeoff. Deployed
λ = 1e-5 sits in the knee: $6.3k expected cost at $24.7k dispersion.

## Findings

1. **Static AC dominates on the robust score.** Front-loading captures the
   impact-vs-timing tradeoff the frontier predicts; realized mean IS
   (−28 bps ≈ −$28k) beats its model E[C] because drift in these seeds
   favoured early completion — IS is measured against arrival mid, so a
   favourable drift legs into the timing term.
2. **`ac_adaptive` is the dispersion killer, not the mean killer.** Cutting
   std 12.4→6.5 bps vs static AC at the cost of ~19 bps of mean is the
   classic adaptive-profile signature: it forfeits the drift windfall but
   nearly eliminates tail dispersion. It is the right choice when the
   objective weights variance more heavily (its S beats TWAP/VWAP/passive).
3. **Passive TWAP is adversely selected.** Posting at best and letting the
   market come to you fills disproportionately when the mid moves against —
   +12.9 bps mean with the *worst* tail (CVaR +24.5). Cheap on spread,
   expensive on selection.
4. **The depth cap is nearly free insurance.** `twap_capped` keeps the
   children inside displayed liquidity and earns −9.5 bps paired vs TWAP —
   the gentle early pace happens to time the drift better in these seeds,
   but structurally it reduces footprint variance.

## Decision

- **Survivor candidate:** `ac` (robust champion) and `ac_adaptive`
  (low-dispersion alternative — survives iff the final selection weights
  the CVaR term as specified).
- `twap_passive` fails the spirit of the mandate (worst tail, worst mean).
- VWAP adds nothing in flat-profile regimes — keep for regimes with a real
  intraday shape, where it should separate from TWAP.

## Artifacts

- `orderflow/execution/adaptive.py` — `AdaptiveAC` (`ac_adaptive`),
  `CappedTWAP` (`twap_capped`)
- `research/execution/run_exec_study.py` — reproducible pipeline
  (`python research/execution/run_exec_study.py [seed_base]`)
- `research/execution/results.json` — measured outputs
- `tests/execution/test_adaptive.py` — completion guarantees + vol tracking
