# Adversarial Researcher — Report

**Branch:** `agent/adversarial` · **Package:** `orderflow/eval/adversarial.py` · **Task:** sell 20k / 600 s / 10 s grid, exec regime, 4 seeds per cell

## Research question

Which attack mechanics actually hurt execution, and which strategies carry
the vulnerability? The harness already had four footprint-chasers (all key
on *market-order* submits). This branch adds a mechanically distinct fifth —
the **PennyJumper**, which detects large *passive* posts at the touch and
steps one tick inside them — and measures the full attack-uplift matrix.

## Method

- Uplift = mean IS(adversary present) − mean IS(alone), bps. Positive =
  the adversary made execution more expensive. Common random numbers:
  every adversary set replays the same four market seeds per algo.
- Algos: `twap` (aggressive slices), `twap_passive` (posts at best),
  `ac` (front-loaded aggressive).
- Adversary sets: each alone, plus `combined` = spoofer + igniter +
  withdrawer + pennyjumper.

## Results — IS uplift vs solo baseline (bps)

| adversary | twap | twap_passive | ac |
|---|---|---|---|
| spoofer | −1.9 | −4.8 | −5.8 |
| igniter | −4.0 | +0.3 | −12.2 |
| withdrawer | −29.4 | **+3333.9** | −24.0 |
| frontrunner | −14.7 | −14.4 | −26.4 |
| pennyjumper | −0.3 | +1.6 | +8.0 |
| combined | −29.8 | **+120.3** | −7.5 |

## Findings

1. **The withdrawer is catastrophic against passive-with-deadline execs.**
   +3334 bps is not noise — it is a mechanism. `twap_passive` posts at best
   and emits no market orders, so the footprint detector stays silent until
   the *deadline liquidation*: the unfulfilled parent is dumped as one huge
   market order, which trips the detector, and the withdrawer pulls the
   remaining bids exactly as the sweep arrives. The lesson is structural:
   **a strategy that relies on a forced-liquidation backstop has a
   one-shot detectable vulnerability.** Any serious passive algo must
   meter its deadline, not dump it.
2. **Against aggressive algos the withdrawer *helps* (−24 to −29 bps).**
   Thinning bids while a sell sweeps sounds harmful, but the cancelled
   depth removes support and lets the mid reset lower *between* slices —
   the exec's subsequent slices arrive after natural sell flow has already
   repriced the book. With n=4 the sign should be treated as indicative,
   but the negative magnitude across both twap and ac is suggestive.
3. **PennyJumper is a real but modest tax** (+1.6 bps on passive). Its
   60-share jump absorbs only the next slice of contra flow — queue-jumping
   degrades fill timing without blocking it. +8.0 on `ac` is likely noise
   (ac posts no passive orders; the jumper only sees natural posts).
4. **Frontrunner and igniter don't hurt at this task scale.** The front
   orders are small relative to the metaorder, and their unwinds can even
   hand back the price move. Footprint-chasing is not automatically costly.
5. **`combined` on passive = +120 bps** — synergistic but dominated by the
   withdrawer's deadline attack; the jumper contributes the queue pressure.

## Decision

- **Adversary to keep for selection:** `withdrawer` — it's the only one
  that breaks the completion-adjacent guarantee catastrophically, and it
  specifically punishes the deadline-dump antipattern. `pennyjumper` stays
  as the passive-strategy probe; the other three stay for coverage.
- **Selection note for the benchmark branch:** adversarial regimes must
  include a *passive* algo cell — an aggressive-only eval would miss the
  +3334 bps mode entirely.

## Artifacts

- `orderflow/eval/adversarial.py` — `PennyJumper` added (`ADVERSARIES["pennyjumper"]`)
- `research/adversarial/run_adv_study.py` — uplift-matrix pipeline
- `research/adversarial/results.json` — measured matrix
- `tests/eval/test_pennyjumper.py` — 4 tests: both-side jump prices,
  ignore rules, expiry
