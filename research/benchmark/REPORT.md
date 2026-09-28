# Benchmark Agent — Report

**Branch:** `agent/benchmark` · **Harness:** `orderflow/eval/` · **Regimes:** exec, normal, volatile, thin, stressed (new)

## Research question

Does the selection pipeline — per-regime runs with common random numbers,
robust scoring `S = mean_IS + 0.5·CVaR95`, a hard completion gate —
produce a sensible, explainable leaderboard? And does the new `stressed`
regime (exec-rate flow into a thin displayed book with 25% hidden
liquidity) discriminate anything the existing four miss?

## Method

- Pass 1: `twap`, `twap_passive`, `vwap`, `ac` × 5 regimes. exec: 2
  episodes of 100k/1800s; others: 4 episodes on scaled tasks.
- Pass 2: `exec` + `thin` under the combined adversary set (spoofer,
  igniter, withdrawer, frontrunner) — same seeds, `algo@combined` rows.
- Selection via `write_leaderboard` → `LEADERBOARD.md` + `survivors.json`.

## Results

Clean pass (S, lower better): **ac 0.44** < twap = vwap 4.15 <
twap_passive 7.68. VWAP ≡ TWAP bit-for-bit (flat volume profile → the two
schedules coincide — verified, not assumed).

Under the combined attack:

| cell | mean IS (bps) | completion |
|---|---|---|
| twap@exec | −13.2 | **50%** |
| twap_passive@exec | +798.5 | **0%** |
| ac@exec | −19.6 | 100% |
| twap@thin | +6.0 | 100% |
| ac@thin | −1.7 | 75% |

The withdrawer mechanism found on `agent/adversarial` shows up in the full
matrix: `twap_passive` relies on its deadline-dump backstop and gets
destroyed (+798 bps mean, +1710 CVaR). `twap` keeps a front-loaded pace
but the attack still halves exec completion. **`ac` is the only cell that
completes under attack on the big task** — front-loading finishes before
the book thins.

## Survivors (this run)

| algo | survives | why |
|---|---|---|
| ac | **yes** | completes everywhere clean; best score |
| twap | yes | completes everywhere clean |
| vwap | yes | = twap here |
| twap_passive | **no** | 0% completion in `normal` |
| any `@combined` | no | adversarial cells always lose completion somewhere |

## Findings for the merge

1. The selection rule does its job: completion is the binding gate — cheap
   IS cannot buy a survivor flag if a regime starves the algo.
2. `stressed` discriminates mildly on the clean pass (all algos complete,
   IS sits between `normal` and `volatile`) — its real value is hidden-depth
   deception, which the OFI/prediction models will feel more than the
   execution algos do. Keep it in the matrix for Phase 7's model eval.
3. Adversarial cells should enter the score average with weight 1.0 —
   already implemented — but they must never gate completion on the clean
   algos' flags: the current `exec_scores` treats `algo@adv` as its own
   row, which is correct.

## Artifacts

- `orderflow/sim/regimes.py` — `stressed` regime (all three flows)
- `research/benchmark/run_benchmark.py` — two-pass pipeline
- `research/benchmark/LEADERBOARD.md`, `survivors.json`, `results.json`
