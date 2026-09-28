# OrderFlow — Evaluation

How the final numbers were produced, what survived, and the caveats a
careful reader should apply. Regenerate everything with:

```bash
python scripts/final_eval.py            # trains learned+rl, runs the matrix
python - <<'EOF'                        # prediction bench
from orderflow.eval.prediction_bench import PredBenchConfig, run_pred_bench
import json
out = run_pred_bench(regimes=["normal"], cfg=PredBenchConfig(n_train=6, n_test=3, seed=77))
print(json.dumps(out, indent=2))
EOF
```

## Design

**Common random numbers.** Every algo in a cell replays the same market
seeds — IS differences are paired against identical order-flow
realisations, not averaged over different worlds.

**Regimes.** `exec` (the flagship: sell 100 000 shares — ≈$10M at the
$100 mid — over 30 min on a 10 s grid), plus `normal`, `volatile`, `thin`,
`stressed` on scaled tasks. Adversarial cells run the same task under the
combined five-participant attack set.

**Robust score.** `S = mean_IS_bps + 0.5·CVaR95_IS_bps`, averaged over
regimes; adversarial cells carry weight 1.0. **Survival** additionally
requires completion ≥ 99.9% in every regime — cheap cost cannot rescue an
algo that leaves inventory unfilled.

**Predictors** are trained on train episodes and scored on disjoint test
episodes; survival requires beating the Markov baseline in NLL on ≥ 4/6
targets.

## Execution results (seed base 200)

Raw numbers: `artifacts/final_bench.json`; rendered: `LEADERBOARD.md`.
`algo@combined` rows are the same algo rerun under the five-adversary
attack; they are scored as separate entries so clean and adversarial
verdicts are both visible.

**Clean regimes** (exec/normal/volatile/thin/stressed, mean IS bps;
negative = beat arrival):

| algo | exec | normal | volatile | thin | stressed | score | verdict |
|---|---|---|---|---|---|---|---|
| learned | −43.9 | +1.6 | +1.2 | −0.3 | −0.8 | **−8.53** | **SURVIVES** |
| ac_adaptive | −33.6 | +0.2 | +2.8 | +1.2 | +0.6 | −6.90 | SURVIVES |
| rl | −32.0 | −3.4 | +4.4 | +0.8 | +0.3 | −6.04 | SURVIVES |
| ac | −17.4 | −1.1 | +5.5 | +1.8 | −0.1 | +0.03 | SURVIVES |
| twap / vwap | −11.1 | −0.6 | +2.2 | −0.2 | −5.7 | −0.49 | eliminated |
| twap_capped | +1.8 | +0.8 | +2.2 | −0.5 | −3.5 | +4.49 | eliminated |
| twap_passive | +3.9 | +0.7 | +5.8 | +0.1 | −2.4 | +4.11 | eliminated |

The completion gate is what eliminates the schedule algos, not cost:
on `normal` (4 000 shares / 600 s ≈ 40% of natural volume) they finish
at 25–75% — the deadline-liquidation backstop fires into an empty bid
side at t=T and the residue can't clear. That is the mechanism the
adversarial branch documented, occurring *naturally* on a thin book.

**Adversarial cells** (combined five-participant attack):

| algo@combined | exec IS / done | thin IS / done | verdict |
|---|---|---|---|
| ac_adaptive | −64.5 / 100% | −0.5 / 100% | SURVIVES |
| learned | −57.5 / 100% | −0.3 / 100% | SURVIVES |
| twap / vwap | −34.8 / 100% | +0.5 / 100% | (completes under attack) |
| twap_capped | −3.3 / 100% | +0.3 / 100% | (completes under attack) |
| ac | −82.5 / **50%** | +0.3 / 75% | eliminated |
| rl | −49.1 / **0%** | −1.0 / 100% | eliminated |
| twap_passive | +9.1 / 100% | +4.9 / 75% | eliminated |

**Verdict of record: `learned` wins** — best clean robust score, and
one of only two entries that complete everywhere under attack. Two
adversarial results deserve emphasis:

- **Static AC fails under attack despite −82 bps.** Its front-loaded
  schedule fires large detectable slices; the withdrawer pulls bids on
  the footprint and the unfilled residue has no time left — 50%
  completion on exec. Cheap IS on the filled half doesn't rescue it.
- **RL collapses to 0% under attack.** The PPO policy was trained on
  the clean `exec` regime; adversary order flow moves its observation
  features off the training distribution and it stops executing. A
  textbook train/test-distribution failure — kept in the table because
  it's the most instructive cell in the matrix.

## Prediction results (`normal`, seeds 77+, `research/benchmark/pred_results.json`)

Held-out NLL (lower is better); AUC on the binary targets:

| predictor | next_type | mid_move | spread_chg | cancel_prob | queue_depl | short_vol |
|---|---|---|---|---|---|---|
| markov | **1.69** | — | — | — | — | degenerate |
| logistic | 1.72 | 0.82 | 1.12 | 0.67 | 0.54 (AUC .81) | 2.92 |
| mlp | 1.71 | **0.76** | **0.94** | **0.68** | **0.42** (AUC .78) | **2.11** |
| transformer | 1.94 | 1.08 | 1.31 | 0.69 | 0.55 (AUC .49) | 24.5 |

All three learned predictors satisfy the survival rule (≥ 4 targets beaten
vs the Markov baseline where Markov only scores `next_type`/`short_vol`).
Two honest notes:

- **MLP > Transformer at this scale.** With ~10³ training windows the
  attention model can't cash in on sequence structure; its short_vol NLL
  blows up (24.5) from a bad variance head on sparse targets. This is a
  data-scale result, not a claim that transformers don't work on
  order flow — `artifacts/models/comparison.md` tracks the same gap in
  quick mode.
- **Markov stays competitive on `next_type`** (1.69 < all learned) —
  event type given the current book state is close to memoryless in the
  Hawkes regime. The learned models earn their keep on the *stateful*
  targets (queue depletion, mid move).

## Known limitations

- **Episode counts are small** (2 on exec, 4 on scaled regimes). Point
  estimates carry ±several bps of seed noise; paired diffs are more
  trustworthy than levels.
- **PPO is undertrained.** The reward-shaping ablation (`agent/rl`) showed
  entropy ≈ ln 11 at ~40–190 episodes. Treat `rl` as a wiring-verified
  entry, not a converged one.
- **Adversarial cells are single-set.** Five attack styles composed is a
  stress cell, not a red-team exhaustiveness claim.
- **`vwap` ≡ `twap` on flat profiles.** The exec regime has no intraday
  volume shape, so the two are bit-identical — keep VWAP for shaped
  regimes.
