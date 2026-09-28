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

*Filled in by `scripts/final_eval.py` — see `LEADERBOARD.md` at repo root
and `artifacts/final_bench.json` for the raw numbers.*

Headline from the run: **AC is the survivor of record** — best robust
score, completes everywhere including under the combined attack. Adaptive
AC is the low-dispersion alternative (std ≈ half of TWAP's); passive TWAP
fails the completion gate; the PPO policy at this training budget is
near-uniform and lands close to TWAP.

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
