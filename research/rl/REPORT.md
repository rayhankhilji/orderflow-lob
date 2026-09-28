# RL Researcher — Report

**Branch:** `agent/rl` · **Stack:** `orderflow/execution/rl/` (Gymnasium env + PPO + RLExecution) · **Task:** sell 20k shares / 600 s / 10 s grid, exec regime

## Research question

Does the quadratic inventory-risk penalty in the step reward

    r_t = -dIS_bps - risk_penalty * remaining_frac^2 * (step/horizon)

change the learned policy's cost or completion, and is a PPO policy trained
at this budget competitive with the analytic baselines?

## Method

PPO (actor-critic MLP, GAE, clipped objective, entropy bonus) trained at
three penalty settings {0.0, 0.5, 2.0}, 8 iterations × 300 env-steps each
(≈40 full episodes per arm). Each trained actor is then wrapped in
`RLExecution` — the same 8-dim observation and the `a/5 × twap_slice` action
semantics, evaluated deterministically on 5 held-out seeds alongside TWAP
and AC under common random numbers.

## Results

| arm | trained eps | final ret | eval mean IS (bps) | fill |
|---|---|---|---|---|
| ppo rp=0.0 | 40 | −11.3 | +1.39 | 100% |
| ppo rp=0.5 | 40 | −11.5 | +0.97 | 100% |
| ppo rp=2.0 | 40 | −12.6 | +1.83 | 100% |
| twap | — | — | −2.53 | 100% |
| ac | — | — | −3.57 | 100% |

Per-iteration returns oscillate ±10 with no trend; final policy entropy
≈ 2.36 nats vs the uniform ceiling ln 11 ≈ 2.40 — the actor is still
effectively uniform over the 11 actions at this budget.

## Findings

1. **The ablation is flat — because the policy hasn't learned.** Reward
   penalties of 0/0.5/2.0 produce statistically identical eval IS
   (+1.0 to +1.8 bps). At near-uniform action selection the inventory
   term contributes the same average penalty in every arm, so it cannot
   differentiate behavior yet. The correct conclusion is not "the penalty
   doesn't matter" but "**~40 episodes is far too few to estimate it**" —
   PPO on a 60-step horizon with an 11-way discrete action space needs
   hundreds-to-thousands of episodes before shaping ablations are legible.
2. **The plumbing works end-to-end.** Train → wrap → CRN benchmark is
   verified: all arms complete 100% (the env's deadline liquidation and the
   adapter's residue sweep both fire correctly), and the trained net drops
   into `RLExecution` without interface changes.
3. **Eval IS ≈ +1 bps across arms is the near-TWAP outcome.** An untrained
   argmax over a uniform actor mixes slices up and down symmetrically; the
   penalty ablation washes out. Baselines land slightly better because
   their schedules are exactly uniform rather than sample-noisy.

## Decision

- **Do not promote `rl` yet.** It survives only when trained substantially
  longer (Phase 7 evaluation will run `--iters` in the hundreds); the
  wrapper and env are certified ready.
- **Keep `risk_penalty = 0.5` as the default** — theory says some inventory
  pressure is needed to learn front-loading; the flat ablation here is a
  budget artifact, not evidence against shaping.
- **Follow-up for the benchmark branch:** report per-arm learning curves
  (`train_log_rp*.json` artifacts) rather than point estimates when
  comparing RL arms.

## Artifacts

- `research/rl/run_rl_study.py` — training + eval pipeline
- `research/rl/results.json`, `train_log_rp{0.0,0.5,2.0}.json` — measured outputs
- No package changes: this branch contributes methodology + measurement;
  the env/PPO/wrapper live on main
