# OrderFlow Leaderboard

Execution score `S = mean_IS_bps + 0.5 * CVaR95` (lower is better),
averaged over regimes. An algo survives only with completion >= 99.9%
in every regime. Predictors survive by beating the Markov baseline in
NLL on at least 4/6 targets per regime.

## Execution (IS bps, mean +/- std | CVaR95 | completion)

- **ac@combined** — score -58.85 (eliminated)
  - exec: -82.47 ± 9.26 | CVaR -73.21 | done 50% | part 6%
  - thin: +0.25 ± 1.76 | CVaR +2.25 | done 75% | part 6%
- **ac_adaptive@combined** — score -43.60 (SURVIVES)
  - exec: -64.52 ± 18.62 | CVaR -45.90 | done 100% | part 6%
  - thin: -0.54 ± 1.46 | CVaR +1.61 | done 100% | part 6%
- **learned@combined** — score -41.91 (SURVIVES)
  - exec: -57.45 ± 0.76 | CVaR -56.68 | done 100% | part 6%
  - thin: -0.29 ± 3.78 | CVaR +4.50 | done 100% | part 6%
- **rl@combined** — score -27.32 (eliminated)
  - exec: -49.07 ± 38.32 | CVaR -10.76 | done 0% | part 5%
  - thin: -1.04 ± 1.66 | CVaR +1.71 | done 100% | part 6%
- **twap@combined** — score -12.72 (SURVIVES)
  - exec: -34.83 ± 45.73 | CVaR +10.90 | done 100% | part 6%
  - thin: +0.45 ± 3.86 | CVaR +6.95 | done 100% | part 6%
- **vwap@combined** — score -12.72 (SURVIVES)
  - exec: -34.83 ± 45.73 | CVaR +10.90 | done 100% | part 6%
  - thin: +0.45 ± 3.86 | CVaR +6.95 | done 100% | part 6%
- **learned** — score -8.53 (SURVIVES)
  - exec: -43.93 ± 29.23 | CVaR -14.70 | done 100% | part 7%
  - normal: +1.57 ± 3.00 | CVaR +4.71 | done 100% | part 17%
  - volatile: +1.15 ± 2.26 | CVaR +4.79 | done 100% | part 5%
  - thin: -0.34 ± 4.35 | CVaR +3.71 | done 100% | part 7%
  - stressed: -0.76 ± 2.66 | CVaR +0.84 | done 100% | part 3%
- **ac_adaptive** — score -6.90 (SURVIVES)
  - exec: -33.64 ± 4.63 | CVaR -29.01 | done 100% | part 7%
  - normal: +0.18 ± 2.56 | CVaR +3.73 | done 100% | part 17%
  - volatile: +2.84 ± 3.46 | CVaR +5.70 | done 100% | part 5%
  - thin: +1.16 ± 2.30 | CVaR +3.96 | done 100% | part 7%
  - stressed: +0.62 ± 3.33 | CVaR +4.29 | done 100% | part 4%
- **rl** — score -6.04 (SURVIVES)
  - exec: -31.95 ± 3.21 | CVaR -28.74 | done 100% | part 7%
  - normal: -3.35 ± 5.39 | CVaR +4.15 | done 100% | part 17%
  - volatile: +4.36 ± 5.57 | CVaR +11.45 | done 100% | part 5%
  - thin: +0.76 ± 1.57 | CVaR +2.25 | done 100% | part 7%
  - stressed: +0.30 ± 6.93 | CVaR +10.23 | done 100% | part 4%
- **twap** — score -0.49 (eliminated)
  - exec: -11.11 ± 23.97 | CVaR +12.85 | done 100% | part 7%
  - normal: -0.63 ± 2.17 | CVaR +2.02 | done 75% | part 18%
  - volatile: +2.17 ± 3.12 | CVaR +4.67 | done 100% | part 6%
  - thin: -0.22 ± 3.25 | CVaR +4.51 | done 100% | part 7%
  - stressed: -5.70 ± 6.73 | CVaR +2.08 | done 100% | part 3%
- **vwap** — score -0.49 (eliminated)
  - exec: -11.11 ± 23.97 | CVaR +12.85 | done 100% | part 7%
  - normal: -0.63 ± 2.17 | CVaR +2.02 | done 75% | part 18%
  - volatile: +2.17 ± 3.12 | CVaR +4.67 | done 100% | part 6%
  - thin: -0.22 ± 3.25 | CVaR +4.51 | done 100% | part 7%
  - stressed: -5.70 ± 6.73 | CVaR +2.08 | done 100% | part 3%
- **ac** — score 0.03 (SURVIVES)
  - exec: -17.43 ± 6.87 | CVaR -10.56 | done 100% | part 7%
  - normal: -1.09 ± 4.82 | CVaR +5.28 | done 100% | part 17%
  - volatile: +5.53 ± 6.76 | CVaR +13.56 | done 100% | part 5%
  - thin: +1.83 ± 2.08 | CVaR +5.07 | done 100% | part 7%
  - stressed: -0.05 ± 8.37 | CVaR +9.34 | done 100% | part 4%
- **twap_capped@combined** — score 0.63 (SURVIVES)
  - exec: -3.27 ± 5.50 | CVaR +2.23 | done 100% | part 6%
  - thin: +0.26 ± 3.61 | CVaR +6.34 | done 100% | part 6%
- **twap_passive** — score 4.11 (eliminated)
  - exec: +3.88 ± 0.71 | CVaR +4.58 | done 100% | part 7%
  - normal: +0.73 ± 5.47 | CVaR +4.50 | done 25% | part 16%
  - volatile: +5.75 ± 2.88 | CVaR +9.23 | done 100% | part 6%
  - thin: +0.09 ± 3.17 | CVaR +4.00 | done 75% | part 6%
  - stressed: -2.40 ± 4.25 | CVaR +2.72 | done 100% | part 4%
- **twap_capped** — score 4.49 (eliminated)
  - exec: +1.82 ± 18.52 | CVaR +20.34 | done 100% | part 7%
  - normal: +0.78 ± 8.97 | CVaR +12.69 | done 75% | part 17%
  - volatile: +2.17 ± 3.12 | CVaR +4.67 | done 100% | part 6%
  - thin: -0.46 ± 2.90 | CVaR +3.54 | done 100% | part 7%
  - stressed: -3.50 ± 4.32 | CVaR +2.08 | done 100% | part 3%
- **twap_passive@combined** — score 15.91 (eliminated)
  - exec: +9.12 ± 15.34 | CVaR +24.47 | done 100% | part 6%
  - thin: +4.90 ± 4.86 | CVaR +11.11 | done 75% | part 6%

## Prediction (targets beaten vs Markov baseline)

- **logistic** — 4/6 target wins (SURVIVES)
- **mlp** — 4/6 target wins (SURVIVES)
- **transformer** — 4/6 target wins (SURVIVES)
