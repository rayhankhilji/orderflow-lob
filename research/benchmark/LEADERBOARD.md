# OrderFlow Leaderboard

Execution score `S = mean_IS_bps + 0.5 * CVaR95` (lower is better),
averaged over regimes. An algo survives only with completion >= 99.9%
in every regime. Predictors survive by beating the Markov baseline in
NLL on at least 4/6 targets per regime.

## Execution (IS bps, mean +/- std | CVaR95 | completion)

- **ac@combined** — score -14.79 (eliminated)
  - exec: -19.63 ± 0.49 | CVaR -19.14 | done 100% | part 6%
  - thin: -1.67 ± 5.57 | CVaR +2.58 | done 75% | part 7%
- **twap@combined** — score -2.22 (eliminated)
  - exec: -13.21 ± 5.13 | CVaR -8.08 | done 50% | part 6%
  - thin: +6.02 ± 6.90 | CVaR +13.59 | done 100% | part 7%
- **vwap@combined** — score -2.22 (eliminated)
  - exec: -13.21 ± 5.13 | CVaR -8.08 | done 50% | part 6%
  - thin: +6.02 ± 6.90 | CVaR +13.59 | done 100% | part 7%
- **ac** — score 0.44 (SURVIVES)
  - exec: -21.59 ± 8.97 | CVaR -12.62 | done 100% | part 7%
  - normal: +5.38 ± 3.51 | CVaR +9.47 | done 100% | part 18%
  - volatile: +4.85 ± 5.98 | CVaR +12.93 | done 100% | part 6%
  - thin: +0.67 ± 5.73 | CVaR +5.90 | done 100% | part 7%
  - stressed: +1.83 ± 3.84 | CVaR +6.41 | done 100% | part 3%
- **twap** — score 4.15 (SURVIVES)
  - exec: -8.62 ± 0.04 | CVaR -8.58 | done 100% | part 6%
  - normal: -0.42 ± 5.85 | CVaR +5.15 | done 100% | part 18%
  - volatile: +7.61 ± 12.20 | CVaR +23.74 | done 100% | part 6%
  - thin: +1.90 ± 4.43 | CVaR +6.33 | done 100% | part 8%
  - stressed: +2.24 ± 6.25 | CVaR +9.43 | done 100% | part 4%
- **vwap** — score 4.15 (SURVIVES)
  - exec: -8.62 ± 0.04 | CVaR -8.58 | done 100% | part 6%
  - normal: -0.42 ± 5.85 | CVaR +5.15 | done 100% | part 18%
  - volatile: +7.61 ± 12.20 | CVaR +23.74 | done 100% | part 6%
  - thin: +1.90 ± 4.43 | CVaR +6.33 | done 100% | part 8%
  - stressed: +2.24 ± 6.25 | CVaR +9.43 | done 100% | part 4%
- **twap_passive** — score 7.68 (eliminated)
  - exec: +18.78 ± 7.98 | CVaR +26.76 | done 100% | part 7%
  - normal: +2.37 ± 4.76 | CVaR +6.83 | done 0% | part 15%
  - volatile: -1.47 ± 8.40 | CVaR +4.51 | done 100% | part 6%
  - thin: +0.02 ± 0.77 | CVaR +1.08 | done 100% | part 8%
  - stressed: -3.76 ± 6.99 | CVaR +5.77 | done 100% | part 4%
- **twap_passive@combined** — score 825.36 (eliminated)
  - exec: +798.46 ± 911.89 | CVaR +1710.35 | done 0% | part 6%
  - thin: -2.82 ± 4.19 | CVaR -0.21 | done 100% | part 7%

## Prediction (targets beaten vs Markov baseline)

