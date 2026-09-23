# OrderFlow — Architecture & Design Spec

A limit-order-book research and simulation environment: matching engine, stochastic
order-flow simulators, statistical microstructure models, neural next-event models,
optimal execution algorithms, and a multi-agent research/evaluation workflow.

This document is the contract. Every package below must conform to the interfaces
described here so that competing implementations (in `agent/*` branches) are
interchangeable and can be scored by the same evaluation harness.

---

## 0. Conventions

- Python 3.12, `uv`-managed venv at `.venv/`. `torch==2.2.2` (Intel macOS wheel
  ceiling), `numpy<2`, `scipy<1.14`, `pandas`, `gymnasium`, `pytest`, `hypothesis`.
- Prices are **integer ticks** (`int`) everywhere inside the engine. Convert to
  float only at the presentation/metrics boundary via `tick_size`.
- Quantities are **integer shares** (`int`).
- Time is a **float seconds** simulation clock (`float`), monotone non-decreasing.
- Side is `Side.BUY` / `Side.SELL` (`enum.IntEnum`, BUY=1, SELL=-1 so that
  `side * price_move` gives signed PnL direction).
- Every stochastic component takes an explicit `rng: numpy.random.Generator`
  (never uses the global RNG). Seeds flow from the top (`seed -> np.random.default_rng`).
- Pure-Python hot loops are acceptable for correctness; use `numpy` vectorization
  for stats/ML. No numba/cython.
- Tests live in `tests/`, mirror the package path, and use `pytest`. Property-based
  invariants (no crossed book, conservation of quantity, FIFO) use `hypothesis`.
- Type hints on all public functions; dataclasses (`slots=True`) for records.
- Line length 100, `ruff` for lint/format.

Package root: `orderflow/`.

```
orderflow/
  book/        L1/L2/L3 limit order book + matching engine + microstructure metrics
  sim/         event-driven simulator and stochastic order-flow generators
  stats/       statistical models (Hawkes, point-process diagnostics, vol, OFI, impact)
  models/      order-flow representation, datasets, Transformer + baselines, training
  execution/   TWAP, VWAP, Almgren–Chriss, RL env + PPO, learned policy
  eval/        benchmark harness, adversarial regimes, selection rule, leaderboard
  registry.py  plugin registry used by branches to register competing implementations
research/      per-agent research notes and experiment scripts (one dir per agent)
tests/
scripts/       CLI entry points (run_sim, train_model, run_benchmark, ...)
```

---

## 1. `orderflow.book` — Order book and matching engine

### 1.1 Types (`book/types.py`)

```python
class Side(IntEnum):
    BUY = 1
    SELL = -1


class OrderType(Enum):
    LIMIT, MARKET


class TimeInForce(Enum):
    GTC, IOC  # IOC: unfilled remainder cancelled, never rests


@dataclass(slots=True)
class Order:
    order_id: int
    side: Side
    price: int | None  # None for MARKET
    qty: int  # original quantity
    remaining: int  # unfilled quantity
    timestamp: float
    order_type: OrderType = OrderType.LIMIT
    tif: TimeInForce = TimeInForce.GTC
    hidden: bool = False  # fully hidden (not shown in L2 depth)
    display_qty: int | None = None  # iceberg: visible peak; None => fully displayed
    owner: str = "flow"  # tag for attribution (e.g. "flow", "exec", "adversary")


@dataclass(slots=True)
class Fill:
    timestamp: float
    price: int
    qty: int
    maker_id: int
    taker_id: int
    taker_side: Side
    maker_hidden: bool


class EventType(Enum):
    SUBMIT, CANCEL, FILL, MODIFY  # public tape of what happened


@dataclass(slots=True)
class BookEvent:
    timestamp: float
    type: EventType
    side: Side | None
    price: int | None
    qty: int
    order_id: int
    is_market: bool = False
```

### 1.2 Price level (`book/level.py`)

`PriceLevel(price: int)` holds a FIFO `deque[Order]` of **visible** resting orders
and a second FIFO `deque[Order]` of **hidden** resting orders. Exposes
`visible_qty`, `hidden_qty`, `total_qty`, `n_orders`, and `position_of(order_id) -> (index, qty_ahead)`.

### 1.3 Book (`book/book.py`) — `LimitOrderBook`

Price-time priority, with the standard exchange rule that at a given price
**displayed liquidity has priority over hidden liquidity**, and within each class
strict FIFO by arrival. Iceberg orders: the displayed peak sits in the visible
queue; when the peak is exhausted, a new peak of `min(display_qty, remaining)` is
re-queued at the **back** of the visible queue (loses time priority), and the
reserve otherwise behaves as hidden.

API:

```python
class LimitOrderBook:
    def __init__(self, tick_size: float = 0.01, record_events: bool = True): ...
    # --- mutation (all return list[Fill]) ---
    def submit(self, order: Order) -> list[Fill]          # limit (may cross) or market
    def cancel(self, order_id: int, qty: int | None = None) -> bool   # partial cancel if qty given
    def modify(self, order_id: int, new_price: int | None = None, new_qty: int | None = None) -> list[Fill]
        # price change or qty increase => loses priority (cancel+resubmit with same id, new timestamp)
        # qty decrease only => keeps priority
    # --- L1 ---
    def best_bid(self) -> int | None; def best_ask(self) -> int | None
    def mid(self) -> float | None; def spread(self) -> int | None
    def microprice(self) -> float | None   # (ask*bidqty + bid*askqty)/(bidqty+askqty) on visible L1
    # --- L2 ---
    def depth(self, side: Side, levels: int = 10, include_hidden: bool = False) -> list[tuple[int, int]]
        # [(price, qty)] best-first
    def snapshot(self, levels: int = 10) -> L2Snapshot   # dataclass with bid/ask arrays + timestamp
    # --- L3 ---
    def get_order(self, order_id: int) -> Order | None
    def queue_position(self, order_id: int) -> QueuePosition   # (price, index, qty_ahead_visible, qty_ahead_total)
    def level(self, side: Side, price: int) -> PriceLevel | None
    # --- tape ---
    events: list[BookEvent]; fills: list[Fill]; now: float
```

Matching semantics:
- A limit order that crosses the spread walks the opposite side level by level,
  best price first, filling FIFO (visible then hidden) until exhausted or price
  no longer crosses; any `remaining` rests (GTC) or is discarded (IOC).
- A market order walks the book until filled or the book is empty; remainder is
  **discarded** (never rests). `Fill.price` is always the **maker's** price.
- Partial fills reduce `remaining` on both sides; an order with `remaining == 0`
  is removed and can no longer be cancelled.
- Cancelling an unknown/filled id returns `False`.
- Book invariants (asserted in tests, never violated): `best_bid < best_ask`
  whenever both exist; sum of level quantities == sum of resting `remaining`;
  quantity conserved across each `submit` (`taker filled == sum maker filled`).

### 1.4 Metrics (`book/metrics.py`)

Pure functions over snapshots / tapes:

- `l1_imbalance(snap) = (bidqty1 - askqty1) / (bidqty1 + askqty1)` in [-1, 1]
- `depth_imbalance(snap, levels=k)` — same over cumulative k levels
- `ofi(snap_prev, snap_next)` — Cont–Kukanov–Stoikov order-flow imbalance:
  `e_n = 1{Pb_n >= Pb_{n-1}} qb_n - 1{Pb_n <= Pb_{n-1}} qb_{n-1} - 1{Pa_n <= Pa_{n-1}} qa_n + 1{Pa_n >= Pa_{n-1}} qa_{n-1}`
- `spread_series(snapshots)`, `realized_spread`, `effective_spread(fill, mid_at_fill)`
- `adverse_selection(fills, mids, horizon)` — for each fill, the signed mid move
  `taker_side * (mid(t+h) - mid(t))` in ticks: positive = taker was informed
  (maker adversely selected). Return per-fill array + mean.
- `queue_depletion_time(...)` helper used as a label generator in `models`.

---

## 2. `orderflow.sim` — Event-driven simulator

### 2.1 Core (`sim/simulator.py`)

```python
class FlowModel(Protocol):
    def next_event(self, book: LimitOrderBook, t: float, rng) -> tuple[float, FlowAction]
        # returns (dt until event, action). Action is one of
        # SubmitLimit(side, price_offset_ticks, qty, hidden, display_qty) | SubmitMarket(side, qty) | Cancel(order_id or "random at side/level")

class Simulator:
    def __init__(self, book, flow: FlowModel, seed: int, participants: list[Participant] = ()): ...
    def run(self, horizon: float, snapshot_every: float | None = None) -> SimResult
    def step(self) -> BookEvent   # advance one event
```

`Participant` is a callback protocol `on_event(book, event, t) -> list[Order|CancelReq]`
used by execution algorithms and adversaries to act between flow events.
`SimResult` holds `snapshots: list[L2Snapshot]`, `events`, `fills`, `mid_series`,
`participant_fills: dict[str, list[Fill]]`.

Initial state: `seed_book(book, mid_ticks, levels, qty_per_level, rng)` populates
both sides so simulation never starts from an empty book.

### 2.2 Flow models (`sim/flow/`)

1. `ZeroIntelligenceFlow` (Smith–Farmer–Gillemot–Iori): independent Poisson limit
   arrivals per level (rate decaying with distance from best), market orders, and
   per-order cancellation hazard. Parameters `lambda_limit, lambda_market, mu_cancel,
   depth_decay, size_dist`.
2. `HawkesFlow`: multivariate Hawkes process with exponential kernels over event
   types `{buy_limit, sell_limit, buy_market, sell_market, buy_cancel, sell_cancel}`.
   Intensity `lambda_i(t) = mu_i + sum_j sum_{t_k^j < t} alpha_ij exp(-beta_ij (t - t_k^j))`.
   Simulated by Ogata's modified thinning. Spectral radius of `alpha/beta` must be < 1
   (assert). Exposes `intensity(t)`.
3. `QueueReactiveFlow` (Huang–Lehalle–Rosenbaum): intensities at each of the first
   k levels are functions of the queue size at that level (lookup tables or
   parametric), so the book mean-reverts in depth. Includes a `theta` probability of
   reference-price change on total depletion.

Each flow model must be reproducible given `(params, seed)`, and must be able to
generate hidden/iceberg orders with probability `p_hidden`.

### 2.3 Regimes (`sim/regimes.py`)

Named parameter presets: `calm`, `normal`, `volatile`, `thin` (used by benchmark).
Adversarial regimes live in `eval/adversarial.py` (see §6).

---

## 3. `orderflow.stats` — Statistical models

- `stats/hawkes.py`: `fit_hawkes_exp(event_times: list[np.ndarray], beta_init, ...) -> HawkesParams`
  by maximum likelihood (log-likelihood with the recursive `R` term for exponential
  kernels; optimise with `scipy.optimize.minimize`, L-BFGS-B, positivity via log-params).
  Also `branching_ratio(params)` and `simulate(params, T, rng)` (shares code with
  `sim/flow/hawkes.py`).
- `stats/point_process.py`: `compensator(times, intensity_fn)`; residual analysis via
  the random time change theorem (transformed inter-arrivals should be Exp(1)):
  `ks_test_exp1(residuals)`; `qq_points`.
- `stats/volatility.py`: `realized_variance(mid, window)`; `GARCH11.fit/forecast`
  (MLE); `LogSV` (log stochastic volatility, estimated by QMLE via Kalman filter
  on log squared returns, Harvey–Ruiz–Shephard); `parkinson`, `garman_klass`.
- `stats/ofi.py`: `ofi_regression(ofi_series, mid_changes) -> (beta, r2, t_stat)`
  reproducing Cont–Kukanov–Stoikov linear price-impact-of-OFI.
- `stats/impact.py`:
  - `kyle_lambda(signed_volume, price_change)` regression,
  - `square_root_law(volumes, impacts, adv, sigma) -> Y` fit `I = Y sigma sqrt(Q/V)`,
  - `PropagatorModel` (Bouchaud transient impact): fit power-law decay kernel
    `G(l) ~ l^{-beta}` from sign autocorrelation and response function,
  - `calibrate_almgren_chriss(sim_results) -> (sigma, eta, gamma)` — estimates the
    temporary and permanent impact coefficients from controlled metaorder
    experiments in the simulator (used by execution).

Every fitter returns a dataclass with parameters, log-likelihood (where applicable),
and `n_obs`.

---

## 4. `orderflow.models` — Neural next-event models

### 4.1 Order-flow representation (`models/features.py`)

Two aligned streams built from `SimResult` (or any L3 tape):

1. **Event tokens** — one per book event: categorical fields
   `type ∈ {limit, market, cancel} × side` (6 classes), `level_bucket` (distance of
   the event price from same-side best, clipped to 0..K), `size_bucket`
   (quantile-binned, B bins), and continuous `log_dt` (log inter-arrival time).
2. **State features** at event time — L2 snapshot of top `k` levels normalised by
   mid and total depth: `[spread, l1_imbalance, depth_imbalance_k, ofi_window,
   microprice - mid, realized_vol_window, time_of_day_frac, queue features]`.

### 4.2 Targets (`models/targets.py`) — label generators, each documented:

- `mid_move`: sign of `mid(t + h_events) - mid(t)` in {-1, 0, +1} (3-class).
- `spread_change`: `spread(t+h) - spread(t)` in ticks, clipped to {-2..2} (5-class).
- `next_event`: the next token's `(type×side, level_bucket, size_bucket, dt)`.
- `cancel_prob`: for a resting order at L1, probability it is cancelled before it
  fills (binary, per resting order at snapshot).
- `short_vol`: realised variance of mid over next `h` events (regression, log-space).
- `queue_depletion`: whether the best-bid (or ask) queue empties within `h` events
  (binary) and, optionally, the time to depletion.

### 4.3 Model (`models/transformer.py`) — `OrderFlowTransformer`

```
tokens (type/side, level, size embeddings + log_dt proj + state MLP)
   -> learned positional embedding, causal TransformerEncoder (pre-LN, d=128, 4 layers, 4 heads)
   -> per-position hidden h_t
   heads (each nn.Linear on h_t):
     next_type (6-way softmax), next_level (K+1), next_size (B), next_dt (log-normal: mu, log_sigma)
     mid_move (3), spread_change (5), cancel_prob (1, BCE), short_vol (1, Gaussian NLL), queue_depletion (1, BCE)
```

Loss = weighted sum; weights in config. `predict_next_event_distribution(x)` returns
the factorised distribution objects. Baselines in `models/baselines.py`:
`LogisticBaseline` (multinomial logit on state features) and `MLPBaseline`, and a
`MarkovBaseline` (empirical transition matrix over type×side) for next_type.

### 4.4 Training (`models/train.py`, `scripts/train_model.py`)

`build_dataset(sim_results, seq_len) -> TensorDataset`, train/val/test split by
episode, AdamW, cosine schedule, early stopping on val loss, saves
`artifacts/models/<name>.pt` + `metrics.json`. Must run end-to-end on CPU in a few
minutes at small scale (`--quick`).

---

## 5. `orderflow.execution` — Optimal execution

Problem instance (the benchmark default): liquidate (or acquire) `notional = $10M`
of a stock at reference price `P0` over `T = 30 min`, child orders every
`dt = 10 s`  (180 decision steps).

```python
@dataclass
class ExecutionTask:
    side: Side; total_qty: int; horizon: float; step: float; start_time: float

class ExecutionAlgo(Protocol):
    name: str
    def reset(self, task: ExecutionTask, book: LimitOrderBook, rng) -> None
    def decide(self, t: float, book: LimitOrderBook, state: ExecState) -> list[ChildOrder]
        # ChildOrder(side, qty, kind: "market"|"limit", price_offset_ticks: int | None)

@dataclass
class ExecState:
    remaining_qty: int; elapsed: float; fills: list[Fill]; arrival_mid: float
```

Implementations:
- `TWAP` — equal slices per step; market orders (`aggressive=True`) or limit at
  best with a fallback to market at the deadline.
- `VWAP` — slices proportional to an expected intraday volume profile, estimated
  from a calibration set of simulations (`VolumeProfile.fit(sim_results)`).
- `AlmgrenChriss` — closed-form optimal trajectory for linear temporary (`eta`)
  and permanent (`gamma`) impact and risk aversion `lambda`:
  `x_j = X sinh(kappa (T - t_j)) / sinh(kappa T)`,
  `kappa = acosh(1 + (lambda sigma^2 tau^2)/(2 eta_tilde)) / tau` with
  `eta_tilde = eta - gamma tau / 2`. Params come from `stats.impact.calibrate_almgren_chriss`.
  Also exposes `expected_cost(...)` and `cost_variance(...)` for the efficient frontier.
- `execution/rl/env.py` — `ExecutionEnv(gymnasium.Env)`: observation
  `[remaining_frac, time_frac, spread, l1_imbalance, depth_imbalance, ofi, recent_return, vol]`,
  action = fraction of remaining to send this step (Discrete(11) over {0, .1, …, 1.0}
  of a TWAP-multiple, i.e. action `a` sends `a/5 * twap_slice`, clipped to
  remaining), reward = `-(implementation shortfall increment) - risk_penalty * remaining^2`,
  forced liquidation at the deadline.
- `execution/rl/ppo.py` — minimal PPO (torch; actor-critic MLP, GAE, clipped
  objective, entropy bonus). `RLExecution(ExecutionAlgo)` wraps a trained policy.
- `execution/learned.py` — `LearnedPolicy`: Almgren–Chriss schedule tilted by the
  Transformer's predicted `mid_move` probabilities and `queue_depletion`
  (front-loads when predicted drift is adverse; posts passively when queue is
  stable). Documented as `x_j' = x_j * (1 + kappa_tilt * signal_j)` renormalised.

Cost accounting (shared helper `execution/cost.py`): implementation shortfall
`IS = side * (arrival_mid * Q - sum(fill_price * fill_qty))` in dollars and in bps of
notional; decomposed into spread cost, temporary impact, permanent impact/drift,
and timing risk; plus fill-rate and participation rate.

---

## 6. `orderflow.eval` — Benchmark and selection (authored by the lead)

- `eval/execution_bench.py`: for each `ExecutionAlgo` in the registry, run `N`
  seeded episodes for each regime in `{calm, normal, volatile, thin}` ∪ adversarial
  regimes. Same seeds across algos (common random numbers). Report per algo:
  mean IS (bps), std, CVaR_95, worst, fill completion rate, and a paired
  comparison vs TWAP with bootstrap CI.
- `eval/prediction_bench.py`: for each registered predictor, NLL / accuracy /
  Brier / AUC per target on held-out episodes from every regime.
- `eval/adversarial.py`: adversarial `Participant`s — `Spoofer` (large fake far
  orders cancelled before fill), `MomentumIgniter`, `LiquidityWithdrawer`
  (cancels depth when exec is detected), `Frontrunner` (detects the execution's
  footprint and trades ahead). Adversaries are scored on how much they raise the
  execution cost of each algorithm.
- `eval/select.py`: the **survival rule**. Execution algos are ranked by a robust
  score `S = mean_IS_bps + 0.5 * CVaR95_IS_bps` averaged over regimes, with
  adversarial regimes weighted 1.0; an algo survives only if `completion_rate >= 0.999`
  in every regime. Predictors survive if they beat the `MarkovBaseline`/logistic
  baseline in NLL on ≥ 4/6 targets. Output: `LEADERBOARD.md` and `survivors.json`.

---

## 7. Agent architecture (`research/`, branches `agent/*`)

Six research roles, each working in its own branch off `main` and its own
`research/<role>/` directory with a `REPORT.md` (question, method, results table,
decision). Each contributes implementations by **registering** them in
`orderflow/registry.py` (decorators `@register_execution(name)`, `@register_predictor(name)`,
`@register_flow(name)`, `@register_adversary(name)`), never by editing another
role's package.

| Role | Branch | Deliverables |
|---|---|---|
| Microstructure Researcher | `agent/microstructure` | flow-model calibration, stylised-fact checks (fat tails, vol clustering, sign autocorrelation, spread/imbalance dynamics), adverse-selection study |
| Statistical Modeler | `agent/statistical` | Hawkes fits & goodness-of-fit, OFI regressions, impact-law estimation, AC parameter calibration |
| Execution Researcher | `agent/execution` | TWAP/VWAP/AC variants (aggressive vs passive), efficient frontier, learned policy |
| RL Researcher | `agent/rl` | PPO execution agent, reward shaping ablation, comparison vs AC |
| Adversarial Researcher | `agent/adversarial` | adversarial participants; stress results per algo |
| Benchmark Agent | `agent/benchmark` | benchmark configs, extra regimes, leaderboard generation |

`main` is the integration branch; the lead merges only the survivors chosen by
`eval/select.py`, records the decision in `EVALUATION.md`.

---

## 8. Scripts

- `scripts/run_sim.py --flow hawkes --regime normal --seed 0 --horizon 1800 --out artifacts/sims/`
- `scripts/calibrate.py` — fits Hawkes/GARCH/impact and AC params, writes `artifacts/params/*.json`
- `scripts/train_model.py --quick`
- `scripts/train_rl.py --quick`
- `scripts/run_benchmark.py --episodes 20 --out artifacts/bench/`
- `scripts/select.py` — writes `LEADERBOARD.md`, `survivors.json`

All scripts are deterministic given `--seed`.
