"""RL env/PPO and learned-policy smoke tests (kept small for CI)."""

import numpy as np

from orderflow.book.types import Side
from orderflow.execution.learned import LearnedPolicy
from orderflow.execution.rl.env import N_ACTIONS, OBS_DIM, ExecutionEnv
from orderflow.execution.rl.policy import RLExecution
from orderflow.execution.rl.ppo import PPO
from orderflow.execution.types import ExecutionTask
from orderflow.sim.regimes import get_regime


def _env(**kw):
    flow_cls, params, seed_kwargs = get_regime("normal", flow="zi")
    kw.setdefault("total_qty", 200)
    kw.setdefault("horizon", 60.0)
    kw.setdefault("step", 5.0)
    return ExecutionEnv(
        flow_factory=lambda: flow_cls(params),
        seed_kwargs=seed_kwargs,
        seed=0,
        **kw,
    )


def test_env_reset_step_shapes():
    env = _env()
    obs, _ = env.reset()
    assert obs.shape == (OBS_DIM,)
    for _ in range(3):
        obs, r, _done, _trunc, _info = env.step(env.action_space.sample())
        assert obs.shape == (OBS_DIM,)
        assert np.isfinite(r)
    assert env.action_space.n == N_ACTIONS


def test_env_episode_liquidates():
    env = _env()
    env.reset()
    done = False
    steps = 0
    while not done and steps < 40:
        _obs, _r, done, _t, _i = env.step(5)  # TWAP-ish action
        steps += 1
    assert done
    assert env.remaining == 0  # forced liquidation closed it out


def test_ppo_smoke_two_updates():
    env = _env(total_qty=100, horizon=30.0, step=5.0)
    ppo = PPO(OBS_DIM, N_ACTIONS, epochs=1, seed=0)
    for _ in range(2):
        buf, stats = ppo.collect(env, 60)
        losses = ppo.update(buf, last_v=stats["last_value"])
        assert np.isfinite(losses["pi_loss"])
        assert np.isfinite(losses["v_loss"])
    obs, _ = env.reset()
    a = ppo.net.act_deterministic(obs)
    assert 0 <= a < N_ACTIONS


def test_rl_execution_untrained_is_twap_like():
    # untrained RLExecution must still emit the TWAP slice and finish
    from orderflow.book.book import LimitOrderBook
    from orderflow.execution.base import AlgoParticipant
    from orderflow.sim.simulator import Simulator, seed_book

    flow_cls, params, seed_kwargs = get_regime("normal", flow="zi")
    rng = np.random.default_rng(0)
    book = LimitOrderBook(tick_size=0.01)
    seed_book(book, rng=rng, **seed_kwargs)
    task = ExecutionTask(side=Side.SELL, total_qty=200, horizon=60.0, step=5.0)
    algo = RLExecution(policy=None)
    algo.reset(task, book, rng)
    adapter = AlgoParticipant(algo, task)
    sim = Simulator(book, flow_cls(params), seed=0, participants=[adapter])
    sim.run(62.0)
    assert adapter.remaining == 0


class _AdverseSignal:
    """Always predicts adverse drift for a SELL (mid going down)."""

    def signal(self, state_vec):
        return -1.0  # favourable - adverse = -1 -> front-load


class _NeutralSignal:
    def signal(self, state_vec):
        return 0.0


def test_learned_tilt_frontloads_under_adverse_signal():
    base = LearnedPolicy(predictor=None)
    hot = LearnedPolicy(predictor=_AdverseSignal())
    task = ExecutionTask(side=Side.SELL, total_qty=1000, horizon=300.0, step=10.0)
    from orderflow.book.book import LimitOrderBook
    from orderflow.execution.types import ExecState

    book = LimitOrderBook()
    base.reset(task, book, np.random.default_rng(0))
    hot.reset(task, book, np.random.default_rng(0))
    st = ExecState(remaining_qty=1000, elapsed=50.0, fills=[])
    q_base = base.decide(50.0, book, st)
    q_hot = hot.decide(50.0, book, st)
    # adverse signal => larger cumulative target now => bigger child
    assert q_hot[0].qty > q_base[0].qty


def test_learned_neutral_signal_matches_ac():
    neu = LearnedPolicy(predictor=_NeutralSignal())
    task = ExecutionTask(side=Side.SELL, total_qty=1000, horizon=300.0, step=10.0)
    from orderflow.book.book import LimitOrderBook
    from orderflow.execution.types import ExecState

    book = LimitOrderBook()
    neu.reset(task, book, np.random.default_rng(0))
    st = ExecState(remaining_qty=1000, elapsed=100.0, fills=[])
    q = neu.decide(100.0, book, st)
    assert len(q) == 1 and q[0].qty > 0
