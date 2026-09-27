"""Minimal PPO for the execution environment.

Actor-critic MLP (64-64 shared trunk, softmax policy + scalar value), GAE
advantages, clipped surrogate, entropy bonus. Deliberately dependency-light —
the whole trainer fits here rather than pulling in RLlib/stable-baselines so
the reward and stopping criteria stay inspectable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh()
        )
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        return self.pi(h), self.v(h).squeeze(-1)

    def act(self, obs: np.ndarray, rng: np.random.Generator):
        with torch.no_grad():
            logits, value = self(torch.tensor(obs, dtype=torch.float32))
            dist = torch.distributions.Categorical(logits=logits)
            if rng is None:
                a = torch.argmax(dist.probs, -1)
            else:
                a = dist.sample()
        return int(a.item()), float(dist.log_prob(a).item()), float(value.item())

    def act_deterministic(self, obs: np.ndarray) -> int:
        a, _, _ = self.act(np.asarray(obs), None)
        return a


@dataclass
class Rollout:
    obs: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    logprobs: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    values: list = field(default_factory=list)
    dones: list = field(default_factory=list)

    def add(self, o, a, lp, r, v, d):
        self.obs.append(o)
        self.actions.append(a)
        self.logprobs.append(lp)
        self.rewards.append(r)
        self.values.append(v)
        self.dones.append(d)


class PPO:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        epochs: int = 4,
        seed: int = 0,
    ) -> None:
        torch.manual_seed(seed)
        self.net = ActorCritic(obs_dim, n_actions, hidden)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma, self.lam, self.clip = gamma, gae_lambda, clip
        self.ent_coef, self.vf_coef, self.epochs = ent_coef, vf_coef, epochs
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------- rollout
    def collect(self, env, n_steps: int) -> tuple[Rollout, dict]:
        buf = Rollout()
        obs, _ = env.reset()
        ep_ret, ep_len, eps = 0.0, 0, 0
        mid_episode = False
        for _ in range(n_steps):
            a, lp, v = self.net.act(np.asarray(obs), self.rng)
            obs2, r, done, trunc, _info = env.step(a)
            buf.add(np.asarray(obs, dtype=np.float32), a, lp, r, v, done)
            ep_ret += r
            ep_len += 1
            obs = obs2
            mid_episode = not (done or trunc)
            if not mid_episode:
                eps += 1
                obs, _ = env.reset()
        # bootstrap value of the trailing state (0 if the episode ended)
        last_v = 0.0
        if mid_episode:
            with torch.no_grad():
                _logits, v2 = self.net(
                    torch.tensor(np.asarray(obs), dtype=torch.float32)
                )
            last_v = float(v2.item())
        stats = {"episodes": eps, "mean_ep_len": ep_len / max(eps, 1),
                 "mean_ep_ret": ep_ret / max(eps, 1), "last_value": last_v}
        return buf, stats

    # -------------------------------------------------------------- update
    def _gae(self, buf: Rollout, last_v: float) -> tuple[np.ndarray, np.ndarray]:
        n = len(buf.rewards)
        adv = np.zeros(n)
        gae = 0.0
        vals = np.array(buf.values + [last_v])
        for i in reversed(range(n)):
            nonterm = 1.0 - float(buf.dones[i])
            delta = buf.rewards[i] + self.gamma * vals[i + 1] * nonterm - vals[i]
            gae = delta + self.gamma * self.lam * nonterm * gae
            adv[i] = gae
        ret = adv + np.array(buf.values)
        return adv, ret

    def update(self, buf: Rollout, last_v: float = 0.0) -> dict[str, float]:
        adv, ret = self._gae(buf, last_v)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        obs = torch.tensor(np.asarray(buf.obs), dtype=torch.float32)
        acts = torch.tensor(buf.actions, dtype=torch.int64)
        old_lp = torch.tensor(buf.logprobs, dtype=torch.float32)
        adv_t = torch.tensor(adv, dtype=torch.float32)
        ret_t = torch.tensor(ret, dtype=torch.float32)
        out: dict[str, float] = {}
        for _ in range(self.epochs):
            logits, v = self.net(obs)
            dist = torch.distributions.Categorical(logits=logits)
            lp = dist.log_prob(acts)
            ratio = torch.exp(lp - old_lp)
            s1 = ratio * adv_t
            s2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv_t
            pi_loss = -torch.min(s1, s2).mean()
            v_loss = ((v - ret_t) ** 2).mean()
            ent = dist.entropy().mean()
            loss = pi_loss + self.vf_coef * v_loss - self.ent_coef * ent
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
            self.opt.step()
            out = {
                "pi_loss": float(pi_loss.item()),
                "v_loss": float(v_loss.item()),
                "entropy": float(ent.item()),
            }
        return out
