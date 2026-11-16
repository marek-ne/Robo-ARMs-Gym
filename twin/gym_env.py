"""Gymnasium wrappers around the CPU MuJoCo twins.

One env = one skill contract. Reach is 19-dim obs / 5-dim action.
Place is 23-dim obs / 6-dim action (gripper included).
Barrier is 29-dim obs / 6-dim action (place plus the fence pose).
"""

from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "gymnasium is required to train the twin. Install the project dependencies."
    ) from exc


def make_twin(skill: str, seed: int = 0):
    if skill == "barrier":
        from twin.barrier_env import BarrierTwin

        return BarrierTwin(seed=seed)
    if skill == "place":
        from twin.place_env import PlaceTwin

        return PlaceTwin(seed=seed)
    if skill == "reach":
        from twin.reach_env import ReachTwin

        return ReachTwin(seed=seed)
    if skill == "aloha-reach":
        from twin.aloha_mini2 import AlohaMini2Reach

        return AlohaMini2Reach(seed=seed)
    raise ValueError(f"unknown skill {skill!r} (use reach, place, barrier, or aloha-reach)")


class TwinGym(gym.Env):
    """Single-process Gymnasium env for Stable-Baselines3."""

    metadata = {"render_modes": []}

    def __init__(self, skill: str = "place", seed: int = 0) -> None:
        super().__init__()
        self.skill = skill
        self._inner = make_twin(skill, seed=seed)
        obs = np.asarray(self._inner.obs(), dtype=np.float32)
        n_act = 5 if skill == "reach" else 6
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(n_act,), dtype=np.float32
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._inner.rng = np.random.default_rng(seed)
        obs = np.asarray(self._inner.reset(), dtype=np.float32)
        return obs, {}

    def step(self, action):
        obs, rew, done, info = self._inner.step(action)
        from twin.rewards import shaped_reward

        obs_arr = np.asarray(obs, dtype=np.float32)
        rew = shaped_reward(self.skill, obs_arr, np.asarray(action, dtype=np.float32), info, float(rew))
        success = bool(info.get("success"))
        terminated = success
        truncated = bool(done) and not success
        return (
            np.asarray(obs, dtype=np.float32),
            float(rew),
            terminated,
            truncated,
            info,
        )

    def close(self) -> None:
        return None
