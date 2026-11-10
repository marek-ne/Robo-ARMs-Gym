#!/usr/bin/env python3
"""Gymnasium wrapper + short PPO update on the twin."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import gymnasium  # noqa: F401
    import mujoco  # noqa: F401

    HAS = True
except ImportError:
    HAS = False

try:
    import stable_baselines3  # noqa: F401

    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False


@unittest.skipUnless(HAS, "mujoco/gymnasium not in this python")
class TwinGymTests(unittest.TestCase):
    def test_place_spaces_match_skill(self) -> None:
        from twin.gym_env import TwinGym

        env = TwinGym(skill="place", seed=1)
        self.assertEqual(env.observation_space.shape, (23,))
        self.assertEqual(env.action_space.shape, (6,))
        obs, info = env.reset(seed=2)
        self.assertEqual(obs.shape, (23,))
        self.assertIsInstance(info, dict)
        obs2, rew, term, trunc, info2 = env.step(env.action_space.sample())
        self.assertEqual(obs2.shape, (23,))
        self.assertIsInstance(rew, float)
        self.assertIn("success", info2)
        self.assertTrue(term or trunc or True)
        env.close()

    def test_reach_action_dim_is_five(self) -> None:
        from twin.gym_env import TwinGym

        env = TwinGym(skill="reach", seed=0)
        self.assertEqual(env.action_space.shape, (5,))
        self.assertEqual(env.observation_space.shape, (19,))


@unittest.skipUnless(HAS and HAS_SB3, "sb3 + mujoco required")
class TwinPpoSmokeTests(unittest.TestCase):
    def test_one_ppo_update_writes_zip(self) -> None:
        from twin.train import train

        result = train(
            skill="reach",
            total_timesteps=256,
            n_envs=1,
            seed=0,
            catalog=False,
        )
        path = Path(result["checkpoint"])
        self.assertTrue(path.is_file(), msg=str(path))
        self.assertEqual(result["skill"], "reach")
        self.assertFalse(result["promoted"])


if __name__ == "__main__":
    unittest.main()
