#!/usr/bin/env python3
"""CPU MuJoCo digital twin — skipped when mujoco is not installed."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import mujoco  # noqa: F401

    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False


@unittest.skipUnless(HAS_MUJOCO, "mujoco not installed (use Robo-arm/.venv)")
class ReachTwinTests(unittest.TestCase):
    def test_obs_matches_skill_yaml_layout(self) -> None:
        from twin.reach_env import HORIZON, ReachTwin

        env = ReachTwin(seed=1)
        obs = env.obs()
        self.assertEqual(obs.shape, (19,))
        self.assertEqual(env.horizon, 300)
        self.assertEqual(HORIZON, 300)
        self.assertEqual(env.distance(), float(obs[-1]))

    def test_step_changes_state(self) -> None:
        from twin.reach_env import ReachTwin

        env = ReachTwin(seed=2)
        before = env.obs().copy()
        obs, reward, done, info = env.step([0.2, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(obs.shape, (19,))
        self.assertIsInstance(reward, float)
        self.assertIn("distance_m", info)
        self.assertEqual(env.steps, 1)
        self.assertFalse((obs == before).all())

    def test_jacobian_reduces_distance_on_nearby_target(self) -> None:
        import numpy as np

        from twin.control import jacobian_p_action
        from twin.reach_env import ReachTwin

        env = ReachTwin(seed=3)
        env.reset()
        nearby = env.ee_pos() + np.array([0.04, 0.0, 0.02])
        env._set_target(nearby)
        d0 = env.distance()
        for _ in range(80):
            _, _, done, _ = env.step(jacobian_p_action(env))
            if done:
                break
        self.assertLess(env.distance(), d0)


if __name__ == "__main__":
    unittest.main()
