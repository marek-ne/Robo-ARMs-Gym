"""AlohaMini2 left-arm reach. Skipped when MuJoCo or the URDF is missing."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import mujoco  # noqa: F401

    HAS = (ROOT / "urdf" / "alohamini2" / "urdf" / "alohamini2.urdf").is_file()
except ImportError:
    HAS = False


@unittest.skipUnless(HAS, "mujoco or AlohaMini2 URDF missing")
class AlohaMini2ReachTests(unittest.TestCase):
    def test_left_arm_only_and_obs_layout(self) -> None:
        from twin.aloha_mini2 import ARM_JOINTS, AlohaMini2Reach

        env = AlohaMini2Reach(seed=1)
        self.assertEqual(env.model.njnt, len(ARM_JOINTS))
        self.assertEqual(env.obs().shape, (19,))
        names = [
            mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(env.model.njnt)
        ]
        self.assertEqual(tuple(names), ARM_JOINTS)
        self.assertNotIn("right_shoulder_pan", names)
        self.assertNotIn("vertical_move", names)

    def test_step_moves_the_arm(self) -> None:
        from twin.aloha_mini2 import AlohaMini2Reach

        env = AlohaMini2Reach(seed=2)
        before = env.ee_pos().copy()
        obs, reward, done, info = env.step([0.8, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(obs.shape, (19,))
        self.assertIsInstance(reward, float)
        self.assertIn("success", info)
        self.assertFalse(done)
        self.assertGreater(float(np_norm(env.ee_pos() - before)), 1e-4)

    def test_scripted_action_is_six(self) -> None:
        from twin.aloha_mini2 import AlohaMini2Reach, scripted_aloha_action

        env = AlohaMini2Reach(seed=3)
        action = scripted_aloha_action(env)
        self.assertEqual(action.shape, (6,))
        self.assertTrue((action >= -1.0).all() and (action <= 1.0).all())


def np_norm(xyz) -> float:
    import numpy as np

    return float(np.linalg.norm(xyz))


if __name__ == "__main__":
    unittest.main()
