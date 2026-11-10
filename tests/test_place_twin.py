#!/usr/bin/env python3
"""Place twin: cube at A, goal at B. Skipped without mujoco."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import mujoco  # noqa: F401
    import numpy as np

    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False


@unittest.skipUnless(HAS_MUJOCO, "mujoco not installed (use Robo-arm/.venv)")
class PlaceTwinTests(unittest.TestCase):
    def test_obs_matches_place_skill_layout(self) -> None:
        from twin.place_env import HORIZON, PlaceTwin

        env = PlaceTwin(seed=1)
        obs = env.obs()
        self.assertEqual(obs.shape, (23,))
        self.assertEqual(env.horizon, 600)
        self.assertEqual(HORIZON, 600)
        cube = env.cube_pos()
        self.assertLess(abs(cube[1] - env.pick[1]), 0.03)

    def test_action_is_six_dof_including_gripper(self) -> None:
        from twin.place_env import PlaceTwin

        env = PlaceTwin(seed=2)
        before = env.gripper_opening()
        env.step([0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
        self.assertLess(env.gripper_opening(), before)

    def test_success_when_cube_is_on_pad_b_and_jaw_open(self) -> None:
        from twin.place_env import GRIPPER_OPEN, PlaceTwin

        env = PlaceTwin(seed=3)
        env.reset()
        jnt = None
        import mujoco

        jnt = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
        adr = int(env.model.jnt_qposadr[jnt])
        env.data.qpos[adr : adr + 3] = env.goal
        env.data.qpos[env.model.jnt_qposadr[env.joint_ids[5]]] = GRIPPER_OPEN
        env.data.ctrl[env.act_ids[5]] = GRIPPER_OPEN
        mujoco.mj_forward(env.model, env.data)
        self.assertTrue(env.success())

    def test_scripted_controller_runs_without_crash(self) -> None:
        from twin.place_control import scripted_place_action
        from twin.place_env import PlaceTwin

        env = PlaceTwin(seed=4)
        for _ in range(40):
            obs, _rew, done, info = env.step(scripted_place_action(env))
            self.assertEqual(obs.shape, (23,))
            self.assertIn("cube_pos", info)
            if done:
                break


if __name__ == "__main__":
    unittest.main()
