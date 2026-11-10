#!/usr/bin/env python3
"""Barrier twin: pick at A, clear the fence, place at B. Skipped without mujoco."""

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
class BarrierTwinTests(unittest.TestCase):
    def test_obs_matches_barrier_skill_layout(self) -> None:
        from twin.barrier_env import HORIZON, BarrierTwin

        env = BarrierTwin(seed=1)
        self.assertEqual(env.obs().shape, (29,))
        self.assertEqual(env.horizon, HORIZON)

    def test_reset_varies_fence_and_goal(self) -> None:
        from twin.barrier_env import BarrierTwin

        env = BarrierTwin(seed=2)
        tops, goals, picks = set(), set(), set()
        for _ in range(8):
            env.reset()
            tops.add(round(env.wall_top(), 4))
            goals.add(tuple(np.round(env.goal, 4)))
            picks.add(tuple(np.round(env.pick, 4)))
        self.assertGreater(len(tops), 1)
        self.assertGreater(len(goals), 1)
        self.assertGreater(len(picks), 1)

    def test_fence_is_between_start_and_goal(self) -> None:
        from twin.barrier_env import BarrierTwin

        env = BarrierTwin(seed=3)
        for _ in range(5):
            env.reset()
            wall_x = float(env.wall_pos()[0])
            self.assertLess(env.pick[0], wall_x)
            self.assertGreater(env.goal[0], wall_x)

    def test_touching_the_fence_fails_the_episode(self) -> None:
        from twin.barrier_env import BarrierTwin

        env = BarrierTwin(seed=4)
        env.reset()
        # Shove the block straight through the fence line.
        jnt = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
        adr = int(env.model.jnt_qposadr[jnt])
        env.data.qpos[adr : adr + 3] = [float(env.wall_pos()[0]), 0.0, env.pick[2]]
        mujoco.mj_forward(env.model, env.data)
        _obs, _rew, done, info = env.step(np.zeros(6))
        self.assertTrue(info["collision"])
        self.assertTrue(done)
        self.assertFalse(env.success())
        self.assertEqual(info["reason"], "collision:cube")

    def test_block_at_goal_with_clean_fence_is_a_pass(self) -> None:
        from twin.barrier_env import GRIPPER_OPEN, BarrierTwin

        env = BarrierTwin(seed=5)
        env.reset()
        jnt = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
        adr = int(env.model.jnt_qposadr[jnt])
        env.data.qpos[adr : adr + 3] = env.goal
        env.data.qpos[env.model.jnt_qposadr[env.joint_ids[5]]] = GRIPPER_OPEN
        env.data.ctrl[env.act_ids[5]] = GRIPPER_OPEN
        mujoco.mj_forward(env.model, env.data)
        self.assertTrue(env.success())

    def test_scripted_controller_clears_the_fence(self) -> None:
        from twin.barrier_control import scripted_barrier_action
        from twin.barrier_env import BarrierTwin

        env = BarrierTwin(seed=6)
        env.reset()
        peak = 0.0
        info: dict = {}
        for _ in range(env.horizon):
            _obs, _rew, done, info = env.step(scripted_barrier_action(env))
            peak = max(peak, float(env.cube_pos()[2]))
            if done:
                break
        self.assertTrue(info.get("success"), info.get("reason"))
        self.assertFalse(info["collision"])
        self.assertGreater(peak, env.wall_top())

    def test_naive_straight_across_always_hits_the_fence(self) -> None:
        # Negative control: if dragging the block straight to the goal ever
        # passes, the fence rule has stopped biting and the exam is worthless.
        from twin.barrier_control import naive_barrier_action
        from twin.barrier_env import BarrierTwin

        for seed in range(6):
            env = BarrierTwin(seed=seed)
            env.reset()
            info: dict = {}
            for _ in range(env.horizon):
                _obs, _rew, done, info = env.step(naive_barrier_action(env))
                if done:
                    break
            self.assertFalse(info.get("success"))
            self.assertTrue(info.get("collision"), f"seed {seed}: {info.get('reason')}")

    def test_gym_wrapper_exposes_barrier(self) -> None:
        from twin.gym_env import TwinGym

        env = TwinGym(skill="barrier", seed=7)
        obs, _info = env.reset()
        self.assertEqual(obs.shape, (29,))
        self.assertEqual(env.action_space.shape, (6,))
        obs, _rew, _term, _trunc, _info = env.step(env.action_space.sample() * 0.0)
        self.assertEqual(obs.shape, (29,))


if __name__ == "__main__":
    unittest.main()
