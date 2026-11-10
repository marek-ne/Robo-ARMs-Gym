"""Waypoint controller for BarrierTwin. Proves the room works; not a certified policy.

Top-down grasp on the can, climb above the fence, cross, set down, release.
Used to check the scene is solvable and to record teaching demonstrations.
"""

from __future__ import annotations

import numpy as np

from twin import ik
from twin.barrier_env import BLOCK_HALF_H, BarrierTwin

READY = np.array([0.24, 0.0, 0.20])
HOVER_H = 0.08
CROSS_MARGIN = 0.022
RELEASE_H = 0.004
JAW_APPROACH = 0.45


def _plan(env: BarrierTwin) -> list[tuple[np.ndarray, float, int]]:
    """(pinch target, jaw, hold steps) waypoints for one episode."""
    block = env.pick.copy()
    goal = env.goal.copy()
    cross_z = env.wall_top() + BLOCK_HALF_H + CROSS_MARGIN
    wall_x = float(env.wall_pos()[0])

    plan: list[tuple[np.ndarray, float, int]] = [
        (READY, JAW_APPROACH, 55),
        (block + np.array([0.0, 0.0, HOVER_H]), JAW_APPROACH, 55),
    ]
    for z in (0.055, 0.04, 0.028, 0.018, 0.009, 0.0):
        plan.append((block + np.array([0.0, 0.0, z]), JAW_APPROACH, 14))
    plan.append((block, ik.JAW_SHUT, 35))
    plan.append((np.array([block[0], block[1], cross_z]), ik.JAW_SHUT, 45))
    plan.append((np.array([wall_x, goal[1], cross_z]), ik.JAW_SHUT, 45))
    plan.append((np.array([goal[0], goal[1], cross_z]), ik.JAW_SHUT, 40))
    plan.append((goal + np.array([0.0, 0.0, RELEASE_H]), ik.JAW_SHUT, 55))
    plan.append((goal + np.array([0.0, 0.0, RELEASE_H]), ik.JAW_OPEN, 30))
    plan.append((goal + np.array([0.0, 0.0, HOVER_H]), ik.JAW_OPEN, 40))
    return plan


def scripted_barrier_action(env: BarrierTwin) -> np.ndarray:
    """Action for the current step, following the waypoint plan."""
    plan = _plan(env)
    t = env.steps
    for target, jaw, hold in plan:
        if t < hold:
            break
        t -= hold
    else:
        target, jaw = plan[-1][0], plan[-1][1]
    arm = ik.ik_delta(env, np.asarray(target, dtype=np.float64), approach=ik.DOWN)
    return np.concatenate([arm, [ik.jaw_delta(env, jaw)]])


def plan_length(env: BarrierTwin) -> int:
    return sum(hold for _, _, hold in _plan(env))


def _naive_plan(env: BarrierTwin) -> list[tuple[np.ndarray, float, int]]:
    """Same pick, then straight across at carrying height. Hits the fence."""
    block = env.pick.copy()
    goal = env.goal.copy()
    plan: list[tuple[np.ndarray, float, int]] = [
        (READY, JAW_APPROACH, 55),
        (block + np.array([0.0, 0.0, HOVER_H]), JAW_APPROACH, 55),
    ]
    for z in (0.055, 0.04, 0.028, 0.018, 0.009, 0.0):
        plan.append((block + np.array([0.0, 0.0, z]), JAW_APPROACH, 14))
    plan.append((block, ik.JAW_SHUT, 35))
    plan.append((block + np.array([0.0, 0.0, 0.012]), ik.JAW_SHUT, 30))
    plan.append((goal + np.array([0.0, 0.0, 0.012]), ik.JAW_SHUT, 120))
    plan.append((goal + np.array([0.0, 0.0, RELEASE_H]), ik.JAW_OPEN, 40))
    return plan


def naive_barrier_action(env: BarrierTwin) -> np.ndarray:
    """The obvious wrong answer: pick it up and slide it straight to the goal.

    Kept as the negative control for the fence rule. If this ever passes, the
    collision check has stopped working and the exam means nothing.
    """
    plan = _naive_plan(env)
    t = env.steps
    for target, jaw, hold in plan:
        if t < hold:
            break
        t -= hold
    else:
        target, jaw = plan[-1][0], plan[-1][1]
    arm = ik.ik_delta(env, np.asarray(target, dtype=np.float64), approach=ik.DOWN)
    return np.concatenate([arm, [ik.jaw_delta(env, jaw)]])
