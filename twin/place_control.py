"""Waypoint controller for PlaceTwin. Proves the room works; not a certified policy.

Same shape as the barrier controller, minus the fence: hover over A, sink onto
the block in small steps, shut the jaw, carry it across, set it down, let go.
Targets are pinch points, so the jaws close around the block instead of beside
it, and the roll is pinned so they straddle a cube face rather than a corner.
"""

from __future__ import annotations

import numpy as np

from twin import ik
from twin.place_env import PlaceTwin

HOVER_H = 0.075
CARRY_H = 0.055
RELEASE_H = 0.006
JAW_APPROACH = 0.45
# Jaws swing open along world +y, so they meet the cube's flat +/-y faces.
STRADDLE = np.array([0.0, 1.0, 0.0])


def _plan(env: PlaceTwin) -> list[tuple[np.ndarray, float, int]]:
    a = env.pick.copy()
    b = env.goal.copy()
    plan: list[tuple[np.ndarray, float, int]] = [
        (a + np.array([0.0, 0.0, HOVER_H]), JAW_APPROACH, 60),
    ]
    for z in (0.05, 0.036, 0.025, 0.016, 0.008, 0.0):
        plan.append((a + np.array([0.0, 0.0, z]), JAW_APPROACH, 14))
    plan.append((a, ik.JAW_SHUT, 40))
    plan.append((a + np.array([0.0, 0.0, CARRY_H]), ik.JAW_SHUT, 45))
    plan.append((b + np.array([0.0, 0.0, CARRY_H]), ik.JAW_SHUT, 70))
    plan.append((b + np.array([0.0, 0.0, RELEASE_H]), ik.JAW_SHUT, 55))
    plan.append((b + np.array([0.0, 0.0, RELEASE_H]), ik.JAW_OPEN, 30))
    plan.append((b + np.array([0.0, 0.0, HOVER_H]), ik.JAW_OPEN, 40))
    return plan


def scripted_place_action(env: PlaceTwin) -> np.ndarray:
    """Open -> hover A -> sink -> close -> carry -> set down at B -> release."""
    plan = _plan(env)
    t = env.steps
    for target, jaw, hold in plan:
        if t < hold:
            break
        t -= hold
    else:
        target, jaw = plan[-1][0], plan[-1][1]
    arm = ik.ik_delta(env, np.asarray(target, dtype=np.float64), approach=ik.DOWN,
                      opening=STRADDLE)
    return np.concatenate([arm, [ik.jaw_delta(env, jaw)]])


def plan_length(env: PlaceTwin) -> int:
    return sum(hold for _, _, hold in _plan(env))
