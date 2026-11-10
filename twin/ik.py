"""Damped least-squares IK for the SO-101 gripper.

Position-only IK on the gripperframe site cannot grasp anything. Two reasons:

  1. The jaws arrive at whatever angle the solver happens to pick, so they
     close beside the block instead of around it. This solver also constrains
     the approach axis (the site's local +x, pointing out of the fingertips).

  2. The site is not where the jaws close. The pinch point sits ~12 mm off the
     site along the jaw axis, measured by walking the finger meshes through the
     site frame and finding the closing gap. Servo the site onto a block and
     the fingers push it away; servo the pinch and they close around it.

Everything is solved for the pinch point directly, so the Jacobian matches the
point being controlled. Returns joint deltas in the twin's action units.
"""

from __future__ import annotations

import mujoco
import numpy as np

ARM_DOF = 5
POS_GAIN = 1.6
ROT_GAIN = 1.2
ROT_WEIGHT = 0.35
DAMPING = 0.06
DOWN = np.array([0.0, 0.0, -1.0])
# Blade tips, read off the collision meshes. The fixed blade is rigid to the
# gripper body (so to the site); the moving blade swings with the jaw joint.
FIXED_TIP_SITE = np.array([0.0063, 0.0, -0.001])
MOVING_TIP_BODY = np.array([-0.01161, -0.08195, 0.0189])
MOVING_JAW_BODY = "moving_jaw_so101_v1"
# Jaw angles: the tip gap is 20 mm shut and grows ~7 cm per radian.
JAW_OPEN = 0.6
JAW_SHUT = 0.0


def _rot(env) -> np.ndarray:
    return np.array(env.data.site_xmat[env.ee_site], dtype=np.float64).reshape(3, 3)


def _moving_jaw_body(env) -> int:
    if getattr(env, "_ik_jaw_body", None) is None:
        env._ik_jaw_body = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_BODY, MOVING_JAW_BODY
        )
    return int(env._ik_jaw_body)


def blade_tips(env) -> tuple[np.ndarray, np.ndarray]:
    """World positions of the fixed and moving blade tips."""
    pos = np.array(env.data.site_xpos[env.ee_site], dtype=np.float64)
    fixed = pos + _rot(env) @ FIXED_TIP_SITE
    body = _moving_jaw_body(env)
    origin = np.array(env.data.xpos[body], dtype=np.float64)
    rot = np.array(env.data.xmat[body], dtype=np.float64).reshape(3, 3)
    return fixed, origin + rot @ MOVING_TIP_BODY


def pinch_pos(env, offset: np.ndarray | None = None) -> np.ndarray:
    """Where the jaws would close right now, in world frame.

    Midway between the two blade tips, so it tracks the jaw as it opens
    instead of assuming a fixed point that is only right at one jaw angle.
    """
    if offset is not None:
        pos = np.array(env.data.site_xpos[env.ee_site], dtype=np.float64)
        return pos + _rot(env) @ offset
    fixed, moving = blade_tips(env)
    return (fixed + moving) / 2.0


def approach_axis(env) -> np.ndarray:
    """Direction the fingers point, in world frame."""
    return _rot(env)[:, 0]


def opening_axis(env) -> np.ndarray:
    """Direction the jaw swings open, in world frame."""
    return _rot(env)[:, 2]


def _axis_error(actual: np.ndarray, want: np.ndarray) -> np.ndarray:
    """Small-angle rotation carrying `actual` onto `want`."""
    want = np.asarray(want, dtype=np.float64)
    want = want / max(1e-9, float(np.linalg.norm(want)))
    err = np.cross(actual, want)
    norm = float(np.linalg.norm(err))
    if norm < 1e-9:
        return np.zeros(3)
    angle = float(np.arccos(np.clip(float(np.dot(actual, want)), -1.0, 1.0)))
    return err / norm * angle


def _arm_dof_cols(env) -> list[int]:
    return [int(env.model.jnt_dofadr[j]) for j in env.joint_ids[:ARM_DOF]]


def ik_delta(
    env,
    target: np.ndarray,
    *,
    approach: np.ndarray | None = DOWN,
    opening: np.ndarray | None = None,
    offset: np.ndarray | None = None,
    pos_gain: float = POS_GAIN,
    rot_gain: float = ROT_GAIN,
) -> np.ndarray:
    """One IK step that drives the pinch point to `target`.

    `offset` overrides the controlled point in the site frame; pass zeros to
    drive the site itself. `approach=None` drops the orientation constraint.
    `opening` additionally fixes the roll, which is what decides whether the
    jaw straddles a block on the table or tries to reach under it.
    """
    point = pinch_pos(env, offset)
    body = int(env.model.site_bodyid[env.ee_site])
    jacp = np.zeros((3, env.model.nv), dtype=np.float64)
    jacr = np.zeros((3, env.model.nv), dtype=np.float64)
    mujoco.mj_jac(env.model, env.data, jacp, jacr, point, body)
    cols = _arm_dof_cols(env)

    rows = [jacp[:, cols]]
    res = [pos_gain * (np.asarray(target, dtype=np.float64) - point)]

    rot_err = np.zeros(3)
    if approach is not None:
        rot_err = rot_err + _axis_error(approach_axis(env), approach)
    if opening is not None:
        rot_err = rot_err + _axis_error(opening_axis(env), opening)
    if approach is not None or opening is not None:
        rows.append(ROT_WEIGHT * jacr[:, cols])
        res.append(ROT_WEIGHT * rot_gain * rot_err)

    jac = np.vstack(rows)
    residual = np.concatenate(res)
    # Damped least squares keeps the solution finite near singularities.
    # Joints on their stops are deliberately left in the Jacobian: letting the
    # wrist saturate is how the arm trades a straight-down grip for the extra
    # height it needs to carry a block over the fence.
    lhs = jac.T @ jac + (DAMPING**2) * np.eye(len(cols))
    dq = np.linalg.solve(lhs, jac.T @ residual)
    mag = float(np.linalg.norm(dq))
    if mag > 1.0:
        dq = dq / mag
    return np.clip(dq, -1.0, 1.0)


def jaw_delta(env, want: float) -> float:
    """Action component that drives the jaw toward `want`."""
    return float(np.clip((want - env.gripper_opening()) / 0.4, -1.0, 1.0))
