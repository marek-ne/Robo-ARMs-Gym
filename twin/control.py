"""Jacobian P-controller for the reach twin. Real physics, not a planted success rate."""

from __future__ import annotations

import mujoco
import numpy as np

from twin.reach_env import ReachTwin


def jacobian_p_action(env: ReachTwin, *, gain: float = 1.8, eps: float = 1e-4) -> np.ndarray:
    """5-dim action that reduces EE error in Cartesian space."""
    ee0 = env.ee_pos().copy()
    err = env.target - ee0
    q0 = np.array(
        [env.data.qpos[env.model.jnt_qposadr[j]] for j in env.joint_ids[:5]],
        dtype=np.float64,
    )
    jac = np.zeros((3, 5), dtype=np.float64)
    for i in range(5):
        env.data.qpos[env.model.jnt_qposadr[env.joint_ids[i]]] = q0[i] + eps
        mujoco.mj_forward(env.model, env.data)
        jac[:, i] = (env.ee_pos() - ee0) / eps
        env.data.qpos[env.model.jnt_qposadr[env.joint_ids[i]]] = q0[i]
    mujoco.mj_forward(env.model, env.data)
    dq, *_ = np.linalg.lstsq(jac, gain * err, rcond=None)
    mag = float(np.linalg.norm(dq))
    if mag > 1.0:
        dq = dq / mag
    return np.clip(dq, -1.0, 1.0)
