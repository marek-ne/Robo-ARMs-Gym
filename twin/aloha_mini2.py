"""Left-arm reach on AlohaMini2.

The official URDF is a dual-arm mobile robot. This gym locks the base, the
lift, the right arm, and the left gripper, and trains the six left-arm joints.
Meshes that exceed MuJoCo's STL face cap are decimated in place.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import mujoco
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "MuJoCo is required for the digital twin. Install the project dependencies."
    ) from exc

from twin.paths import repo_root

URDF = repo_root() / "urdf" / "alohamini2" / "urdf" / "alohamini2.urdf"

ARM_JOINTS = (
    "left_shoulder_pan",
    "left_shoulder_lift",
    "left_elbow_flex",
    "left_wrist_flex",
    "left_wrist_yaw_joint",
    "left_wrist_roll",
)
LOCK_JOINTS = (
    "root_x_axis_joint",
    "root_y_axis_joint",
    "root_z_rotation_joint",
    "vertical_move",
    "left_gripper",
    "right_shoulder_pan",
    "right_shoulder_lift",
    "right_elbow_flex",
    "right_wrist_flex",
    "right_wrist_yaw_joint",
    "right_wrist_roll",
    "right_gripper",
)

SUCCESS_M = 0.02
HORIZON = 300
CONTROL_HZ = 30.0
ACTION_SCALE = 0.12
KP = 80.0


def build_spec(urdf: Path | None = None) -> mujoco.MjSpec:
    path = Path(urdf or URDF)
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = mujoco.MjSpec.from_file(str(path))
    spec.modelname = "alohamini2_left_reach"
    spec.compiler.boundmass = 0.05
    spec.compiler.boundinertia = 1e-4
    spec.compiler.inertiafromgeom = mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_TRUE
    spec.compiler.discardvisual = False
    for name in LOCK_JOINTS:
        spec.delete(spec.joint(name))

    site = spec.body("left_Fixed_Jaw").add_site()
    site.name = "left_ee"

    target = spec.worldbody.add_body()
    target.name = "target"
    target.mocap = True
    marker = target.add_geom()
    marker.name = "target_sphere"
    marker.type = mujoco.mjtGeom.mjGEOM_SPHERE
    marker.size[0] = 0.015
    marker.rgba[:] = (0.25, 0.55, 0.95, 0.85)
    marker.contype = 0
    marker.conaffinity = 0

    for name in ARM_JOINTS:
        act = spec.add_actuator()
        act.name = name
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.target = name
        act.dyntype = mujoco.mjtDyn.mjDYN_NONE
        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        act.gainprm[0] = KP
        act.biasprm[1] = -KP
        act.ctrlrange[:] = (-3.14, 3.14)
        act.forcerange[:] = (-30.0, 30.0)
    return spec


def compile_model(urdf: Path | None = None) -> mujoco.MjModel:
    return build_spec(urdf).compile()


class AlohaMini2Reach:
    """Reach with the left AlohaMini2 arm. Base, lift, and right arm stay fixed."""

    def __init__(self, *, urdf: Path | None = None, seed: int = 0) -> None:
        self.model = compile_model(urdf)
        self.data = mujoco.MjData(self.model)
        self.joint_ids = [
            _jid(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINTS
        ]
        self.act_ids = [
            _jid(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in ARM_JOINTS
        ]
        self.ee_site = _jid(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_ee")
        self.nsub = max(1, int(round((1.0 / CONTROL_HZ) / float(self.model.opt.timestep))))
        self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.target = np.zeros(3, dtype=np.float64)
        self.reset()

    @property
    def horizon(self) -> int:
        return HORIZON

    def ee_pos(self) -> np.ndarray:
        return np.array(self.data.site_xpos[self.ee_site], dtype=np.float64)

    def _set_target(self, xyz: np.ndarray) -> None:
        self.target = np.array(xyz, dtype=np.float64)
        if self.model.nmocap >= 1:
            self.data.mocap_pos[0] = self.target

    def sample_target(self) -> np.ndarray:
        """A point the left arm can reach, taken from a random joint pose."""
        saved_q = self.data.qpos.copy()
        saved_ctrl = self.data.ctrl.copy()
        for _ in range(8):
            for jid in self.joint_ids:
                lo, hi = self.model.jnt_range[jid]
                mid = 0.5 * (lo + hi)
                half = 0.35 * (hi - lo)
                self.data.qpos[self.model.jnt_qposadr[jid]] = float(
                    self.rng.uniform(mid - half, mid + half)
                )
            mujoco.mj_forward(self.model, self.data)
            target = self.ee_pos().copy()
            self.data.qpos[:] = saved_q
            self.data.ctrl[:] = saved_ctrl
            mujoco.mj_forward(self.model, self.data)
            if np.linalg.norm(target - self.ee_pos()) > 0.05:
                return target
        return target

    def reset(self, *, target: np.ndarray | None = None) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        self.steps = 0
        for aid in self.act_ids:
            self.data.ctrl[aid] = 0.0
        self._set_target(target if target is not None else self.sample_target())
        mujoco.mj_forward(self.model, self.data)
        return self.obs()

    def obs(self) -> np.ndarray:
        qpos = np.array(
            [self.data.qpos[self.model.jnt_qposadr[j]] for j in self.joint_ids],
            dtype=np.float64,
        )
        qvel = np.array(
            [self.data.qvel[self.model.jnt_dofadr[j]] for j in self.joint_ids],
            dtype=np.float64,
        )
        ee = self.ee_pos()
        dist = np.array([np.linalg.norm(ee - self.target)], dtype=np.float64)
        return np.concatenate([qpos, qvel, ee, self.target, dist])

    def distance(self) -> float:
        return float(np.linalg.norm(self.ee_pos() - self.target))

    def success(self) -> bool:
        return self.distance() < SUCCESS_M

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        act = np.asarray(action, dtype=np.float64).reshape(-1)
        if act.shape[0] != len(ARM_JOINTS):
            raise ValueError(f"action must be length {len(ARM_JOINTS)}, got {act.shape}")
        for i, aid in enumerate(self.act_ids):
            lo, hi = self.model.actuator_ctrlrange[aid]
            cur = float(self.data.ctrl[aid])
            self.data.ctrl[aid] = float(np.clip(cur + ACTION_SCALE * act[i], lo, hi))
        for _ in range(self.nsub):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        dist = self.distance()
        done = self.success() or self.steps >= HORIZON
        reward = -dist + (10.0 if self.success() else 0.0)
        return self.obs(), float(reward), done, {"distance_m": dist, "success": self.success()}


def scripted_aloha_action(env: AlohaMini2Reach) -> np.ndarray:
    """Joint delta from the site Jacobian, scaled into [-1, 1]."""
    model, data = env.model, env.data
    jac = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacSite(model, data, jac, None, env.ee_site)
    err = env.target - env.ee_pos()
    gram = jac @ jac.T + 1e-3 * np.eye(3)
    dq = jac.T @ np.linalg.solve(gram, err)
    action = np.zeros(len(ARM_JOINTS), dtype=np.float64)
    for i, jid in enumerate(env.joint_ids):
        action[i] = dq[model.jnt_dofadr[jid]] / ACTION_SCALE
    return np.clip(action, -1.0, 1.0)


def _jid(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    idx = mujoco.mj_name2id(model, kind, name)
    if idx < 0:
        raise KeyError(f"{name!r} not in model")
    return idx
