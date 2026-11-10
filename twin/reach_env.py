"""SO-101 reach task on the official MuJoCo model.

Matches skills/reach/skill.yaml:
  obs = joint_pos(6) + joint_vel(6) + ee_pos(3) + target_pos(3) + ee_to_target(1)
  action = joint_position_delta(5)  # gripper held
  horizon_steps = 300
  success = EE within 2 cm of the target
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

REPO = repo_root()
SCENE = REPO / "urdf" / "so101" / "reach_twin.xml"

JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
ARM_JOINTS = JOINT_NAMES[:-1]
SUCCESS_M = 0.02
HORIZON = 300
CONTROL_HZ = 30.0
ACTION_SCALE = 0.15
WORKSPACE = {
    "x": (-0.25, 0.25),
    "y": (-0.25, 0.25),
    "z": (0.05, 0.30),
}


def _jid(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    idx = mujoco.mj_name2id(model, kind, name)
    if idx < 0:
        raise KeyError(f"{kind} {name!r} not in model")
    return idx


class ReachTwin:
    """One MuJoCo reach episode at a time. No Isaac, no GPU."""

    def __init__(self, *, scene: Path | None = None, seed: int = 0) -> None:
        path = Path(scene or SCENE)
        if not path.is_file():
            raise FileNotFoundError(path)
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self.joint_ids = [_jid(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
        self.act_ids = [_jid(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in JOINT_NAMES]
        self.ee_site = _jid(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
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
        xyz = np.array(
            [
                self.rng.uniform(*WORKSPACE["x"]),
                self.rng.uniform(*WORKSPACE["y"]),
                self.rng.uniform(*WORKSPACE["z"]),
            ],
            dtype=np.float64,
        )
        return xyz

    def reset(self, *, target: np.ndarray | None = None) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        self.steps = 0
        noise = self.rng.uniform(-0.05, 0.05, size=5)
        for i, jid in enumerate(self.joint_ids[:5]):
            qadr = self.model.jnt_qposadr[jid]
            lo, hi = self.model.jnt_range[jid]
            self.data.qpos[qadr] = float(np.clip(noise[i], lo, hi))
        self.data.qpos[self.model.jnt_qposadr[self.joint_ids[5]]] = 0.0
        for i, aid in enumerate(self.act_ids[:5]):
            self.data.ctrl[aid] = float(self.data.qpos[self.model.jnt_qposadr[self.joint_ids[i]]])
        self.data.ctrl[self.act_ids[5]] = 0.0
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
        if act.shape[0] != 5:
            raise ValueError(f"action must be length 5, got {act.shape}")
        for i, aid in enumerate(self.act_ids[:5]):
            jid = self.joint_ids[i]
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
