"""Pick-at-A / place-at-B on the official SO-101 MuJoCo model.

Matches skills/place/skill.yaml:
  obs = joint_pos(6)+joint_vel(6)+ee(3)+cube(3)+goal(3)+gripper(1)+cube_to_goal(1)
  action = joint_position_delta(6)  # gripper included
  success = cube within 2 cm of B, on the table, jaw open
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
SCENE = REPO / "urdf" / "so101" / "place_twin.xml"

JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
SUCCESS_M = 0.02
HORIZON = 600
CONTROL_HZ = 30.0
ACTION_SCALE = 0.15
GRIPPER_SCALE = 0.25
# Table pads (world frame), match place_twin.xml
POSE_A = np.array([0.20, 0.09, 0.034], dtype=np.float64)
POSE_B = np.array([0.20, -0.09, 0.034], dtype=np.float64)
TABLE_Z = 0.022
GRIPPER_OPEN = 1.2
GRIPPER_CLOSED = 0.0
OPEN_THRESH = 0.4


def _jid(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    idx = mujoco.mj_name2id(model, kind, name)
    if idx < 0:
        raise KeyError(f"{kind} {name!r} not in model")
    return idx


class PlaceTwin:
    """One pick-and-place episode. Cube starts on pad A. Goal is pad B."""

    def __init__(self, *, scene: Path | None = None, seed: int = 0) -> None:
        path = Path(scene or SCENE)
        if not path.is_file():
            raise FileNotFoundError(path)
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self.joint_ids = [_jid(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
        self.act_ids = [_jid(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in JOINT_NAMES]
        self.ee_site = _jid(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self.cube_body = _jid(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.nsub = max(1, int(round((1.0 / CONTROL_HZ) / float(self.model.opt.timestep))))
        self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.pick = POSE_A.copy()
        self.goal = POSE_B.copy()
        self.reset()

    @property
    def horizon(self) -> int:
        return HORIZON

    def ee_pos(self) -> np.ndarray:
        return np.array(self.data.site_xpos[self.ee_site], dtype=np.float64)

    def cube_pos(self) -> np.ndarray:
        return np.array(self.data.xpos[self.cube_body], dtype=np.float64)

    def gripper_opening(self) -> float:
        qadr = self.model.jnt_qposadr[self.joint_ids[5]]
        return float(self.data.qpos[qadr])

    def cube_to_goal(self) -> float:
        delta = self.cube_pos() - self.goal
        delta[2] = self.cube_pos()[2] - TABLE_Z - 0.012
        return float(np.linalg.norm(np.array([delta[0], delta[1], max(0.0, delta[2])])))

    def success(self) -> bool:
        c = self.cube_pos()
        xy = float(np.linalg.norm(c[:2] - self.goal[:2]))
        on_table = c[2] < 0.06
        open_jaw = self.gripper_opening() >= OPEN_THRESH
        return xy < SUCCESS_M and on_table and open_jaw

    def obs(self) -> np.ndarray:
        qpos = np.array(
            [self.data.qpos[self.model.jnt_qposadr[j]] for j in self.joint_ids],
            dtype=np.float64,
        )
        qvel = np.array(
            [self.data.qvel[self.model.jnt_dofadr[j]] for j in self.joint_ids],
            dtype=np.float64,
        )
        grip = np.array([self.gripper_opening()], dtype=np.float64)
        dist = np.array([np.linalg.norm(self.cube_pos() - self.goal)], dtype=np.float64)
        return np.concatenate(
            [qpos, qvel, self.ee_pos(), self.cube_pos(), self.goal, grip, dist]
        )

    def reset(self) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        self.steps = 0
        noise = self.rng.uniform(-0.03, 0.03, size=5)
        for i, jid in enumerate(self.joint_ids[:5]):
            qadr = self.model.jnt_qposadr[jid]
            lo, hi = self.model.jnt_range[jid]
            self.data.qpos[qadr] = float(np.clip(noise[i], lo, hi))
        gadr = self.model.jnt_qposadr[self.joint_ids[5]]
        self.data.qpos[gadr] = GRIPPER_OPEN
        for i, aid in enumerate(self.act_ids[:5]):
            self.data.ctrl[aid] = float(self.data.qpos[self.model.jnt_qposadr[self.joint_ids[i]]])
        self.data.ctrl[self.act_ids[5]] = GRIPPER_OPEN
        self.goal = POSE_B.copy()
        # Cube freejoint: pos (3) + quat (4)
        jnt = _jid(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
        adr = int(self.model.jnt_qposadr[jnt])
        jitter = self.rng.uniform(-0.01, 0.01, size=2)
        # pick has to be where the block actually is, not the pad centre it was
        # jittered away from, or anything aiming at pick closes beside it.
        self.pick = np.array(
            [POSE_A[0] + jitter[0], POSE_A[1] + jitter[1], POSE_A[2]],
            dtype=np.float64,
        )
        self.data.qpos[adr : adr + 3] = self.pick
        self.data.qpos[adr + 3 : adr + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self.obs()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        act = np.asarray(action, dtype=np.float64).reshape(-1)
        if act.shape[0] != 6:
            raise ValueError(f"action must be length 6, got {act.shape}")
        for i, aid in enumerate(self.act_ids[:5]):
            lo, hi = self.model.actuator_ctrlrange[aid]
            # Command off the measured joint, not off the last command. Adding
            # deltas to ctrl lets it wind up radians past a blocked joint, and
            # the arm then lurches when the block clears.
            qadr = self.model.jnt_qposadr[self.joint_ids[i]]
            base = float(self.data.qpos[qadr])
            self.data.ctrl[aid] = float(np.clip(base + ACTION_SCALE * act[i], lo, hi))
        glo, ghi = self.model.actuator_ctrlrange[self.act_ids[5]]
        gcur = float(self.data.ctrl[self.act_ids[5]])
        self.data.ctrl[self.act_ids[5]] = float(
            np.clip(gcur + GRIPPER_SCALE * act[5], glo, ghi)
        )
        for _ in range(self.nsub):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        dist = float(np.linalg.norm(self.cube_pos() - self.goal))
        ok = self.success()
        done = ok or self.steps >= HORIZON
        reward = -dist + (20.0 if ok else 0.0)
        info = {
            "success": ok,
            "distance_m": dist,
            "cube_pos": self.cube_pos(),
            "ee_pos": self.ee_pos(),
        }
        return self.obs(), float(reward), done, info
