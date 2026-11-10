"""Lift a block over a barrier: pick at A, clear the wall, place at B.

Matches skills/barrier/skill.yaml:
  obs    = place obs (23) + wall_pos(3) + wall_size(3) = 29
  action = joint_position_delta(6)  # gripper included
  pass   = cube within 2 cm of B, resting on the table, jaw open, wall untouched
  fail   = any contact with the wall, cube off the table, or 600 steps

Every reset varies wall height, wall offset, cube start and target pad, so a
single memorised trajectory cannot pass a 50-episode exam.
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
from twin.place_env import (
    ACTION_SCALE,
    GRIPPER_OPEN,
    GRIPPER_SCALE,
    OPEN_THRESH,
    SUCCESS_M,
    PlaceTwin,
    _jid,
)

REPO = repo_root()
SCENE = REPO / "urdf" / "so101" / "barrier_twin.xml"

HORIZON = 600
BLOCK_RADIUS = 0.012
BLOCK_HALF_H = 0.016
TABLE_TOP = 0.020
TABLE_CENTRE_X = 0.26
TABLE_HALF = 0.14
# The fence sits between the near pad and the far pad and spans the table
# sideways, so the only way across is over the top.
#
# All of these ranges sit inside the arm's straight-down working envelope,
# which is x 0.16..0.32 m at this table height (measured by sweeping the IK
# over the wall-free scene). Push the pads outside it and the wrist cannot
# point at the table any more, so a miss stops being the policy's fault.
START = np.array([0.175, 0.0, TABLE_TOP + BLOCK_HALF_H], dtype=np.float64)
WALL_HALF_X = 0.006
WALL_HALF_Y = 0.15
WALL_HALF_Z_RANGE = (0.015, 0.032)
WALL_X_RANGE = (0.225, 0.245)
GOAL_X_RANGE = (0.280, 0.315)
GOAL_Y_RANGE = (-0.045, 0.045)
CUBE_JITTER_X = 0.012
CUBE_JITTER_Y = 0.030
# Contacts with these are normal support, not a barrier strike.
IGNORE_CONTACT = ("table_top", "floor")


class BarrierTwin(PlaceTwin):
    """One over-the-wall episode. Touching the wall ends it as a failure."""

    def __init__(self, *, scene: Path | None = None, seed: int = 0) -> None:
        self.collided = False
        self.hit_by = ""
        self.dropped = False
        # PlaceTwin.__init__ ends with reset(); the wall ids do not exist yet.
        self._ready = False
        super().__init__(scene=Path(scene or SCENE), seed=seed)
        self.wall_body = _jid(self.model, mujoco.mjtObj.mjOBJ_BODY, "wall")
        self.wall_geom = _jid(self.model, mujoco.mjtObj.mjOBJ_GEOM, "wall_geom")
        self.pad_b_body = _jid(self.model, mujoco.mjtObj.mjOBJ_BODY, "pad_b")
        self.cube_geom = _jid(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        self._ignore_geoms = set()
        for name in IGNORE_CONTACT:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                self._ignore_geoms.add(gid)
        self._ready = True
        self.reset()

    @property
    def horizon(self) -> int:
        return HORIZON

    # --- scene layout -------------------------------------------------
    def wall_pos(self) -> np.ndarray:
        return np.array(self.model.body_pos[self.wall_body], dtype=np.float64)

    def wall_size(self) -> np.ndarray:
        return np.array(self.model.geom_size[self.wall_geom], dtype=np.float64)

    def wall_top(self) -> float:
        return float(self.wall_pos()[2] + self.wall_size()[2])

    def clearance_z(self) -> float:
        """Lowest block centre height that carries it over the fence."""
        return self.wall_top() + BLOCK_HALF_H + 0.012

    def _randomise_scene(self) -> None:
        half_z = float(self.rng.uniform(*WALL_HALF_Z_RANGE))
        wall_x = float(self.rng.uniform(*WALL_X_RANGE))
        self.model.geom_size[self.wall_geom] = [WALL_HALF_X, WALL_HALF_Y, half_z]
        self.model.body_pos[self.wall_body] = [wall_x, 0.0, TABLE_TOP + half_z]
        self.goal = np.array(
            [
                float(self.rng.uniform(*GOAL_X_RANGE)),
                float(self.rng.uniform(*GOAL_Y_RANGE)),
                START[2],
            ],
            dtype=np.float64,
        )
        self.model.body_pos[self.pad_b_body] = [self.goal[0], self.goal[1], 0.021]

    # --- episode ------------------------------------------------------
    def success(self) -> bool:
        return bool(super().success()) and not self.collided and not self.dropped

    def obs(self) -> np.ndarray:
        if not self._ready:
            return super().obs()
        return np.concatenate([super().obs(), self.wall_pos(), self.wall_size()])

    def reset(self) -> np.ndarray:
        self.collided = False
        self.hit_by = ""
        self.dropped = False
        super().reset()
        if not self._ready:
            return super().obs()
        self._randomise_scene()
        jnt = _jid(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
        adr = int(self.model.jnt_qposadr[jnt])
        self.pick = np.array(
            [
                START[0] + float(self.rng.uniform(-CUBE_JITTER_X, CUBE_JITTER_X)),
                START[1] + float(self.rng.uniform(-CUBE_JITTER_Y, CUBE_JITTER_Y)),
                START[2],
            ],
            dtype=np.float64,
        )
        self.data.qpos[adr : adr + 3] = self.pick
        self.data.qpos[adr + 3 : adr + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[:] = 0.0
        if self.model.nmocap >= 1:
            self.data.mocap_pos[0] = self.pick + np.array([0.0, 0.0, 0.10])
        mujoco.mj_forward(self.model, self.data)
        return self.obs()

    def _wall_touch(self) -> str:
        """Name of what is touching the wall this instant, '' if nothing."""
        for i in range(int(self.data.ncon)):
            con = self.data.contact[i]
            g1, g2 = int(con.geom1), int(con.geom2)
            if self.wall_geom not in (g1, g2):
                continue
            other = g2 if g1 == self.wall_geom else g1
            if other in self._ignore_geoms:
                continue
            return "cube" if other == self.cube_geom else "arm"
        return ""

    def _cube_off_table(self) -> bool:
        c = self.cube_pos()
        return bool(
            c[2] < TABLE_TOP - 0.002
            or abs(c[0] - TABLE_CENTRE_X) > TABLE_HALF + 0.015
            or abs(c[1]) > 0.175
        )

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
            touch = self._wall_touch()
            if touch and not self.collided:
                self.collided = True
                self.hit_by = touch
        self.steps += 1
        if self._cube_off_table():
            self.dropped = True

        cube = self.cube_pos()
        dist = float(np.linalg.norm(cube - self.goal))
        ok = self.success()
        done = ok or self.collided or self.dropped or self.steps >= HORIZON

        reward = self._reward(cube, dist, ok)
        reason = (
            "success"
            if ok
            else f"collision:{self.hit_by}"
            if self.collided
            else "dropped"
            if self.dropped
            else "timeout"
            if self.steps >= HORIZON
            else ""
        )
        info = {
            "success": ok,
            "distance_m": dist,
            "cube_pos": cube,
            "ee_pos": self.ee_pos(),
            "collision": self.collided,
            "hit_by": self.hit_by,
            "dropped": self.dropped,
            "wall_top_m": self.wall_top(),
            "reason": reason,
        }
        return self.obs(), float(reward), done, info

    def _reward(self, cube: np.ndarray, dist: float, ok: bool) -> float:
        """Approach the block, get it above the wall, carry it to B."""
        ee_to_cube = float(np.linalg.norm(self.ee_pos() - cube))
        reward = -1.5 * dist - 0.5 * ee_to_cube
        if cube[0] < self.wall_pos()[0]:
            # Before the fence, only height counts as progress.
            reward += 2.0 * min(cube[2], self.clearance_z()) / self.clearance_z()
        else:
            reward += 2.0
        if self.collided:
            reward -= 10.0
        if self.dropped:
            reward -= 5.0
        if ok:
            reward += 30.0
        return reward


__all__ = [
    "BarrierTwin",
    "GRIPPER_OPEN",
    "HORIZON",
    "OPEN_THRESH",
    "SUCCESS_M",
    "SCENE",
]
