"""Teach the twin by moving a 3D marker — RViz-style, without ROS.

The RGB sticks are a mocap handle. The claw follows with the same Jacobian
used for reach. Keys nudge the handle if the mouse perturb is awkward.
Recording writes a demo JSON (your motion). It is not a certificate.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import os

import numpy as np

if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

from twin.paths import repo_root

ROOT = repo_root()
DEMOS = ROOT / ".cache" / "twin-demos"
PREVIEW = ROOT / ".cache" / "twin-preview.jpg"


def preview_path(skill: str | None = None) -> Path:
    """One JPEG per skill so two classrooms cannot overwrite each other."""
    if skill:
        safe = "".join(c for c in skill if c.isalnum() or c in "-_") or "twin"
        return ROOT / ".cache" / f"twin-preview-{safe}.jpg"
    return PREVIEW

# GLFW key codes (viewer key_callback).
KEY_SPACE = 32
KEY_ENTER = 257
KEY_LEFT = 263
KEY_RIGHT = 262
KEY_UP = 265
KEY_DOWN = 264
KEY_LEFT_BRACKET = 91
KEY_RIGHT_BRACKET = 93
KEY_G = 71
KEY_O = 79
KEY_K = 75
KEY_R = 82
KEY_H = 72
KEY_P = 80

NUDGE_M = 0.015

HELP = """
TEACH — mix of the 3D window and the skill files
  The coloured sticks are the goal. The arm follows them.

  Mouse (MuJoCo): double-click the RGB marker, then Ctrl + drag.
  Keys:  arrows = move marker on the table plane
         [ ]    = move marker down / up
         G / O  = close / open the claw (place and barrier)
         K      = start or stop recording your motion
         Space  = save a waypoint (a still pose)
         Enter  = write the demo file to disk
         R      = reset the room
         P      = snapshot to the mission board
         H      = print this help

  This is practice with your hands in 3D. It does not stamp a license.
""".strip()


def mocap_xyz(env) -> np.ndarray:
    if env.model.nmocap < 1:
        raise RuntimeError("this scene has no mocap marker")
    return np.array(env.data.mocap_pos[0], dtype=np.float64)


def set_mocap_xyz(env, xyz: np.ndarray) -> None:
    import mujoco

    env.data.mocap_pos[0] = np.asarray(xyz, dtype=np.float64)
    if hasattr(env, "target"):
        env.target = np.array(env.data.mocap_pos[0], dtype=np.float64)
    mujoco.mj_forward(env.model, env.data)


def nudge_mocap(env, delta: np.ndarray) -> None:
    set_mocap_xyz(env, mocap_xyz(env) + np.asarray(delta, dtype=np.float64))


def park_marker_at_ee(env) -> None:
    if hasattr(env, "target"):
        set_mocap_xyz(env, env.ee_pos())
        return
    from twin import ik

    set_mocap_xyz(env, ik.pinch_pos(env))


def park_marker_for_teach(env) -> None:
    """Put the RGB handle on the work, not parked on a stretched wrist."""
    if hasattr(env, "cube_pos"):
        hover = np.asarray(env.cube_pos(), dtype=np.float64).copy()
        hover[2] += 0.05
        set_mocap_xyz(env, hover)
        return
    park_marker_at_ee(env)


def teach_action(env, *, gripper_cmd: float | None = None) -> np.ndarray:
    """Follow the marker. Reach = 5-DoF. Gripper skills = 6-DoF, jaw included.

    For the gripper skills the marker is the pinch point, not the wrist, so
    parking it on a block and pressing G actually picks the block up.
    """
    if env.model.nmocap >= 1 and hasattr(env, "target"):
        env.target = mocap_xyz(env)
        from twin.control import jacobian_p_action

        return jacobian_p_action(env)
    from twin import ik

    arm = ik.ik_delta(env, mocap_xyz(env), approach=ik.DOWN)
    target = ik.JAW_OPEN if gripper_cmd is None else float(gripper_cmd)
    return np.concatenate([arm, [ik.jaw_delta(env, target)]])


def snapshot_frame(env) -> dict:
    frame: dict = {
        "qpos": [float(x) for x in env.data.qpos],
        "ee": [float(x) for x in env.ee_pos()],
        "mocap": [float(x) for x in mocap_xyz(env)] if env.model.nmocap >= 1 else None,
    }
    if hasattr(env, "cube_pos"):
        frame["cube"] = [float(x) for x in env.cube_pos()]
        frame["gripper"] = float(env.gripper_opening())
    return frame


def apply_frame(env, frame: dict) -> None:
    qpos = np.asarray(frame["qpos"], dtype=np.float64)
    if qpos.shape[0] != env.data.qpos.shape[0]:
        raise ValueError("demo qpos does not match this scene")
    env.data.qpos[:] = qpos
    env.data.qvel[:] = 0.0
    if frame.get("mocap") is not None and env.model.nmocap >= 1:
        set_mocap_xyz(env, np.asarray(frame["mocap"], dtype=np.float64))
    import mujoco

    mujoco.mj_forward(env.model, env.data)


def write_demo(
    *,
    skill: str,
    frames: list[dict],
    waypoints: list[dict],
    catalog: bool = True,
) -> Path:
    DEMOS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = DEMOS / skill / f"demo-{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "skill": skill,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_frames": len(frames),
        "n_waypoints": len(waypoints),
        "frames": frames,
        "waypoints": waypoints,
        "certified": False,
    }
    path.write_text(json.dumps(doc) + "\n")
    if catalog:
        try:
            from platform.catalog import upsert_artifact

            upsert_artifact(
                tenant="robo-arm",
                stage="datasets",
                uri=f"robo-arm/datasets/twin-demos/{skill}/{path.name}",
                metrics={
                    "skill": skill,
                    "n_frames": len(frames),
                    "n_waypoints": len(waypoints),
                    "local_path": str(path),
                    "source": "twin-teach",
                },
                promoted=False,
            )
        except Exception:
            pass
    return path


def load_demo(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def write_preview(
    env, path: Path | None = None, *, skill: str | None = None
) -> Path | None:
    """RGB still for the mission board. Fails quietly if the renderer cannot start."""
    dest = Path(path) if path is not None else preview_path(skill)
    try:
        import mujoco
        from PIL import Image
    except ImportError:
        return None
    try:
        renderer = mujoco.Renderer(env.model, height=360, width=640)
        cam = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "classroom")
        if cam >= 0:
            renderer.update_scene(env.data, camera=cam)
        else:
            renderer.update_scene(env.data)
        pixels = renderer.render()
        renderer.close()
    except Exception:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(dest, format="JPEG", quality=80)
    return dest


class TeachSession:
    def __init__(self, env, skill: str) -> None:
        self.env = env
        self.skill = skill
        self.recording = False
        self.frames: list[dict] = []
        self.waypoints: list[dict] = []
        from twin import ik

        self.gripper_cmd = float(ik.JAW_OPEN)
        self.last_demo: str | None = None
        park_marker_for_teach(env)

    def on_key(self, key: int) -> None:
        env = self.env
        if key == KEY_H:
            print(HELP, flush=True)
            return
        if key == KEY_LEFT:
            nudge_mocap(env, [-NUDGE_M, 0.0, 0.0])
        elif key == KEY_RIGHT:
            nudge_mocap(env, [NUDGE_M, 0.0, 0.0])
        elif key == KEY_UP:
            nudge_mocap(env, [0.0, NUDGE_M, 0.0])
        elif key == KEY_DOWN:
            nudge_mocap(env, [0.0, -NUDGE_M, 0.0])
        elif key == KEY_LEFT_BRACKET:
            nudge_mocap(env, [0.0, 0.0, -NUDGE_M])
        elif key == KEY_RIGHT_BRACKET:
            nudge_mocap(env, [0.0, 0.0, NUDGE_M])
        elif key == KEY_G:
            from twin import ik

            self.gripper_cmd = float(ik.JAW_SHUT)
            print(json.dumps({"claw": "closed"}), flush=True)
        elif key == KEY_O:
            from twin import ik

            self.gripper_cmd = float(ik.JAW_OPEN)
            print(json.dumps({"claw": "open"}), flush=True)
        elif key == KEY_K:
            self.recording = not self.recording
            print(json.dumps({"recording": self.recording, "frames": len(self.frames)}), flush=True)
        elif key == KEY_SPACE:
            wp = snapshot_frame(env)
            wp["label"] = f"wp{len(self.waypoints) + 1}"
            self.waypoints.append(wp)
            print(json.dumps({"waypoint": wp["label"], "ee": wp["ee"]}), flush=True)
        elif key == KEY_ENTER:
            path = write_demo(
                skill=self.skill,
                frames=list(self.frames),
                waypoints=list(self.waypoints),
            )
            self.last_demo = str(path)
            print(json.dumps({"wrote": str(path), "n_frames": len(self.frames)}), flush=True)
        elif key == KEY_R:
            env.reset()
            park_marker_for_teach(env)
            print(json.dumps({"reset": True}), flush=True)
        elif key == KEY_P:
            dest = write_preview(env, skill=self.skill)
            print(json.dumps({"preview": str(dest) if dest else None}), flush=True)

    def tick(self) -> None:
        g = self.gripper_cmd if self.skill in {"place", "barrier"} else None
        action = teach_action(self.env, gripper_cmd=g)
        self.env.step(action)
        if self.recording:
            self.frames.append(snapshot_frame(self.env))
