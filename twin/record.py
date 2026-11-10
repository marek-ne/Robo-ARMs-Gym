"""Record twin episodes to PNG frames and an mp4.

Headless by design: MUJOCO_GL=egl, offscreen mujoco.Renderer, ffmpeg for the
encode. Used to produce the demo reel for the barrier skill, and reusable for
any skill the CLI can drive.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from twin.paths import repo_root

REPO = repo_root()
OUT = REPO / "videos" / "twin"
WIDTH, HEIGHT = 960, 540
LINKEDIN_SIM = (1080, 1110)
FPS = 30
# The twin steps at 1 action per control tick; keeping every frame makes a
# 600-step episode 20 s long, which is longer than anyone will watch.
STRIDE = 2


class Camera:
    """Near side-on, so the fence is edge-on and the lift reads as an arc."""

    azimuth = 95.0
    elevation = -18.0
    distance = 0.72
    lookat = (0.22, 0.0, 0.10)


# The teach marker is a control handle, not part of the task. A scripted or
# trained policy never touches it, so it is hidden while recording.
MARKER_GEOMS = ("ee_goal_sphere", "ee_goal_x", "ee_goal_y", "ee_goal_z")


def hide_marker(model: mujoco.MjModel) -> None:
    for name in MARKER_GEOMS:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid >= 0:
            model.geom_rgba[gid, 3] = 0.0


def _free_camera(model: mujoco.MjModel) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.azimuth = Camera.azimuth
    cam.elevation = Camera.elevation
    cam.distance = Camera.distance
    cam.lookat[:] = Camera.lookat
    return cam


def record_episode(
    env,
    action_fn,
    *,
    out_dir: Path,
    max_steps: int | None = None,
    stride: int = STRIDE,
    label: str = "",
    width: int = WIDTH,
    height: int = HEIGHT,
    brand: bool = False,
) -> dict:
    """Roll one episode, writing frame-NNNN.png into out_dir. Returns the info."""
    from PIL import Image, ImageDraw

    from twin.brand import frame_4x5

    out_dir.mkdir(parents=True, exist_ok=True)
    hide_marker(env.model)
    renderer = mujoco.Renderer(env.model, height=height, width=width)
    cam = _free_camera(env.model)
    n = 0
    kept = 0
    info: dict = {"success": False, "distance_m": float("nan")}
    limit = max_steps if max_steps is not None else env.horizon

    def _save(pixels, *, verdict: str | None = None) -> None:
        nonlocal kept
        img = Image.fromarray(pixels)
        if brand:
            img = frame_4x5(img, kicker=label, verdict=verdict)
        elif label:
            ImageDraw.Draw(img).text((12, 10), label, fill=(255, 255, 255))
        img.save(out_dir / f"frame-{kept:04d}.png")
        kept += 1

    try:
        while n < limit:
            if n % stride == 0:
                renderer.update_scene(env.data, camera=cam)
                _save(renderer.render())
            _obs, _rew, done, info = env.step(action_fn(env))
            n += 1
            if done:
                break
        renderer.update_scene(env.data, camera=cam)
        verdict = "PASS" if info.get("success") else f"FAIL  {info.get('reason', '?')}"
        last = renderer.render()
        hold = FPS if brand else FPS // 2
        for _ in range(hold):
            _save(last, verdict=verdict)
    finally:
        renderer.close()
    return {**info, "frames": kept, "steps": n}


def encode(frame_dirs: list[Path], out_mp4: Path, *, fps: int = FPS) -> Path:
    """Concatenate frame directories in order into one mp4."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found")
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    staging = out_mp4.parent / f".{out_mp4.stem}-frames"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    i = 0
    for d in frame_dirs:
        for f in sorted(d.glob("frame-*.png")):
            shutil.copy(f, staging / f"f-{i:05d}.png")
            i += 1
    if i == 0:
        raise RuntimeError("no frames to encode")
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(fps),
            "-i", str(staging / "f-%05d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-profile:v", "high", "-level", "4.1",
            "-crf", "18", "-preset", "medium",
            "-movflags", "+faststart",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            str(out_mp4),
        ],
        check=True,
    )
    shutil.rmtree(staging)
    return out_mp4


def filmstrip(frame_dir: Path, out_png: Path, *, count: int = 6, cols: int = 3) -> Path:
    """Evenly spaced stills laid out as a grid, for docs and chat."""
    from PIL import Image

    frames = sorted(frame_dir.glob("frame-*.png"))
    if not frames:
        raise RuntimeError(f"no frames in {frame_dir}")
    idx = np.linspace(0, len(frames) - 1, count).round().astype(int)
    picks = [Image.open(frames[i]) for i in idx]
    w, h = picks[0].size
    rows = (count + cols - 1) // cols
    sheet = Image.new("RGB", (w * cols, h * rows), (12, 12, 14))
    for k, im in enumerate(picks):
        sheet.paste(im, ((k % cols) * w, (k // cols) * h))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)
    return out_png


__all__ = ["record_episode", "encode", "filmstrip", "OUT", "LINKEDIN_SIM"]
