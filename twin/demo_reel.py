"""Build the barrier demo reel: a few passes, then the naive failure.

  .venv/bin/python -m twin.demo_reel

Writes LinkedIn 4:5 (1080×1350) plus 9:16 and a 16:9 1920×1080 hub copy.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")

from twin.barrier_control import naive_barrier_action, scripted_barrier_action  # noqa: E402
from twin.barrier_env import BarrierTwin  # noqa: E402
from twin.brand import end_4x5, title_4x5  # noqa: E402
from twin.record import FPS, LINKEDIN_SIM, OUT, encode, filmstrip, record_episode  # noqa: E402

PASS_SEEDS = (0, 3, 7)
FAIL_SEED = 5


def _hold(img, n: int, dest, start: int) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    i = start
    for _ in range(n):
        img.save(dest / f"frame-{i:04d}.png")
        i += 1
    return i


def _ffmpeg_pad(src, dest, vf: str):
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(src),
            "-vf", vf,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-profile:v", "high", "-crf", "18",
            "-movflags", "+faststart",
            str(dest),
        ],
        check=True,
    )
    return dest


def main() -> int:
    frames_root = OUT / "frames"
    if frames_root.exists():
        shutil.rmtree(frames_root)
    dirs = []
    episode_dirs = []
    results = []

    title_dir = frames_root / "title"
    _hold(title_4x5(), FPS * 3, title_dir, 0)
    dirs.append(title_dir)

    w, h = LINKEDIN_SIM
    for seed in PASS_SEEDS:
        env = BarrierTwin(seed=seed)
        env.reset()
        label = f"over the fence  ·  {env.wall_top() * 1000:.0f} mm fence"
        d = frames_root / f"pass-{seed}"
        info = record_episode(
            env,
            scripted_barrier_action,
            out_dir=d,
            label=label,
            width=w,
            height=h,
            brand=True,
        )
        dirs.append(d)
        episode_dirs.append(d)
        results.append({"seed": seed, "mode": "over", **_summary(info)})

    env = BarrierTwin(seed=FAIL_SEED)
    env.reset()
    label = "straight across  ·  must fail"
    d = frames_root / f"fail-{FAIL_SEED}"
    info = record_episode(
        env,
        naive_barrier_action,
        out_dir=d,
        label=label,
        width=w,
        height=h,
        brand=True,
    )
    dirs.append(d)
    episode_dirs.append(d)
    results.append({"seed": FAIL_SEED, "mode": "naive", **_summary(info)})

    end_dir = frames_root / "end"
    _hold(end_4x5(), FPS * 3, end_dir, 0)
    dirs.append(end_dir)

    linkedin = encode(dirs, OUT / "barrier-demo-linkedin.mp4")
    # 4:5 → 9:16 (285+1350+285 = 1920). 4:5 → 16:9 (864×1080 centred in 1920×1080).
    vertical = _ffmpeg_pad(
        linkedin,
        OUT / "barrier-demo-linkedin-9x16.mp4",
        "pad=1080:1920:0:285:color=090c14,format=yuv420p",
    )
    landscape = _ffmpeg_pad(
        linkedin,
        OUT / "barrier-demo-linkedin-16x9.mp4",
        "scale=864:1080,pad=1920:1080:(ow-iw)/2:0:color=090c14,format=yuv420p",
    )
    shutil.copy(landscape, OUT / "barrier-demo.mp4")
    landscape = OUT / "barrier-demo.mp4"
    strip = filmstrip(episode_dirs[0], OUT / "barrier-filmstrip.png", count=6, cols=3)
    fail_strip = filmstrip(episode_dirs[-1], OUT / "barrier-naive-fails.png", count=3, cols=3)
    shutil.rmtree(frames_root)
    print(
        json.dumps(
            {
                "linkedin_4x5": str(linkedin),
                "linkedin_9x16": str(vertical),
                "hub_16x9": str(landscape),
                "filmstrip": str(strip),
                "naive_filmstrip": str(fail_strip),
                "episodes": results,
            },
            indent=2,
        )
    )
    return 0


def _summary(info: dict) -> dict:
    return {
        "success": bool(info.get("success")),
        "reason": info.get("reason", ""),
        "distance_cm": round(float(info.get("distance_m", 0.0)) * 100, 2),
        "steps": info.get("steps"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
