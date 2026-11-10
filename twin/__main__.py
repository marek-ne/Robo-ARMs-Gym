#!/usr/bin/env python3
"""Digital twin CLI.

    .venv/bin/python -m twin teach  --skill barrier
    .venv/bin/python -m twin viewer --skill barrier --policy scripted
    .venv/bin/python -m twin eval   --skill barrier --policy scripted --episodes 50
    .venv/bin/python -m twin train  --skill barrier --steps 50000
    .venv/bin/python -m twin eval   --skill barrier --policy ppo --model <zip> --episodes 50
    .venv/bin/python -m twin run    --device so101-twin-a --skill barrier
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _heartbeat(device_id: str, env, skill: str) -> None:
    from fleet.registry import heartbeat

    extra = {
        "ee_pos": [float(x) for x in env.ee_pos()],
        "qpos": [float(env.data.qpos[env.model.jnt_qposadr[j]]) for j in env.joint_ids],
        "backend": "mujoco",
    }
    if hasattr(env, "cube_pos"):
        extra["cube_pos"] = [float(x) for x in env.cube_pos()]
        extra["goal_pos"] = [float(x) for x in env.goal]
        extra["distance_m"] = float(np.linalg.norm(env.cube_pos() - env.goal))
    if hasattr(env, "target"):
        extra["target_pos"] = [float(x) for x in env.target]
        extra["distance_m"] = float(env.distance())
    if hasattr(env, "wall_top"):
        extra["wall_top_m"] = float(env.wall_top())
        extra["collision"] = bool(env.collided)
    if env.model.nmocap >= 1:
        extra["mocap"] = [float(x) for x in env.data.mocap_pos[0]]
    last_exc: Exception | None = None
    for _ in range(8):
        try:
            heartbeat(
                device_id,
                skill=skill,
                status="running",
                kind="digital-twin",
                extra=extra,
            )
            return
        except Exception as exc:
            last_exc = exc
            time.sleep(0.05)
    print(json.dumps({"heartbeat_error": str(last_exc), "device": device_id}), flush=True)


SKILLS = ("reach", "place", "barrier")


def _make_env(skill: str, seed: int):
    if skill == "barrier":
        from twin.barrier_env import BarrierTwin

        return BarrierTwin(seed=seed)
    if skill == "place":
        from twin.place_env import PlaceTwin

        return PlaceTwin(seed=seed)
    from twin.reach_env import ReachTwin

    return ReachTwin(seed=seed)


def _load_ppo(model_path: str):
    from stable_baselines3 import PPO

    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"--model not found: {path}")
    return PPO.load(str(path), device="cpu")


def _policy_action(env, skill: str, policy: str, model=None):
    if policy == "ppo":
        if model is None:
            raise RuntimeError("--policy ppo requires --model PATH.zip")
        from twin.train import predict_action

        return predict_action(model, env.obs())
    if skill == "barrier":
        from twin.barrier_control import scripted_barrier_action

        if policy in {"scripted", "jacobian"}:
            return scripted_barrier_action(env)
        return env.rng.uniform(-1.0, 1.0, size=6)
    if skill == "place":
        from twin.place_control import scripted_place_action

        if policy in {"scripted", "jacobian"}:
            return scripted_place_action(env)
        return env.rng.uniform(-1.0, 1.0, size=6)
    from twin.control import jacobian_p_action

    # Reach has no separate scripted plan: the Jacobian controller *is* it.
    # Without this, --policy scripted quietly evaluated random actions.
    if policy in {"jacobian", "scripted"}:
        return jacobian_p_action(env)
    return env.rng.uniform(-1.0, 1.0, size=5)


def _bind_model(args: argparse.Namespace) -> None:
    args._model = None
    if getattr(args, "policy", None) == "ppo":
        if not getattr(args, "model", None):
            raise SystemExit("error: --policy ppo requires --model PATH.zip")
        args._model = _load_ppo(args.model)


def cmd_train(args: argparse.Namespace) -> int:
    from twin.train import train

    result = train(
        skill=args.skill,
        total_timesteps=args.steps,
        n_envs=args.n_envs,
        seed=args.seed,
        resume=args.resume,
        catalog=not args.no_catalog,
    )
    print(json.dumps(result, indent=2))
    print(
        f"next: .venv/bin/python -m twin eval --skill {args.skill} "
        f"--policy ppo --model {result['checkpoint']} --episodes 50",
        flush=True,
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    env = _make_env(args.skill, args.seed)
    t0 = time.time()
    n = 0
    period = 1.0 / 30.0
    next_tick = time.perf_counter()
    print(
        json.dumps(
            {
                "device": args.device,
                "skill": args.skill,
                "policy": args.policy,
                "backend": "mujoco",
            }
        ),
        flush=True,
    )
    try:
        while True:
            action = _policy_action(env, args.skill, args.policy, model=getattr(args, "_model", None))
            _obs, _rew, done, info = env.step(action)
            n += 1
            if n == 1 or n % args.heartbeat_every == 0:
                _heartbeat(args.device, env, args.skill)
                from twin.teach import write_preview

                write_preview(env)
            if done:
                print(
                    json.dumps(
                        {
                            "episode_end": True,
                            "success": bool(info["success"]),
                            "distance_m": info["distance_m"],
                            "steps": env.steps,
                        }
                    ),
                    flush=True,
                )
                env.reset()
            if args.seconds is not None and (time.time() - t0) >= args.seconds:
                break
            if args.max_steps is not None and n >= args.max_steps:
                break
            next_tick += period
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.perf_counter()
    finally:
        try:
            _heartbeat(args.device, env, args.skill)
            from fleet.registry import set_status

            set_status(args.device, "idle")
        except Exception as exc:
            print(json.dumps({"shutdown_error": str(exc), "device": args.device}), flush=True)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    env = _make_env(args.skill, args.seed)
    successes = 0
    dists: list[float] = []
    for _ in range(args.episodes):
        env.reset()
        done = False
        info = {"success": False, "distance_m": 0.0}
        while not done:
            action = _policy_action(env, args.skill, args.policy, model=getattr(args, "_model", None))
            _obs, _rew, done, info = env.step(action)
        successes += int(bool(info["success"]))
        dists.append(float(info["distance_m"]))
    rate = successes / args.episodes if args.episodes else 0.0
    mean_cm = (sum(dists) / len(dists) * 100.0) if dists else None
    result = {
        "skill": args.skill,
        "policy": args.policy,
        "n_episodes": args.episodes,
        "success_rate": rate,
        "mean_distance_cm": mean_cm,
        "backend": "mujoco",
    }
    print(json.dumps(result, indent=2))
    if args.emit_eval:
        coolify = str(ROOT / "coolify")
        if coolify not in sys.path:
            sys.path.insert(0, coolify)
        try:
            import eval_store
        except Exception:
            out = ROOT / ".cache" / "evals" / "latest.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"recorded": str(out), "promoted": False}, indent=2))
            return 0

        ckpt = args.model if args.policy == "ppo" and args.model else f"twin:{args.policy}"
        rec = eval_store.record_eval(
            checkpoint_uri=ckpt,
            success_rate=rate,
            mean_distance_cm=mean_cm,
            n_episodes=args.episodes,
            notes="mujoco digital twin — real episodes",
            skill=args.skill,
            skill_version=args.skill_version,
        )
        print(json.dumps({"recorded": rec.get("eval_id"), "promoted": False}, indent=2))
    return 0


def cmd_teach(args: argparse.Namespace) -> int:
    import mujoco.viewer

    from twin.teach import HELP, TeachSession, apply_frame, load_demo, write_preview

    env = _make_env(args.skill, args.seed)
    if args.replay:
        demo = load_demo(Path(args.replay))
        frames = demo.get("frames") or []
        if not frames:
            raise SystemExit(f"no frames in {args.replay}")
        print(json.dumps({"replay": args.replay, "n_frames": len(frames)}), flush=True)
        i = 0
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            while viewer.is_running():
                apply_frame(env, frames[i % len(frames)])
                i += 1
                if i == 1 or i % 45 == 0:
                    write_preview(env)
                viewer.sync()
                time.sleep(1.0 / 30.0)
        return 0

    session = TeachSession(env, args.skill)
    print(HELP, flush=True)
    n = 0
    with mujoco.viewer.launch_passive(
        env.model, env.data, key_callback=session.on_key
    ) as viewer:
        while viewer.is_running():
            session.tick()
            n += 1
            if n == 1 or n % 45 == 0:
                write_preview(env)
                if args.device:
                    _heartbeat(args.device, env, args.skill)
            viewer.sync()
            time.sleep(1.0 / 30.0)
    return 0


def cmd_viewer(args: argparse.Namespace) -> int:
    import mujoco.viewer

    from twin.teach import write_preview

    env = _make_env(args.skill, args.seed)
    n = 0
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            action = _policy_action(env, args.skill, args.policy, model=getattr(args, "_model", None))
            _obs, _rew, done, _info = env.step(action)
            if done:
                env.reset()
            n += 1
            if n == 1 or n % 45 == 0:
                write_preview(env)
            if args.device:
                _heartbeat(args.device, env, args.skill)
            viewer.sync()
    return 0


def _add_skill_policy(sp: argparse.ArgumentParser, *, default_skill: str = "reach") -> None:
    sp.add_argument("--skill", choices=SKILLS, default=default_skill)
    sp.add_argument(
        "--policy",
        choices=("jacobian", "scripted", "random", "ppo"),
        default="jacobian",
    )
    sp.add_argument("--model", default=None, help="PPO zip (required when --policy ppo)")
    sp.add_argument("--seed", type=int, default=0)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="SO-101 digital twin (MuJoCo)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="Step physics and heartbeat the catalog")
    pr.add_argument("--device", default="so101-twin-a")
    _add_skill_policy(pr)
    pr.add_argument("--seconds", type=float, default=None)
    pr.add_argument("--max-steps", type=int, default=None)
    pr.add_argument("--heartbeat-every", type=int, default=90)

    pe = sub.add_parser("eval", help="Run real episodes (does not promote)")
    pe.add_argument("--episodes", type=int, default=20)
    _add_skill_policy(pe)
    pe.add_argument("--skill-version", default="0.1.0")
    pe.add_argument(
        "--emit-eval",
        action="store_true",
        help="Record measured rates into the eval store (never auto-promotes)",
    )

    pv = sub.add_parser("viewer", help="Interactive MuJoCo window")
    pv.add_argument("--device", default="so101-twin-a")
    _add_skill_policy(pv)

    pt = sub.add_parser("train", help="PPO on the CPU twin (does not promote)")
    pt.add_argument("--skill", choices=SKILLS, default="barrier")
    pt.add_argument("--steps", type=int, default=50_000)
    pt.add_argument("--n-envs", type=int, default=1)
    pt.add_argument("--seed", type=int, default=0)
    pt.add_argument("--resume", default=None, help="Existing PPO zip")
    pt.add_argument(
        "--no-catalog",
        action="store_true",
        help="Do not upsert the checkpoint into the artifact catalog",
    )

    th = sub.add_parser(
        "teach",
        help="3D window: drag/nudge the RGB marker, arm follows (not a certificate)",
    )
    th.add_argument("--skill", choices=SKILLS, default="barrier")
    th.add_argument("--device", default="so101-twin-a")
    th.add_argument("--seed", type=int, default=0)
    th.add_argument("--replay", default=None, help="Play back a demo JSON")

    pw = sub.add_parser(
        "web",
        help="Browser classroom for Windows PCs on the LAN (port 8092)",
    )
    pw.add_argument("--skill", choices=SKILLS, default="barrier")
    pw.add_argument("--host", default="0.0.0.0")
    pw.add_argument("--port", type=int, default=8092)
    pw.add_argument("--device", default="so101-twin-web")
    pw.add_argument("--seed", type=int, default=0)

    args = p.parse_args(argv)
    if args.cmd == "train":
        return cmd_train(args)
    if args.cmd == "teach":
        return cmd_teach(args)
    if args.cmd == "web":
        from twin.web import serve

        serve(
            skill=args.skill,
            host=args.host,
            port=args.port,
            device=args.device,
            seed=args.seed,
        )
        return 0
    _bind_model(args)
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "eval":
        return cmd_eval(args)
    if args.cmd == "viewer":
        return cmd_viewer(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
