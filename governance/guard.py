"""Which pull-request diffs may spend a training job."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

GYMS = ("reach", "place", "barrier")

_ALLOWED = (
    re.compile(r"^skills/[a-z][a-z0-9_-]+/skill\.yaml$"),
    re.compile(r"^skills/[a-z][a-z0-9_-]+/reward\.py$"),
    re.compile(r"^policies/[a-z][a-z0-9_-]+/.+\.zip$"),
)


def git_diff_names(repo: Path, base: str, head: str) -> list[str]:
    out = subprocess.check_output(
        ["git", "diff", "--name-only", f"{base}...{head}"],
        cwd=repo,
        text=True,
    )
    return [line.strip() for line in out.splitlines() if line.strip()]


def review(paths: list[str]) -> dict[str, Any]:
    """Accept a diff only when every file is a skill contract, reward, or policy zip."""
    if not paths:
        return {"ok": False, "reason": "empty diff"}
    skills: set[str] = set()
    zips: list[str] = []
    rewards: list[str] = []
    for raw in paths:
        path = raw.replace("\\", "/").lstrip("./")
        if path == "twin/train.py" or path.startswith(".github/workflows/"):
            return {"ok": False, "reason": f"refusing GPU run: {path} changed"}
        if not any(pattern.match(path) for pattern in _ALLOWED):
            return {"ok": False, "reason": f"path not allowed: {path}"}
        parts = path.split("/")
        skills.add(parts[1])
        if path.endswith(".zip"):
            zips.append(path)
        if path.endswith("/reward.py"):
            rewards.append(path)
    if len(skills) != 1:
        return {"ok": False, "reason": "one skill per train request"}
    if zips and rewards:
        return {"ok": False, "reason": "send a reward.py or a policy zip, not both"}
    if len(zips) > 1:
        return {"ok": False, "reason": "one policy zip per train request"}
    skill = next(iter(skills))
    if zips:
        mode = "eval-zip"
    elif rewards:
        mode = "train"
    else:
        mode = "scripted"
    return {
        "ok": True,
        "reason": "ok",
        "skill": skill,
        "mode": mode,
        "model": zips[0] if zips else None,
        "paths": [p.replace("\\", "/").lstrip("./") for p in paths],
    }


def gym_from_manifest(text: str, skill: str) -> str:
    """Built-in twin used to train this skill. ``gym:`` in the manifest wins."""
    data = yaml.safe_load(text) or {}
    gym = str(data.get("gym") or skill)
    if gym not in GYMS:
        raise ValueError(f"no gym for {skill!r} (gym={gym!r})")
    return gym


def manifest_text(repo: Path, skill: str, ref: str | None = None) -> str:
    """Read skill.yaml from a git ref when given, otherwise from the worktree."""
    if ref:
        try:
            return subprocess.check_output(
                ["git", "show", f"{ref}:skills/{skill}/skill.yaml"],
                cwd=repo,
                text=True,
            )
        except subprocess.CalledProcessError:
            pass
    path = repo / "skills" / skill / "skill.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"missing skills/{skill}/skill.yaml")
    return path.read_text(encoding="utf-8")


def gym_for(repo: Path, skill: str, ref: str | None = None) -> str:
    return gym_from_manifest(manifest_text(repo, skill, ref), skill)


def decide(repo: Path, paths: list[str], *, ref: str | None = None) -> dict[str, Any]:
    decision = review(paths)
    if not decision["ok"]:
        return decision
    try:
        decision["gym"] = gym_for(repo, decision["skill"], ref)
    except (FileNotFoundError, ValueError) as exc:
        return {"ok": False, "reason": str(exc)}
    return decision


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Allow or refuse a train diff")
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--repo", default=".")
    args = parser.parse_args(argv)
    repo = Path(args.repo).resolve()
    try:
        paths = git_diff_names(repo, args.base, args.head)
    except subprocess.CalledProcessError as exc:
        print(json.dumps({"ok": False, "reason": f"git diff failed: {exc.returncode}"}))
        return 2
    decision = decide(repo, paths, ref=args.head)
    print(json.dumps(decision))
    return 0 if decision["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
