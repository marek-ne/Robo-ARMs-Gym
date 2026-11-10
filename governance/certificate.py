"""Certificate that ties an eval to a Snyk result and a gateway receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from governance.redact import public_mapping

MIN_RATE = 0.8
MIN_EPISODES = 50


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def git_sha(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def gate_passes(
    *,
    success_rate: float,
    n_episodes: int,
    snyk: dict[str, Any] | None,
    gateway: dict[str, Any] | None,
    min_rate: float = MIN_RATE,
    min_episodes: int = MIN_EPISODES,
) -> bool:
    if not isinstance(snyk, dict) or snyk.get("status") != "passed":
        return False
    if not isinstance(gateway, dict) or not gateway.get("admitted"):
        return False
    if n_episodes < min_episodes:
        return False
    if success_rate < min_rate:
        return False
    return True


def build_certificate(
    *,
    skill: str,
    gym: str,
    success_rate: float,
    n_episodes: int,
    mean_distance_cm: float | None,
    checkpoint: str | None,
    snyk: dict[str, Any] | None,
    gateway: dict[str, Any] | None,
    repo: Path,
    policy: str,
    min_rate: float = MIN_RATE,
    min_episodes: int = MIN_EPISODES,
) -> dict[str, Any]:
    snyk_public = public_mapping(snyk)
    gateway_public = public_mapping(gateway)
    digest = ""
    if checkpoint:
        path = Path(checkpoint)
        if path.is_file():
            digest = sha256_file(path)
    passed = gate_passes(
        success_rate=success_rate,
        n_episodes=n_episodes,
        snyk=snyk_public,
        gateway=gateway_public,
        min_rate=min_rate,
        min_episodes=min_episodes,
    )
    return {
        "skill": skill,
        "gym": gym,
        "policy": policy,
        "success_rate": success_rate,
        "n_episodes": n_episodes,
        "mean_distance_cm": mean_distance_cm,
        "checkpoint": checkpoint,
        "checkpoint_sha256": digest,
        "git_sha": git_sha(repo),
        "snyk": snyk_public,
        "gateway": gateway_public,
        "min_success_rate": min_rate,
        "min_episodes": min_episodes,
        "passed": passed,
    }


def render_comment(certificate: dict[str, Any]) -> str:
    """Pull-request comment. Receipts are already redacted."""
    passed = "pass" if certificate.get("passed") else "fail"
    gateway = certificate.get("gateway") or {}
    snyk = certificate.get("snyk") or {}
    lines = [
        f"## Train certificate — {passed}",
        "",
        f"- skill: `{certificate.get('skill')}`",
        f"- gym: `{certificate.get('gym')}`",
        f"- policy: `{certificate.get('policy')}`",
        f"- success: {certificate.get('success_rate')} over {certificate.get('n_episodes')} episodes",
        f"- mean distance cm: {certificate.get('mean_distance_cm')}",
        f"- checkpoint sha256: `{certificate.get('checkpoint_sha256') or 'n/a'}`",
        f"- git sha: `{certificate.get('git_sha') or 'n/a'}`",
        f"- snyk: `{snyk.get('status')}`",
        f"- gateway: `{gateway.get('reason')}` model `{gateway.get('model')}` worker `{gateway.get('worker')}`",
        "",
        "Merge bar: Snyk passed, LiteLLM confirmed Node D, the card was free, and the eval is at least "
        f"{certificate.get('min_success_rate')} over {certificate.get('min_episodes')} episodes.",
    ]
    return "\n".join(lines) + "\n"


def stamp_manifest(repo: Path, certificate: dict[str, Any], *, version: str | None = None) -> dict[str, Any] | None:
    """Write eval_id and checkpoint_uri when the gate passed. Returns the manifest or None."""
    if not certificate.get("passed"):
        return None
    import sys

    import yaml

    root = str(repo)
    if root not in sys.path:
        sys.path.insert(0, root)
    from skills.registry import write_certificate

    skill = str(certificate["skill"])
    manifest_path = repo / "skills" / skill / "skill.yaml"
    current = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    pinned = version or str(current.get("version") or "0.1.0")
    eval_id = f"eval-{certificate.get('git_sha') or 'local'}-{skill}"
    checkpoint = certificate.get("checkpoint") or f"twin:{certificate.get('policy')}"
    return write_certificate(
        skill,
        pinned,
        eval_id=eval_id,
        checkpoint_uri=str(checkpoint),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a train certificate JSON")
    parser.add_argument("--skill", required=True)
    parser.add_argument("--gym", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--success-rate", type=float, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--mean-distance-cm", type=float, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--snyk", required=True, help="Snyk receipt JSON file")
    parser.add_argument("--gateway", required=True, help="Gateway receipt JSON file")
    parser.add_argument("--repo", default=".")
    parser.add_argument("--out", required=True)
    parser.add_argument("--comment", help="Also write the pull-request comment markdown")
    parser.add_argument("--stamp", action="store_true", help="Update skill.yaml when the gate passes")
    args = parser.parse_args(argv)
    repo = Path(args.repo).resolve()
    snyk = json.loads(Path(args.snyk).read_text(encoding="utf-8"))
    gateway = json.loads(Path(args.gateway).read_text(encoding="utf-8"))
    certificate = build_certificate(
        skill=args.skill,
        gym=args.gym,
        success_rate=args.success_rate,
        n_episodes=args.episodes,
        mean_distance_cm=args.mean_distance_cm,
        checkpoint=args.checkpoint,
        snyk=snyk,
        gateway=gateway,
        repo=repo,
        policy=args.policy,
    )
    dest = Path(args.out)
    dest.write_text(json.dumps(certificate, indent=2) + "\n", encoding="utf-8")
    if args.comment:
        Path(args.comment).write_text(render_comment(certificate), encoding="utf-8")
    if args.stamp:
        stamp_manifest(repo, certificate)
    print(json.dumps({"passed": certificate["passed"], "out": str(dest)}))
    return 0 if certificate["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
