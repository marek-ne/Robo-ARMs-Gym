"""Repo layout and artifact-catalog location for the Robo-arm platform layer."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = REPO_ROOT / "skills"
CACHE_DIR = REPO_ROOT / ".cache"

CATALOG_ENV = "FIMPO_ARTIFACT_CATALOG"
LOCAL_CATALOG = CACHE_DIR / "artifact_catalog.json"


def resolve_fimpo_runtime() -> Path | None:
    """Fimpo agent-runtime, if this machine has one.

    Set ``FIMPO_RUNTIME`` to point at a checkout. The private tree may also
    ship ``platform/private_runtime.py``. The public gym export does not.
    """
    env = os.environ.get("FIMPO_RUNTIME")
    if env:
        return Path(env)
    try:
        from platform import private_runtime
    except ImportError:
        return None
    path = Path(private_runtime.bundled_runtime())
    return path if path.is_dir() else None


FIMPO_RUNTIME = resolve_fimpo_runtime()


def catalog_path() -> Path:
    """Resolve the JSON catalog file (env override, then fimpo, then repo cache)."""
    env = os.environ.get(CATALOG_ENV)
    if env:
        return Path(env)
    runtime = resolve_fimpo_runtime()
    if runtime is not None and (runtime / "artifact_catalog.py").is_file():
        return runtime / "out" / "artifact_catalog.json"
    return LOCAL_CATALOG
