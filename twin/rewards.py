"""Optional per-skill reward override.

``skills/<name>/reward.py`` may define ``reward(obs, action, info, default) -> float``.
Missing file means the env reward is used unchanged. Import errors propagate:
a broken contribution should fail the run, not silently score the default.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Callable

from twin.paths import repo_root

RewardFn = Callable[..., float]


def reward_path(skill: str, root: Path | None = None) -> Path:
    base = root if root is not None else repo_root()
    return base / "skills" / skill / "reward.py"


def load_reward(skill: str, root: Path | None = None) -> RewardFn | None:
    path = reward_path(skill, root)
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location(f"roboarm_reward_{skill}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load reward module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, "reward", None)
    if not callable(fn):
        raise TypeError(f"{path} must define reward(obs, action, info, default)")
    return fn


def shaped_reward(
    skill: str,
    obs: Any,
    action: Any,
    info: dict,
    default: float,
    *,
    root: Path | None = None,
) -> float:
    fn = load_reward(skill, root)
    if fn is None:
        return float(default)
    return float(fn(obs, action, info, float(default)))
