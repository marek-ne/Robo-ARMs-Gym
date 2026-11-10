"""Robo-arm platform helpers (paths + artifact catalog wrapper).

This directory is named ``platform``, which shadows the stdlib module when the
repo root is on ``sys.path``. Re-export the stdlib ``platform`` API so modules
such as ``uuid`` keep working, then expose local helpers.
"""

from __future__ import annotations

import importlib.util
import sysconfig
from pathlib import Path

_STDLIB_PLATFORM = Path(sysconfig.get_path("stdlib")) / "platform.py"


def _load_stdlib_platform():
    spec = importlib.util.spec_from_file_location(
        "_stdlib_platform",
        _STDLIB_PLATFORM,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"stdlib platform not found at {_STDLIB_PLATFORM}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_stdlib_platform = _load_stdlib_platform()
for _name in dir(_stdlib_platform):
    if _name in {
        "__name__",
        "__file__",
        "__package__",
        "__path__",
        "__spec__",
        "__loader__",
        "__builtins__",
        "__cached__",
        "__doc__",
    }:
        continue
    globals()[_name] = getattr(_stdlib_platform, _name)
del _name

from .paths import CATALOG_ENV, REPO_ROOT, SKILLS_DIR, catalog_path  # noqa: E402

__all__ = ["CATALOG_ENV", "REPO_ROOT", "SKILLS_DIR", "catalog_path"]
