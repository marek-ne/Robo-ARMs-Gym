"""CPU digital twin of the 12V SO-101 (official MuJoCo MJCF)."""

from __future__ import annotations

TWIN_IDS = ("so101-twin-a", "so101-twin-b")
KIND = "digital-twin"
ROBOT = "so101-12v"
DEFAULT_SKILL = "reach"
