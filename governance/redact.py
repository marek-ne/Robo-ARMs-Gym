"""Drop secret-looking fields before a receipt is stored or printed."""

from __future__ import annotations

from typing import Any

_FORBIDDEN = ("key", "token", "secret", "password", "authorization", "credential")


def is_secret_field(name: str) -> bool:
    lowered = name.lower().replace("-", "_")
    return any(part in lowered for part in _FORBIDDEN)


def public_mapping(data: Any, *, default_status: str = "absent") -> dict[str, Any]:
    """Copy a receipt dict with secret fields removed.

    Non-dicts become ``{"status": default_status}``.
    """
    if not isinstance(data, dict):
        return {"status": default_status}
    out: dict[str, Any] = {}
    for key, value in data.items():
        if is_secret_field(str(key)):
            continue
        if isinstance(value, dict):
            out[str(key)] = public_mapping(value, default_status=default_status)
        else:
            out[str(key)] = value
    if not out:
        return {"status": default_status}
    return out
