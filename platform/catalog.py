"""Thin wrapper around fimpo ``artifact_catalog``, with a local JSON fallback.

When ``FIMPO_RUNTIME`` points at an agent-runtime checkout, records go there.
Otherwise they land in ``.cache/artifact_catalog.json`` (or ``$FIMPO_ARTIFACT_CATALOG``).
"""

from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .paths import CATALOG_ENV, FIMPO_RUNTIME, catalog_path

_FIMPO: Any = None
_FIMPO_TRIED = False


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _try_fimpo():
    global _FIMPO, _FIMPO_TRIED
    if _FIMPO_TRIED:
        return _FIMPO
    _FIMPO_TRIED = True
    if FIMPO_RUNTIME is None:
        _FIMPO = None
        return None
    runtime = str(FIMPO_RUNTIME)
    if runtime not in sys.path:
        sys.path.insert(0, runtime)
    try:
        import artifact_catalog as catalog  # noqa: E402
    except Exception:
        _FIMPO = None
        return None
    _FIMPO = catalog
    return _FIMPO


def _bind_backend_path() -> Any:
    """Point the fimpo module at the currently resolved catalog file."""
    path = catalog_path()
    mod = _try_fimpo()
    if mod is not None:
        mod.CATALOG_PATH = path
    return mod


def _local_load() -> dict[str, Any]:
    path = catalog_path()
    if path.is_file():
        return json.loads(path.read_text())
    return {"version": 1, "artifacts": []}


def _local_save(data: dict[str, Any]) -> None:
    path = catalog_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def upsert_artifact(
    *,
    tenant: str,
    stage: str,
    uri: str,
    metrics: dict | None = None,
    promoted: bool = False,
    artifact_id: str | None = None,
) -> dict:
    """Insert or update a catalog record. Delegates to fimpo when available."""
    mod = _bind_backend_path()
    if mod is not None:
        return mod.upsert_artifact(
            tenant=tenant,
            stage=stage,
            uri=uri,
            metrics=metrics,
            promoted=promoted,
            artifact_id=artifact_id,
        )
    data = _local_load()
    aid = artifact_id or str(uuid.uuid4())
    existing = next(
        (
            a
            for a in data["artifacts"]
            if a["artifact_id"] == aid or a.get("uri") == uri
        ),
        None,
    )
    record = {
        "artifact_id": existing["artifact_id"] if existing else aid,
        "tenant": tenant,
        "stage": stage,
        "uri": uri,
        "minio_key": uri if uri.startswith("robo-arm/") else None,
        "metrics": metrics or {},
        "promoted": promoted,
        "updated_at": now_iso(),
        "created_at": existing["created_at"] if existing else now_iso(),
    }
    if existing:
        data["artifacts"] = [
            record if a["artifact_id"] == existing["artifact_id"] else a
            for a in data["artifacts"]
        ]
    else:
        data["artifacts"].append(record)
    _local_save(data)
    return record


def search_artifacts(
    *,
    tenant: str | None = None,
    stage: str | None = None,
    promoted: bool | None = None,
    q: str | None = None,
    limit: int = 50,
) -> list[dict]:
    mod = _bind_backend_path()
    if mod is not None:
        return mod.search_artifacts(
            tenant=tenant,
            stage=stage,
            promoted=promoted,
            q=q,
            limit=limit,
        )
    arts = _local_load()["artifacts"]
    out: list[dict] = []
    for a in arts:
        if tenant and a.get("tenant") != tenant:
            continue
        if stage and a.get("stage") != stage:
            continue
        if promoted is not None and bool(a.get("promoted")) != promoted:
            continue
        if q:
            blob = json.dumps(a).lower()
            if q.lower() not in blob:
                continue
        out.append(a)
    out.sort(key=lambda x: x.get("updated_at") or "", reverse=True)
    return out[:limit]


def search(
    *,
    tenant: str | None = None,
    stage: str | None = None,
    promoted: bool | None = None,
    q: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """Alias for ``search_artifacts`` (WI-0.1 wrapper surface)."""
    return search_artifacts(
        tenant=tenant,
        stage=stage,
        promoted=promoted,
        q=q,
        limit=limit,
    )


def upsert_skill(manifest: dict) -> dict:
    """Record a skill version in the artifact catalog (stage=skills)."""
    name = manifest["skill"]
    version = manifest["version"]
    uri = f"robo-arm/skills/{name}/{version}/skill.yaml"
    cert = manifest.get("eval_certificate") or {}
    policy = manifest.get("policy") or {}
    eval_id = cert.get("eval_id")
    return upsert_artifact(
        tenant="robo-arm",
        stage="skills",
        uri=uri,
        metrics={
            "skill": name,
            "version": version,
            "summary": manifest.get("summary"),
            "policy_kind": policy.get("kind"),
            "eval_id": eval_id,
            "checkpoint_uri": policy.get("checkpoint_uri"),
            "min_success_rate": cert.get("min_success_rate"),
            "min_episodes": cert.get("min_episodes"),
        },
        promoted=bool(eval_id),
        artifact_id=f"skill-{name}-{version}",
    )


__all__ = [
    "CATALOG_ENV",
    "catalog_path",
    "now_iso",
    "search",
    "search_artifacts",
    "upsert_artifact",
    "upsert_skill",
]
