"""Skill manifest registry: list, load, validate, certify, catalog.

A skill is a folder under ``skills/<name>/`` with a ``skill.yaml`` (D2).
Validation is against ``skills/schema.json``. Field-path errors are raised
as ``ValueError`` (e.g. ``preconditions.dof: ...``).
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import yaml

from platform.catalog import upsert_skill
from platform.paths import SKILLS_DIR

SCHEMA_PATH = SKILLS_DIR / "schema.json"

_SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)")


def _schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


def _field_path(parts: list[Any]) -> str:
    if not parts:
        return "$"
    out: list[str] = []
    for p in parts:
        if isinstance(p, int):
            if out:
                out[-1] = f"{out[-1]}[{p}]"
            else:
                out.append(f"[{p}]")
        else:
            out.append(str(p))
    return ".".join(out)


def _is_type(value: Any, type_name: str) -> bool:
    if type_name == "object":
        return isinstance(value, dict)
    if type_name == "array":
        return isinstance(value, list)
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "null":
        return value is None
    return True


def _type_names(schema: dict) -> list[str]:
    t = schema.get("type")
    if t is None:
        return []
    if isinstance(t, list):
        return [str(x) for x in t]
    return [str(t)]


def _validate_node(instance: Any, schema: dict, path: list[Any]) -> None:
    loc = _field_path(path)
    types = _type_names(schema)
    if types and not any(_is_type(instance, t) for t in types):
        got = "null" if instance is None else type(instance).__name__
        raise ValueError(f"{loc}: expected {'|'.join(types)}, got {got}")

    if "const" in schema and instance != schema["const"]:
        raise ValueError(f"{loc}: expected const {schema['const']!r}, got {instance!r}")

    if "enum" in schema and instance not in schema["enum"]:
        raise ValueError(f"{loc}: {instance!r} is not one of {schema['enum']}")

    if isinstance(instance, str):
        pattern = schema.get("pattern")
        if pattern and re.search(pattern, instance) is None:
            raise ValueError(f"{loc}: {instance!r} does not match pattern {pattern}")
        if "minLength" in schema and len(instance) < schema["minLength"]:
            raise ValueError(f"{loc}: shorter than minLength {schema['minLength']}")

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            raise ValueError(f"{loc}: {instance} is below minimum {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            raise ValueError(f"{loc}: {instance} is above maximum {schema['maximum']}")
        if "exclusiveMinimum" in schema and instance <= schema["exclusiveMinimum"]:
            raise ValueError(
                f"{loc}: {instance} is not above exclusiveMinimum {schema['exclusiveMinimum']}"
            )

    if isinstance(instance, list) and "items" in schema:
        if "minItems" in schema and len(instance) < schema["minItems"]:
            raise ValueError(f"{loc}: fewer than minItems {schema['minItems']}")
        items = schema["items"]
        if isinstance(items, dict):
            for i, item in enumerate(instance):
                _validate_node(item, items, path + [i])

    if isinstance(instance, dict):
        required = schema.get("required") or []
        for key in required:
            if key not in instance:
                raise ValueError(f"{_field_path(path + [key])}: missing required field")
        props = schema.get("properties") or {}
        additional = schema.get("additionalProperties", True)
        for key, value in instance.items():
            if key in props:
                _validate_node(value, props[key], path + [key])
            elif additional is False:
                raise ValueError(f"{_field_path(path + [key])}: additional property not allowed")
            elif isinstance(additional, dict):
                _validate_node(value, additional, path + [key])


def _validate_with_jsonschema(manifest: dict, schema: dict) -> None:
    import jsonschema

    validator = jsonschema.Draft7Validator(schema)
    errors = sorted(validator.iter_errors(manifest), key=lambda e: list(e.path))
    if not errors:
        return
    err = errors[0]
    loc = _field_path(list(err.absolute_path))
    raise ValueError(f"{loc}: {err.message}") from err


def validate(manifest: dict) -> None:
    """Raise ``ValueError`` with a dotted field path if the manifest is invalid."""
    if not isinstance(manifest, dict):
        raise ValueError("$: expected object, got " + type(manifest).__name__)
    schema = _schema()
    try:
        _validate_with_jsonschema(manifest, schema)
    except ImportError:
        _validate_node(manifest, schema, [])


def _iter_manifest_files() -> list[Path]:
    if not SKILLS_DIR.is_dir():
        return []
    return sorted(p for p in SKILLS_DIR.glob("*/skill.yaml") if p.is_file())


def _read_manifest(path: Path) -> dict:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: skill.yaml must be a mapping")
    return data


def _version_key(version: str) -> tuple[int, int, int]:
    m = _SEMVER.match(version or "")
    if not m:
        return (0, 0, 0)
    return tuple(int(x) for x in m.groups())  # type: ignore[return-value]


def list_skills() -> list[dict]:
    """Scan ``skills/*/skill.yaml`` and return parsed manifests."""
    out: list[dict] = []
    for path in _iter_manifest_files():
        out.append(_read_manifest(path))
    return out


def _matches(name: str, version: str | None) -> list[tuple[Path, dict]]:
    found: list[tuple[Path, dict]] = []
    for path in _iter_manifest_files():
        manifest = _read_manifest(path)
        if manifest.get("skill") != name:
            continue
        if version is not None and str(manifest.get("version")) != version:
            continue
        found.append((path, manifest))
    if version is None:
        found.sort(key=lambda item: _version_key(str(item[1].get("version", ""))), reverse=True)
    return found


def load(name: str, version: str | None = None) -> dict:
    """Load a skill manifest. ``version=None`` selects the highest semver."""
    found = _matches(name, version)
    if not found:
        ver = version if version is not None else "latest"
        raise FileNotFoundError(f"skill {name!r} version {ver!r} not found")
    manifest = copy.deepcopy(found[0][1])
    validate(manifest)
    return manifest


def write_certificate(
    name: str,
    version: str,
    *,
    eval_id: str,
    checkpoint_uri: str,
) -> dict:
    """Update ``eval_certificate.eval_id`` and ``policy.checkpoint_uri`` in place."""
    found = _matches(name, version)
    if not found:
        raise FileNotFoundError(f"skill {name!r} version {version!r} not found")
    path, manifest = found[0]
    manifest = copy.deepcopy(manifest)
    manifest.setdefault("eval_certificate", {})["eval_id"] = eval_id
    manifest.setdefault("policy", {})["checkpoint_uri"] = checkpoint_uri
    validate(manifest)
    dumped = yaml.safe_dump(
        manifest,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    )
    path.write_text(dumped if dumped.endswith("\n") else dumped + "\n")
    return manifest


def register_in_catalog(manifest: dict) -> dict:
    """Validate then upsert the skill version into the artifact catalog."""
    validate(manifest)
    return upsert_skill(manifest)


def create_skill(
    name: str,
    *,
    template: str = "place",
    summary: str | None = None,
    skills_dir: Path | None = None,
    register: bool = True,
) -> dict:
    """Copy a template skill.yaml into ``skills/<name>/`` with a clean certificate."""
    if not re.fullmatch(r"[a-z][a-z0-9_-]*", name):
        raise ValueError(f"skill name {name!r} must match [a-z][a-z0-9_-]*")
    root = Path(skills_dir) if skills_dir is not None else SKILLS_DIR
    dest_dir = root / name
    dest = dest_dir / "skill.yaml"
    if dest.is_file():
        raise FileExistsError(str(dest))
    manifest = load(template)
    manifest["skill"] = name
    manifest["version"] = "0.1.0"
    if summary:
        manifest["summary"] = summary
    manifest.setdefault("policy", {})["checkpoint_uri"] = None
    manifest.setdefault("eval_certificate", {})["eval_id"] = None
    validate(manifest)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dumped = yaml.safe_dump(
        manifest,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    )
    dest.write_text(dumped if dumped.endswith("\n") else dumped + "\n")
    if register:
        register_in_catalog(manifest)
    return manifest


def _register_all() -> None:
    for manifest in list_skills():
        validate(manifest)
        rec = register_in_catalog(manifest)
        print(
            f"registered {manifest.get('skill')}@{manifest.get('version')} "
            f"-> {rec.get('artifact_id')} {rec.get('uri')}"
        )


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Robo-arm skill registry")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("register", help="Validate and upsert every skills/*/skill.yaml")
    sub.add_parser("list", help="Print on-disk skills")
    pc = sub.add_parser("create", help="Copy a template into skills/<name>/")
    pc.add_argument("name")
    pc.add_argument("--from", dest="template", default="place")
    pc.add_argument("--summary", default=None)
    pc.add_argument("--no-register", action="store_true")
    args = p.parse_args(argv)
    if args.cmd in (None, "register"):
        _register_all()
        return
    if args.cmd == "list":
        for m in list_skills():
            print(f"{m.get('skill')}@{m.get('version')} eval_id={m.get('eval_certificate', {}).get('eval_id')}")
        return
    if args.cmd == "create":
        m = create_skill(
            args.name,
            template=args.template,
            summary=args.summary,
            register=not args.no_register,
        )
        print(f"created {m['skill']}@{m['version']}")
        return


if __name__ == "__main__":
    main()
