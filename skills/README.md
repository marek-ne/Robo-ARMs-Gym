# Skills

A **skill** is a folder plus a `skill.yaml` manifest, not training code. The
policy net is referenced from the manifest (`policy.checkpoint_uri`) after an
eval is promoted. That split matches the base+adapter pattern even while reach
is still a standalone SB3 PPO net.

```
skills/
  schema.json
  registry.py
  reach/skill.yaml     # EE to a 3D point (no object)
  place/skill.yaml     # pick block at A, set down at B
```

## Manifest

Required blocks (see `schema.json` and WI-0.1 in `docs/IMPLEMENTATION_PLAN.md`):

| Block | Role |
|-------|------|
| `task_spec` | Observation tokens, action, horizon |
| `preconditions` | Robot id, DoF, workspace, camera |
| `policy` | Kind (`sb3-ppo` / `sb3-sac` / `lerobot-act` / `vla-lora`), optional base, checkpoint URI |
| `eval_certificate` | Gates (`min_success_rate`, `min_episodes`) + `eval_id` filled on promote |
| `safety_envelope` | Velocity/torque fractions, e-stop, geofence |

Nothing in this tree talks to hardware. Sim-first: a skill can be listed, loaded,
and catalogued with `checkpoint_uri: null`.

## Registry API

```python
from skills.registry import list_skills, load, validate, write_certificate, register_in_catalog

list_skills()                          # all skills/*/skill.yaml
load("reach")                          # latest version
load("reach", "0.1.0")                 # pinned
validate(manifest)                     # ValueError with field path
write_certificate("reach", "0.1.0", eval_id="...", checkpoint_uri="...")
register_in_catalog(manifest)          # catalog stage="skills"
```

Register every on-disk skill into the artifact catalog:

```bash
python3 -m skills.registry              # upsert all
python3 -m skills.registry list
python3 -m skills.registry create my-job --from place --summary "pass = …"
```

`create` copies a template, clears `eval_id` / `checkpoint_uri`, writes
`skills/<name>/skill.yaml`, and upserts the catalog. It does not train.

The catalog wrapper is `platform/catalog.py` (fimpo `artifact_catalog`, or
`.cache/artifact_catalog.json` if that module is absent). Override the file with
`FIMPO_ARTIFACT_CATALOG`.
