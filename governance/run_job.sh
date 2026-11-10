#!/usr/bin/env bash
# Run a guarded train or eval in a pull-request checkout.
# The gate code is imported from the trusted tree. Secrets must already be unset.
set -euo pipefail

if [[ $# -ne 9 ]]; then
  echo "usage: run_job.sh TRUSTED PR MODE GYM SKILL MODEL SNYK_JSON GATEWAY_JSON OUT_JSON" >&2
  exit 2
fi

TRUSTED="$1"
PR="$2"
MODE="$3"
GYM="$4"
SKILL="$5"
MODEL="$6"
SNYK="$7"
GATEWAY="$8"
OUT="$9"

cd "$PR"
export PYTHONPATH="$PR"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
unset AI_GATEWAY_KEY AI_GATEWAY_BASE_URL SNYK_TOKEN GITHUB_TOKEN GH_TOKEN || true

python -m pip install -e "$PR" >/tmp/roboarm-pip.log

CKPT=""
POLICY="scripted"
if [[ "$MODE" == "train" ]]; then
  python -m twin train --skill "$GYM" --steps 50000 --no-catalog
  CKPT=$(python - <<'PY'
import json
from pathlib import Path
runs = sorted(Path(".cache/twin-runs").glob("*/*/run.json"), key=lambda path: path.stat().st_mtime)
if not runs:
    raise SystemExit("no run.json written")
print(json.loads(runs[-1].read_text())["checkpoint"])
PY
  )
  POLICY="ppo"
elif [[ "$MODE" == "eval-zip" ]]; then
  CKPT="$MODEL"
  POLICY="ppo"
fi

if [[ -n "$CKPT" ]]; then
  python -m twin eval --skill "$GYM" --policy "$POLICY" --model "$CKPT" --episodes 50 > /tmp/roboarm-eval.json
else
  python -m twin eval --skill "$GYM" --policy scripted --episodes 50 > /tmp/roboarm-eval.json
fi

PYTHONPATH="$TRUSTED" python - "$SKILL" "$GYM" "$POLICY" "$CKPT" "$SNYK" "$GATEWAY" "$PR" "$OUT" <<'PY'
import json
import sys
from pathlib import Path

from governance.certificate import build_certificate, render_comment, stamp_manifest

skill, gym, policy, ckpt, snyk_path, gateway_path, repo, out = sys.argv[1:9]
eval_doc = json.loads(Path("/tmp/roboarm-eval.json").read_text(encoding="utf-8"))
certificate = build_certificate(
    skill=skill,
    gym=gym,
    success_rate=float(eval_doc["success_rate"]),
    n_episodes=int(eval_doc["n_episodes"]),
    mean_distance_cm=eval_doc.get("mean_distance_cm"),
    checkpoint=ckpt or None,
    snyk=json.loads(Path(snyk_path).read_text(encoding="utf-8")),
    gateway=json.loads(Path(gateway_path).read_text(encoding="utf-8")),
    repo=Path(repo),
    policy=policy,
)
dest = Path(out)
dest.write_text(json.dumps(certificate, indent=2) + "\n", encoding="utf-8")
dest.with_suffix(".md").write_text(render_comment(certificate), encoding="utf-8")
stamp_manifest(Path(repo), certificate)
print(json.dumps({"passed": certificate["passed"]}))
PY
