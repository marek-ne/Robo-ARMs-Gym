#!/usr/bin/env bash
# Register a self-hosted runner on Node D. Run this on Node D, not in the public tree's CI.
# Requires: gh auth, a user that can read the GPU, and nvidia-smi.
set -euo pipefail

REPO="${ROBOARM_RUNNER_REPO:-marek-ne/Robo-ARMs-Gym}"
NAME="${ROBOARM_RUNNER_NAME:-robo-arm-node-d}"
DIR="${ROBOARM_RUNNER_DIR:-$HOME/actions-runner-robo-arm}"

if ! command -v nvidia-smi >/dev/null; then
  echo "nvidia-smi is not on this machine. The runner belongs on the GPU node." >&2
  exit 1
fi

mkdir -p "$DIR"
cd "$DIR"
if [[ ! -x ./config.sh ]]; then
  VERSION="${ROBOARM_RUNNER_VERSION:-}"
  if [[ -z "$VERSION" ]]; then
    VERSION=$(gh api repos/actions/runner/releases/latest --jq .tag_name | sed 's/^v//')
  fi
  curl -fsSL -o actions-runner.tar.gz \
    "https://github.com/actions/runner/releases/download/v${VERSION}/actions-runner-linux-x64-${VERSION}.tar.gz"
  tar xzf actions-runner.tar.gz
fi

TOKEN=$(gh api -X POST "repos/${REPO}/actions/runners/registration-token" --jq .token)
./config.sh --url "https://github.com/${REPO}" --token "$TOKEN" --name "$NAME" --labels "robo-arm,gpu" --unattended --replace
echo "Start it with: cd $DIR && ./run.sh"
echo "Or install the service from this directory: sudo ./svc.sh install && sudo ./svc.sh start"
