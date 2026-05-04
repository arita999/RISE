#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR/deploy/packages/openpi-client/src:${PYTHONPATH:-}"

HOST="${HOST:-localhost}"
PORT="${PORT:-8000}"
PROMPT="${PROMPT:-Untangle the parts}"

exec python "$ROOT_DIR/deploy/star_ai_robot_arm_deploy_no_ros.py" \
  --host "$HOST" \
  --port "$PORT" \
  --prompt "$PROMPT" \
  "$@"
