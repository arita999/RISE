#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY_DIR="$ROOT_DIR/policy_and_value/policy_offline_and_value"
export PYTHONPATH="$POLICY_DIR/src:${PYTHONPATH:-}"

CONFIG="${CONFIG:-Pi05_style_training}"
EXP_NAME="${EXP_NAME:-Pi05_style_training}"
PORT="${PORT:-8000}"
PROMPT="${PROMPT:-Untangle the parts}"
CHECKPOINT_DIR="${1:-${CHECKPOINT_DIR:-}}"

if [[ -z "$CHECKPOINT_DIR" ]]; then
  CHECKPOINT_ROOT="$POLICY_DIR/checkpoints/$CONFIG/$EXP_NAME"
  if [[ ! -d "$CHECKPOINT_ROOT" ]]; then
    echo "Checkpoint root does not exist: $CHECKPOINT_ROOT" >&2
    echo "Training needs to save at least one checkpoint first." >&2
    exit 1
  fi

  LATEST_STEP="$(
    find "$CHECKPOINT_ROOT" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' \
      | awk '/^[0-9]+$/ { print }' \
      | sort -n \
      | tail -n 1
  )"
  if [[ -z "$LATEST_STEP" ]]; then
    echo "No numeric checkpoint directory found under: $CHECKPOINT_ROOT" >&2
    echo "Wait until training reaches the first save interval, or pass a checkpoint dir explicitly." >&2
    exit 1
  fi
  CHECKPOINT_DIR="$CHECKPOINT_ROOT/$LATEST_STEP"
fi

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
  echo "Checkpoint directory does not exist: $CHECKPOINT_DIR" >&2
  exit 1
fi
CHECKPOINT_DIR="$(cd "$CHECKPOINT_DIR" && pwd)"

cd "$POLICY_DIR"
echo "Serving config: $CONFIG"
echo "Serving checkpoint: $CHECKPOINT_DIR"
echo "Port: $PORT"
echo "Default prompt: $PROMPT"

exec python scripts/serve_policy.py \
  --port "$PORT" \
  --default-prompt "$PROMPT" \
  policy:checkpoint \
  --policy.config "$CONFIG" \
  --policy.dir "$CHECKPOINT_DIR"
