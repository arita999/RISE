#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR/deploy/packages/openpi-client/src:${PYTHONPATH:-}"

HOST="${HOST:-localhost}"
PORT="${PORT:-8000}"
PROMPT="${PROMPT:-Untangle the parts}"

SIDE_IMAGE_TOPIC="${SIDE_IMAGE_TOPIC:-/camera_side/color/image_raw}"
REAR_IMAGE_TOPIC="${REAR_IMAGE_TOPIC:-/camera_rear/color/image_raw}"
ONHAND_IMAGE_TOPIC="${ONHAND_IMAGE_TOPIC:-/camera_onhand/color/image_raw}"
JOINT_STATE_TOPIC="${JOINT_STATE_TOPIC:-/joint_states}"
JOINT_CMD_TOPIC="${JOINT_CMD_TOPIC:-/joint_command}"
JOINT_NAMES="${JOINT_NAMES:-joint0,joint1,joint2,joint3,joint4,joint5,joint6}"

exec python "$ROOT_DIR/deploy/star_ai_robot_arm_deploy.py" \
  --host "$HOST" \
  --port "$PORT" \
  --prompt "$PROMPT" \
  --side_image_topic "$SIDE_IMAGE_TOPIC" \
  --rear_image_topic "$REAR_IMAGE_TOPIC" \
  --onhand_image_topic "$ONHAND_IMAGE_TOPIC" \
  --joint_state_topic "$JOINT_STATE_TOPIC" \
  --joint_cmd_topic "$JOINT_CMD_TOPIC" \
  --joint_names "$JOINT_NAMES" \
  "$@"
