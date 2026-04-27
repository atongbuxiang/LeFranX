#!/usr/bin/env bash

PROMPT=${PROMPT:-"Pick up the green apple and place it in the upper left corner of the table."}

python scripts/franka_gripper/openpi_remote_client.py \
  --remote-host "${REMOTE_HOST:-10.1.83.246}" \
  --remote-port "${REMOTE_PORT:-8000}" \
  --openpi-root "${OPENPI_ROOT:-/home/rognuc/openpi0.5}" \
  --prompt "${PROMPT}" \
  --fps "${FPS:-30}" \
  --robot-ip "${ROBOT_IP:-172.16.0.1}" \
  --robot-port "${ROBOT_PORT:-5000}" \
  --gripper-port "${GRIPPER_PORT:-/dev/ttyUSB0}" \
  --gripper-baud "${GRIPPER_BAUD:-115200}" \
  --gripper-home "${GRIPPER_HOME:-1.0}" \
  --camera realsense_wrist="${REALSENSE_WRIST_ID:-241122305042}" \
  --camera realsense_topdown="${REALSENSE_TOPDOWN_ID:-241122300571}" \
  --base-camera realsense_topdown \
  --wrist-camera realsense_wrist \
  --camera-width "${CAMERA_WIDTH:-640}" \
  --camera-height "${CAMERA_HEIGHT:-480}"
