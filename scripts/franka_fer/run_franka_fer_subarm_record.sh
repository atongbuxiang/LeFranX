#!/usr/bin/env bash
set -euo pipefail

python scripts/franka_fer/franka_fer_subarm_record.py \
  --dataset-name "${DATASET_NAME:-franka_fer_subarm}" \
  --task "${TASK:-Teleoperate Franka arm with subarm.}" \
  --fps "${FPS:-30}" \
  --robot-ip "${ROBOT_IP:-172.16.0.1}" \
  --robot-port "${ROBOT_PORT:-5000}" \
  --leader-port "${LEADER_PORT:-/dev/ttyACM0}" \
  --calibration-json "${CALIBRATION_JSON:-scripts/franka_gripper/subarm_cal.json}" \
  --camera wrist="${REALSENSE_WRIST_ID:-241122305042}" \
  --camera third_person="${REALSENSE_THIRD_PERSON_ID:-241122300571}" \
  --camera-width "${CAMERA_WIDTH:-640}" \
  --camera-height "${CAMERA_HEIGHT:-480}"
