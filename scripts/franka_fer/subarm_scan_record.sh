#!/usr/bin/env bash
set -euo pipefail

args=(
  scripts/franka_fer/franka_fer_subarm_tum_rgbd_record.py
  --dataset-path "${DATASET_PATH:-/mnt/data/box02}"
  --fps "${FPS:-30}"
  --camera-fps "${CAMERA_FPS:-30}"
  --robot-ip "${ROBOT_IP:-172.16.0.1}"
  --robot-port "${ROBOT_PORT:-5000}"
  --leader-port "${LEADER_PORT:-/dev/ttyACM0}"
  --calibration-json "${CALIBRATION_JSON:-scripts/franka_gripper/subarm_cal.json}"
  --camera-name "${CAMERA_NAME:-realsense_wrist}"
  --realsense-id "${REALSENSE_WRIST_ID:-241122305042}"
  --camera-width "${CAMERA_WIDTH:-640}"
  --camera-height "${CAMERA_HEIGHT:-480}"
)

if [[ "${OVERWRITE:-0}" == "1" ]]; then
  args+=(--overwrite)
fi

if [[ "${LEADER_NORMALIZED:-0}" == "1" ]]; then
  args+=(--leader-normalized)
fi

if [[ "${CALIBRATE_LEADER:-0}" == "1" ]]; then
  args+=(--calibrate-leader)
fi

python "${args[@]}"
