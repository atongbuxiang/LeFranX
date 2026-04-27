#!/usr/bin/env bash
set -euo pipefail

python scripts/franka_fer/franka_fer_subarm_teleoperator.py \
  --fps "${FPS:-30}" \
  --robot-ip "${ROBOT_IP:-172.16.0.1}" \
  --robot-port "${ROBOT_PORT:-5000}" \
  --leader-port "${LEADER_PORT:-/dev/ttyACM0}" \
  --calibration-json "${CALIBRATION_JSON:-scripts/franka_gripper/subarm_cal.json}"
