#!/usr/bin/env bash
set -euo pipefail

python scripts/franka_gripper/franka_gripper_spacemouse_record.py \
  --dataset-name franka_gripper_spacemouse \
  --task "Teleoperate Franka + gripper with SpaceMouse." \
  --fps 30 \
  --robot-ip 172.16.0.1 \
  --robot-port 5000 \
  --gripper-port /dev/ttyUSB0 \
  --gripper-baud 115200 \
  --gripper-home 1.0 \
  --camera wrist=241122305042 \
  --camera-width 640 \
  --camera-height 480 \
