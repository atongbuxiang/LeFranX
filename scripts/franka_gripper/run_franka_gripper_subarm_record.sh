#!/usr/bin/env bash
set -euo pipefail

python scripts/franka_gripper/franka_gripper_subarm_record.py \
  --dataset-name franka_gripper_subarm \
  --task "Teleoperate Franka + gripper with subarm." \
  --fps 30 \
  --robot-ip 172.10.0.1 \
  --robot-port 5000 \
  --gripper-port /dev/ttyUSB0 \
  --gripper-baud 115200 \
  --gripper-home 1.0 \
  --leader-port /dev/ttyACM0 \
  --camera wrist=233522071373 \
  --camera third_person=218622278749 \
  --camera-width 640 \
  --camera-height 480 \
  --use-depth
