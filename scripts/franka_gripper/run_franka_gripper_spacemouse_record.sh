#!/usr/bin/env bash
set -euo pipefail

python scripts/franka_gripper/franka_gripper_spacemouse_record.py \
  --dataset-name franka_gripper_spacemouse \
  --num-episodes 1 \
  --episode-time 60 \
  --task "Teleoperate Franka + gripper with SpaceMouse." \
  --fps 30 \
  --robot-ip 172.10.0.1 \
  --robot-port 5000 \
  --gripper-port /dev/ttyACM1 \
  --gripper-baud 115200 \
  --gripper-home 1.0 \
  --camera wrist=233522071373 \
  --camera third_person=218622278749 \
  --camera-width 640 \
  --camera-height 480 \
  --use-depth
