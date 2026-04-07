#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.

from dataclasses import dataclass
from ..config import TeleoperatorConfig

@TeleoperatorConfig.register_subclass("sofranka")
@dataclass
class SoFrankaConfig(TeleoperatorConfig):
    # 连接机械臂的串口地址 (例如 /dev/ttyACM0)
    port: str
    
    # 是否使用角度单位，默认为 False (使用 -100 到 100 范围)
    use_degrees: bool = False