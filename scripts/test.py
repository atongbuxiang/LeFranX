# spacemouse test

import threading
import pyspacemouse
import time
#每次拿 sm_state 作为action
sm_state = None
sm_lock = threading.Lock()
def sm_reader():
    global sm_lock, sm_state
    try:
        with pyspacemouse.open() as device:
            while True:
                state = device.read()
                if state:
                    with sm_lock:
                        sm_state = state
                time.sleep(0.001) # 1000Hz 采样
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"SpaceMouse Error: {e}")

# homomorphic arm test
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_so100_leader,
    gamepad,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    so100_leader,
    so101_leader,
    sofranka,
)
from dataclasses import dataclass
from lerobot.teleoperators.sofranka import SoFrankaConfig
@dataclass
class TeleoperateConfig:
    # TODO: pepijn, steven: if more robots require multiple teleoperators (like lekiwi) its good to make this possibele in teleop.py and record.py with List[Teleoperator]
    teleop: TeleoperatorConfig
    # Limit the maximum frames per second.
    fps: int = 60
    teleop_time_s: float | None = None
    # Display all cameras on screen
    display_data: bool = False

teleop = make_teleoperator_from_config(SoFrankaConfig(port="/dev/ttyACM0"))
teleop.connect()
teleop.get_action()

#zhixing driver test
from franka_fer.zhixing_driver import ZhixingDriver
driver = ZhixingDriver(serial_dev="/dev/ttyACM0")
driver.start()
driver.move_to(0.5)
driver.stop()
