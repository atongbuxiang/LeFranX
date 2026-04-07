import minimalmodbus
import numpy as np


class ZhixingDriver:
    POSITION_HIGH_8_REG = 0x0102
    POSITION_LOW_8_REG = 0x0103
    SPEED_REG = 0x0104
    FORCE_REG = 0x0105
    MOTION_TRIGGER_REG = 0x0108
    ID_REG = 0x0810

    POS_ARR_REG = 0x0602
    FORCE_ARR_REG = 0x0601

    FEEDBACK_POS_HIGH_8_REG = 0x0609
    FEEDBACK_POS_LOW_8_REG = 0x060A

    MAX_POS = 0
    MIN_POS = 12000

    MAX_FEEDBACK_POS = 3000
    OPEN_FEEDBACK_POS = 0
    CLOSE_FEEDBACK_POS = 3000

    def __init__(self, serial_dev: str, baud: int = 115200, slave_id: int = 1, timeout_s: float = 1.0):
        self.serial_dev = serial_dev
        self.baud = baud
        self.slave_id = slave_id
        self.timeout_s = timeout_s
        self.instrument = None

    def start(self):
        self.instrument = minimalmodbus.Instrument(self.serial_dev, self.slave_id)
        self.instrument.debug = False
        self.instrument.serial.baudrate = self.baud
        self.instrument.serial.timeout = self.timeout_s

    def stop(self):
        if self.instrument is not None and getattr(self.instrument, "serial", None) is not None:
            try:
                self.instrument.serial.close()
            except Exception:
                pass
        self.instrument = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def write_position(self, value: int):
        self.instrument.write_long(self.POSITION_HIGH_8_REG, value)

    def trigger_motion(self):
        self.instrument.write_register(self.MOTION_TRIGGER_REG, 1, functioncode=6)

    def read_force(self) -> int:
        return self.instrument.read_register(self.FORCE_REG, functioncode=3)

    def set_force(self, force: int):
        self.instrument.write_register(self.FORCE_REG, force, functioncode=6)

    def move_to(self, normalized_pos: float):
        normalized_pos = float(np.clip(normalized_pos, 0.0, 1.0))
        pos = int(normalized_pos * (self.MAX_POS - self.MIN_POS) + self.MIN_POS)
        self.write_position(pos)
        self.trigger_motion()

    def read_pos(self) -> float:
        pos_high8 = self.instrument.read_long(self.FEEDBACK_POS_HIGH_8_REG)
        if pos_high8 > 10000:
            pos_high8 = 0

        if not 0 <= pos_high8 <= self.MAX_FEEDBACK_POS:
            raise AssertionError(f"Invalid feedback position: {pos_high8}")

        real_width = (pos_high8 - self.CLOSE_FEEDBACK_POS) / (
            self.OPEN_FEEDBACK_POS - self.CLOSE_FEEDBACK_POS
        )
        if not 0.0 <= real_width <= 1.0:
            raise AssertionError(f"Invalid normalized gripper position: {real_width}")

        return float(real_width)
