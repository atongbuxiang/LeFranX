import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


class SpaceMouseStateReader:
    def __init__(self, poll_interval_s: float = 0.001):
        self.poll_interval_s = poll_interval_s
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        self._state = None

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._thread = None

    def get_state(self) -> dict[str, Any] | None:
        with self._lock:
            if self._state is None:
                return None
            return dict(self._state)

    def _reader_loop(self):
        import pyspacemouse

        try:
            with pyspacemouse.open() as device:
                while not self._stop_event.is_set():
                    state = device.read()
                    if state is not None:
                        parsed = self._normalize_state(state)
                        with self._lock:
                            self._state = parsed
                    time.sleep(self.poll_interval_s)
        except Exception as exc:
            logger.error("SpaceMouse reader stopped: %s", exc)

    @staticmethod
    def _normalize_state(state) -> dict[str, Any]:
        buttons = getattr(state, "buttons", None)
        if buttons is None:
            buttons = [
                bool(getattr(state, "button0", False)),
                bool(getattr(state, "button1", False)),
            ]

        return {
            "x": float(getattr(state, "x", 0.0)),
            "y": float(getattr(state, "y", 0.0)),
            "z": float(getattr(state, "z", 0.0)),
            "roll": float(getattr(state, "roll", 0.0)),
            "pitch": float(getattr(state, "pitch", 0.0)),
            "yaw": float(getattr(state, "yaw", 0.0)),
            "buttons": list(buttons),
        }
