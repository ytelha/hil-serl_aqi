import threading
import time
import numpy as np
from franka_env.spacemouse import pyspacemouse
from typing import Tuple


class SpaceMouseExpert:
    """
    This class provides an interface to the SpaceMouse.
    It continuously reads the SpaceMouse state and provides
    a "get_action" method to get the latest action and button state.

    A background *thread* (not a forked process) does the polling. Threading is
    used deliberately: the HIL actor initialises JAX, which is multithreaded, and
    forking a multiprocessing child after JAX starts triggers a fork-safety
    warning and can deadlock / raise BrokenPipeError. A thread shares the address
    space and avoids that entirely.
    """

    def __init__(self):
        pyspacemouse.open()

        # Shared state protected by a lock (updated by the reader thread).
        self.latest_data = {"action": [0.0] * 6, "buttons": [0, 0, 0, 0]}
        self._lock = threading.Lock()
        self._running = True

        # Start a daemon thread to continuously read the SpaceMouse state.
        self.thread = threading.Thread(target=self._read_spacemouse, daemon=True)
        self.thread.start()

    def _read_spacemouse(self):
        while self._running:
            state = pyspacemouse.read_all()
            action = [0.0] * 6
            buttons = [0, 0, 0, 0]

            if len(state) == 2:
                action = [
                    -state[0].y, state[0].x, state[0].z,
                    -state[0].roll, -state[0].pitch, -state[0].yaw,
                    -state[1].y, state[1].x, state[1].z,
                    -state[1].roll, -state[1].pitch, -state[1].yaw
                ]
                buttons = state[0].buttons + state[1].buttons
            elif len(state) == 1:
                action = [
                    -state[0].y, state[0].x, state[0].z,
                    -state[0].roll, -state[0].pitch, -state[0].yaw
                ]
                buttons = state[0].buttons

            # Update the shared state
            with self._lock:
                self.latest_data["action"] = action
                self.latest_data["buttons"] = buttons

            # Yield the GIL so the polling loop does not starve the main thread
            # (pyspacemouse is opened non-blocking, so read_all returns at once).
            time.sleep(0.001)

    def get_action(self) -> Tuple[np.ndarray, list]:
        """Returns the latest action and button state of the SpaceMouse."""
        with self._lock:
            action = self.latest_data["action"]
            buttons = self.latest_data["buttons"]
        return np.array(action), buttons

    def close(self):
        self._running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)

