# Headless stand-ins for robot-hardware modules (cluster only)

`examples/experiments/mappings.py` imports every experiment config, and even
the simulator config imports `franka_env`, which imports at module level:

| module | why it cannot be used on CLAIX | real use |
|---|---|---|
| `easyhid` | loads the native `libhidapi` (USB HID) | SpaceMouse |
| `pyrealsense2` | Intel RealSense camera driver | real-robot cameras |
| `pynput` | needs an X display at import | keyboard listener on the real robot |

`cluster/claix/env.sh` puts this folder first on `PYTHONPATH`, so these names
resolve to the stand-ins here. They import fine and raise a clear error only
if something actually tries to **use** the hardware. The simulator with a
synthetic operator never does (`SIM_DISABLE_SPACEMOUSE=1`).

They are only on the path when `env.sh` is sourced, i.e. only on the cluster.
Your PC and the training machine keep the real packages.
