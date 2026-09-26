#!/usr/bin/env python3
"""Find an EGL device on which MuJoCo can render inside this job.

Prints one line: "" (the default device works), a device index for
MUJOCO_EGL_DEVICE_ID, or "none". With -v it also prints each attempt.
With --bench a second line gives the seconds per 128x128 frame on that
device (999 if it timed out): a node whose EGL path is broken can pass
the one-frame test yet render at ~10 s per frame (seen on CLAIX
n23g0004, where the actor then aborted inside the renderer).
Exit code 0 if some device works, 1 otherwise.
"""

import os
import subprocess
import sys

TEST = """
import os
import mujoco
m = mujoco.MjModel.from_xml_string(
    '<mujoco><worldbody><light pos="0 0 1"/><geom size="0.1"/></worldbody></mujoco>')
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, 32, 32)
r.update_scene(d)
img = r.render()
assert img.shape == (32, 32, 3), img.shape
print("ok", int(img.sum()))
"""

BENCH = """
import time
import mujoco
m = mujoco.MjModel.from_xml_string(
    '<mujoco><worldbody><light pos="0 0 1"/><geom size="0.1"/>'
    '<geom type="box" size="0.05 0.05 0.05" pos="0.2 0 0"/></worldbody></mujoco>')
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, 128, 128)
r.update_scene(d)
r.render()
t = time.perf_counter()
for _ in range(5):
    r.update_scene(d)
    r.render()
print("spf", (time.perf_counter() - t) / 5)
"""


def _env(dev):
    env = dict(os.environ, MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")
    if dev is None:
        env.pop("MUJOCO_EGL_DEVICE_ID", None)
    else:
        env["MUJOCO_EGL_DEVICE_ID"] = str(dev)
    return env


def seconds_per_frame(dev, timeout=40):
    """Mean seconds per 128x128 frame on ``dev``; 999 on timeout/failure."""
    try:
        r = subprocess.run([sys.executable, "-c", BENCH], env=_env(dev),
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 999.0
    for line in r.stdout.splitlines():
        if line.startswith("spf "):
            return float(line.split()[1])
    return 999.0


def works(dev, verbose):
    env = _env(dev)
    try:
        r = subprocess.run([sys.executable, "-c", TEST], env=env,
                           capture_output=True, text=True, timeout=90)
    except subprocess.TimeoutExpired:
        return False
    ok = r.returncode == 0 and r.stdout.startswith("ok")
    if verbose:
        tail = (r.stdout + r.stderr).strip().splitlines()[-1:] or [""]
        print(f"  device={dev!s:>4}: {'OK' if ok else 'fail'}  {tail[0][:120]}",
              file=sys.stderr)
    return ok


def main():
    verbose = "-v" in sys.argv
    for dev in [None] + list(range(8)):
        if works(dev, verbose):
            print("" if dev is None else dev)
            if "--bench" in sys.argv:
                spf = seconds_per_frame(dev)
                if verbose:
                    print(f"  device={dev!s:>4}: {spf:.4f} s per 128x128 frame",
                          file=sys.stderr)
                print(f"{spf:.4f}")
            return 0
    print("none")
    return 1


if __name__ == "__main__":
    sys.exit(main())
