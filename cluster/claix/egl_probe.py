#!/usr/bin/env python3
"""Find an EGL device on which MuJoCo can render inside this job.

Prints one line: "" (the default device works), a device index for
MUJOCO_EGL_DEVICE_ID, or "none". With -v it also prints each attempt.
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


def works(dev, verbose):
    env = dict(os.environ, MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")
    if dev is None:
        env.pop("MUJOCO_EGL_DEVICE_ID", None)
    else:
        env["MUJOCO_EGL_DEVICE_ID"] = str(dev)
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
            return 0
    print("none")
    return 1


if __name__ == "__main__":
    sys.exit(main())
