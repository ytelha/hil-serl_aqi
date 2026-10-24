"""Tests in the MuJoCo peg env (skipped without MuJoCo): the oracle's tilt /
yaw split of the orientation error, and the env's optional orientation box
(rpy_box, $SIM_RPY_BOX) under random rotation actions. No images, CPU only.

Run from the repo root with
    python -m pytest tests/test_sim_orientation.py -q
    python tests/test_sim_orientation.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
for _p in ("serl_launcher", "examples", "franka_sim", "serl_robot_infra"):
    sys.path.insert(0, str(REPO / _p))


def _sim():
    if importlib.util.find_spec("mujoco") is None:
        print("SKIP (no MuJoCo)")
        return None
    from experiments.sim_peg_insertion.wrapper import (SimPegEnv,
                                                       SimPegPhaseOracle)
    return SimPegEnv, SimPegPhaseOracle


def _env(shape, box=None):
    SimPegEnv, SimPegPhaseOracle = _sim()
    env = SimPegEnv(fake_env=True, image_obs=False, peg_shape=shape, rpy_box=box)
    return env, SimPegPhaseOracle(env, peg_shape=shape)


def test_tilt_yaw_split():
    if _sim() is None:
        return
    env, orc = _env("cuboid")
    env.set_eval_reset_pose(0.0, 0.2)
    env.reset()
    tilt, yaw = orc.tilt_yaw_error(None)
    assert abs(yaw - 0.2) < 0.03 and tilt < 0.03, (tilt, yaw)
    # the EE-frame error the operator uses has the same size
    assert abs(np.linalg.norm(orc.orientation_error(None)) - np.hypot(tilt, yaw)) < 1e-9
    env_c, orc_c = _env("cylinder")
    env_c.set_eval_reset_pose(0.0, 0.2)
    env_c.reset()
    assert orc_c.tilt_yaw_error(None)[1] == 0.0          # yaw is free in a round bore
    assert orc.orientation_tolerance() == (0.10, 0.25)
    assert orc_c.orientation_tolerance()[1] is None
    env.close(), env_c.close()


def test_box_bounds_the_tool_under_random_rotations():
    if _sim() is None:
        return
    spec = importlib.util.spec_from_file_location(
        "orientation_box", REPO / "franka_sim" / "franka_sim" / "envs" / "orientation_box.py")
    ob = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ob)
    box = np.array([0.05, 0.05, 0.15])
    worst = {}
    for name, b in (("box", box), ("free", None)):
        env, _orc = _env("cuboid", b)
        env.reset(seed=0)
        rng = np.random.default_rng(0)
        w_des = w_act = 0.0
        for _ in range(120):
            a = np.zeros(7)
            a[3:6] = rng.uniform(-1.0, 1.0, 3)
            env.step(a)
            dev = np.abs(ob.rpy_deviation(env._des_quat, env._reset_quat))
            _pos, q_act = env._pinch_pose()
            act = np.abs(ob.rpy_deviation(q_act, env._reset_quat))
            if b is not None:
                assert np.all(dev <= box + 1e-9), dev
            w_des, w_act = max(w_des, float(np.max(dev[:2]))), max(w_act, float(np.max(act[:2])))
        worst[name] = (w_des, w_act, env.rpy_clip_steps)
        env.close()
    assert worst["box"][1] < box[0] + 0.05, worst          # the tool follows the box
    assert worst["box"][2] > 0 and worst["free"][2] == 0
    assert worst["free"][0] > box[0], worst                 # without it, it tilts further


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
