"""Tests for the sim env's optional orientation box
(franka_sim/franka_sim/envs/orientation_box.py): the rotation maths, that the
box leaves an orientation inside it untouched, and that it clips each axis
about the world axes around the home orientation. The module is numpy only and
is loaded by path, so neither MuJoCo nor scipy is needed; with scipy the
conventions are checked against scipy's as well.

Run from the repo root with
    python -m pytest tests/test_orientation_box.py -q
    python tests/test_orientation_box.py
"""
import importlib.util
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "orientation_box", REPO / "franka_sim" / "franka_sim" / "envs" / "orientation_box.py")
ob = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ob)

# the env's home orientation: tool pointing down (180 deg about world x)
HOME = np.array([1.0, 0.0, 0.0, 0.0])


def _rand_quat(rng):
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


def _rotvec_quat(rv):
    a = float(np.linalg.norm(rv))
    if a == 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return np.concatenate([np.sin(a / 2) * np.asarray(rv) / a, [np.cos(a / 2)]])


def _mul(q1, q2):
    """q1 * q2 (apply q2 first), xyzw."""
    return ob.mat_to_quat(ob.quat_to_mat(q1) @ ob.quat_to_mat(q2))


def test_quat_matrix_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(200):
        q = _rand_quat(rng)
        R = ob.quat_to_mat(q)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert abs(np.linalg.det(R) - 1.0) < 1e-12
        q2 = ob.mat_to_quat(R)
        assert q2[3] >= 0.0
        assert np.allclose(q2, q if q[3] >= 0 else -q, atol=1e-10)


def test_euler_round_trip():
    rng = np.random.default_rng(1)
    for _ in range(200):
        e = rng.uniform([-3.0, -1.5, -3.0], [3.0, 1.5, 3.0])
        assert np.allclose(ob.euler_xyz(ob.mat_from_euler_xyz(e)), e, atol=1e-9)


def test_inside_the_box_is_untouched():
    rng = np.random.default_rng(2)
    box = np.array([0.1, 0.1, 0.4])
    for _ in range(100):
        e = rng.uniform(-0.9 * box, 0.9 * box)
        q = ob.mat_to_quat(ob.mat_from_euler_xyz(e) @ ob.quat_to_mat(HOME))
        out, clipped = ob.clip_to_rpy_box(q, HOME, box)
        assert out is q and not clipped
        assert np.allclose(ob.rpy_deviation(q, HOME), e, atol=1e-9)


def test_each_axis_is_clipped_about_the_world_axes():
    box = np.array([0.1, 0.1, 0.4])
    for axis, ang in ((0, 0.3), (1, -0.25), (2, 0.6), (2, -0.9)):
        e = np.zeros(3)
        e[axis] = ang
        q = ob.mat_to_quat(ob.mat_from_euler_xyz(e) @ ob.quat_to_mat(HOME))
        out, clipped = ob.clip_to_rpy_box(q, HOME, box)
        dev = ob.rpy_deviation(out, HOME)
        want = np.zeros(3)
        want[axis] = np.sign(ang) * box[axis]
        assert clipped and np.allclose(dev, want, atol=1e-9), (axis, dev)
    # a world-z rotation of the home pose is pure yaw, as the env's reset yaw
    q = _mul(_rotvec_quat([0.0, 0.0, 0.07]), HOME)
    assert np.allclose(ob.rpy_deviation(q, HOME), [0.0, 0.0, 0.07], atol=1e-12)


def test_random_rotation_actions_stay_in_the_box():
    """The env applies drot = rotvec(a * 0.05) on the left each step; with the
    box, the desired orientation never leaves it, without it, it drifts."""
    rng = np.random.default_rng(3)
    box = np.array([0.1, 0.1, 0.4])
    q_box, q_free, n_clip = HOME.copy(), HOME.copy(), 0
    worst = 0.0
    for _ in range(300):
        drot = _rotvec_quat(rng.uniform(-1.0, 1.0, 3) * 0.05)
        q_box, clipped = ob.clip_to_rpy_box(_mul(drot, q_box), HOME, box)
        q_free = _mul(drot, q_free)
        n_clip += int(clipped)
        dev = np.abs(ob.rpy_deviation(q_box, HOME))
        assert np.all(dev <= box + 1e-9), dev
        worst = max(worst, float(np.max(np.abs(ob.rpy_deviation(q_free, HOME))[:2])))
    assert n_clip > 0 and worst > 0.1          # unbounded, the tool tilts further


def test_parse():
    assert ob.parse_rpy_box(None) is None and ob.parse_rpy_box("") is None
    assert ob.parse_rpy_box(" , ") is None
    assert np.allclose(ob.parse_rpy_box("0.1,0.1,0.4"), [0.1, 0.1, 0.4])
    assert np.allclose(ob.parse_rpy_box([0.05, 0.05, 0.2]), [0.05, 0.05, 0.2])
    for bad in ("0.1,0.1", "0,0.1,0.4", "0.1,0.1,2.0", "a,b,c"):
        try:
            ob.parse_rpy_box(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad!r}")


def test_matches_scipy_conventions():
    try:
        from scipy.spatial.transform import Rotation
    except ImportError:
        print("SKIP (no scipy)")
        return
    rng = np.random.default_rng(4)
    for _ in range(100):
        q = _rand_quat(rng)
        R = Rotation.from_quat(q)
        assert np.allclose(ob.quat_to_mat(q), R.as_matrix(), atol=1e-12)
        e = ob.euler_xyz(R.as_matrix())
        assert np.allclose(Rotation.from_euler("xyz", e).as_matrix(), R.as_matrix(), atol=1e-9)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
