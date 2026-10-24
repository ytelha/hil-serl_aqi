"""Tests for the optional orientation channel of the synthetic operator's
monitor (MonitoringSignal.set_orientation_limits, operator_gate.py): off, z_t
is exactly as before; on, a tilt or yaw error raises z_t near the surface
(1.0 at the limit), is free up to the limit higher up, is ignored inside the
bore, and shows up as the takeover channel 'orientation' (patch 0015 logging).

Without gymnasium (this PC) the stand-in of test_takeover_channels.py is used.

Run from the repo root with
    python -m pytest tests/test_orientation_channel.py -q
    python tests/test_orientation_channel.py
"""
import runpy
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
for _p in ("serl_launcher", "examples"):
    sys.path.insert(0, str(REPO / _p))
# installs the gymnasium stand-in when needed, and gives the toy helpers
_tc = runpy.run_path(str(REPO / "tests" / "test_takeover_channels.py"),
                     run_name="takeover_channels_helpers")

from serl_launcher.aqi.operator_gate import (MonitoringSignal,  # noqa: E402
                                             SyntheticOperatorWrapper)
from serl_launcher.aqi.operator_model import (SyntheticOperator,  # noqa: E402
                                              make_operator_theta)
from serl_launcher.aqi.toy_insertion import (ToyInsertionEnv,  # noqa: E402
                                             ToyInsertionOracle)


class _Orc:
    """Height (mm), lateral error (mm) and tilt / yaw error (rad), set by hand."""

    def __init__(self, h=8.0, err=0.0, tilt=0.0, yaw=0.0):
        self.h, self.err, self.tilt, self.yaw = h, err, tilt, yaw

    def height(self, obs):
        return self.h

    def lateral_error(self, obs):
        return np.array([self.err, 0.0])

    def approach_height(self):
        return 25.0

    def tilt_yaw_error(self, obs):
        return self.tilt, self.yaw


def _z(orc, tilt_lim=None, yaw_lim=None, steps=1):
    m = MonitoringSignal(orc, corridor_mm=5.0, corridor_height_mm=12.0,
                         hover_corridor_mm=12.5, budget_steps=300,
                         tilt_limit_rad=tilt_lim, yaw_limit_rad=yaw_lim)
    m.reset()
    for _ in range(steps):
        z = m(None)
    return z, m.last_channel


def test_off_by_default_and_exactly_as_before():
    orc = _Orc(h=8.0, err=1.0, tilt=0.5, yaw=0.5)          # badly twisted
    z_off, ch = _z(orc)
    assert ch != "orientation"
    m = MonitoringSignal(orc, corridor_mm=5.0, corridor_height_mm=12.0,
                         hover_corridor_mm=12.5, budget_steps=300)
    m.reset()
    assert m.tilt_limit is None and m.yaw_limit is None
    assert m(None) == z_off
    m.set_orientation_limits(0.1, 0.25)
    m.set_orientation_limits(None, None)                    # and off again
    m.reset()
    assert m(None) == z_off


def test_near_the_surface_one_at_the_limit():
    z, ch = _z(_Orc(h=8.0, tilt=0.05), tilt_lim=0.10, yaw_lim=0.25)
    assert abs(z - 0.5) < 1e-12 and ch == "orientation"
    z, ch = _z(_Orc(h=8.0, yaw=0.20), tilt_lim=0.10, yaw_lim=0.25)
    assert abs(z - 0.8) < 1e-12 and ch == "orientation"
    z, _ = _z(_Orc(h=8.0, tilt=0.3), tilt_lim=0.10, yaw_lim=0.25)
    assert z == 1.0                                         # saturates
    z, ch = _z(_Orc(h=8.0, yaw=0.20), tilt_lim=0.10, yaw_lim=None)
    assert ch != "orientation"                              # yaw free (round bore)


def test_higher_up_free_up_to_the_limit():
    z, ch = _z(_Orc(h=40.0, tilt=0.09), tilt_lim=0.10)
    assert ch != "orientation" and z < 0.01                 # within the limit
    z, ch = _z(_Orc(h=40.0, tilt=0.15), tilt_lim=0.10)
    assert abs(z - 0.5) < 1e-9 and ch == "orientation"      # 1.5 x the limit
    z, _ = _z(_Orc(h=40.0, tilt=0.25), tilt_lim=0.10)
    assert z == 1.0                                         # twice the limit


def test_inside_the_bore_the_walls_hold_the_orientation():
    z, ch = _z(_Orc(h=-5.0, tilt=0.3), tilt_lim=0.10)
    assert ch != "orientation"


def test_an_oracle_without_orientation_leaves_z_unchanged():
    orc = ToyInsertionOracle()
    a = MonitoringSignal(orc)
    b = MonitoringSignal(orc, tilt_limit_rad=0.1, yaw_limit_rad=0.2)
    a.reset(), b.reset()
    env = ToyInsertionEnv(seed=0)
    obs, info = env.reset()
    for _ in range(80):
        obs, _r, term, trunc, info = env.step(np.array([20.0, -30.0]))
        assert a(obs, info) == b(obs, info) and a.last_channel == b.last_channel
        if term or trunc:
            break


class _TwistOracle(ToyInsertionOracle):
    """The toy oracle plus a tool twist that grows during the episode, as an
    untrained policy's rotation commands would."""

    def tilt_yaw_error(self, obs):
        return 0.002 * float(obs["state"][-1] * 150), 0.0


def test_a_twist_becomes_a_takeover_channel():
    orc = _TwistOracle()
    mon = MonitoringSignal(orc, tilt_limit_rad=0.05)
    op = SyntheticOperator(orc, make_operator_theta("expert"),
                           rng=np.random.default_rng(0))
    env = SyntheticOperatorWrapper(
        _tc["_SucceedKey"](seed=1), op, mon, rng=np.random.default_rng(1),
        log_takeover_channels=True)
    env.reset()
    channels = []
    for _ in range(600):
        _o, _r, term, trunc, info = env.step(np.array([0.0, -20.0]))
        syn = info["synthetic_operator"]
        if "takeover_channel" in syn:
            channels.append(syn["takeover_channel"])
        if term or trunc:
            env.reset()
    assert "orientation" in channels, channels


def test_sweep_tolerance_rule():
    """oracle_insertion_test.tolerance: the largest swept angle up to which
    success stays >= 80 % of the success at 0 deg (the script imports MuJoCo,
    so the pure function is taken from its source)."""
    import ast
    src = (REPO / "examples" / "oracle_insertion_test.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "tolerance")
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "tolerance", "exec"), ns)
    tol = ns["tolerance"]
    rows = lambda pairs: [dict(angle_deg=a, success=s) for a, s in pairs]
    assert tol(rows([(0, 1.0), (4, 1.0), (8, 0.9), (12, 0.7), (15, 0.9)])) == 8
    assert tol(rows([(10, 0.5), (0, 0.5), (5, 0.4)])) == 10       # order-free
    assert tol(rows([(0, 0.0), (5, 0.0)])) is None
    assert tol(rows([(2, 1.0)])) is None                          # no 0 deg


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
