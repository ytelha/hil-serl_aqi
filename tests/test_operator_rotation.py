"""Tests for the synthetic operator's rotation control (operator_model.py):
off, the operator is the original translation-only one; on, it turns the tool
back to an insertable orientation, commits only once aligned, treats a twisted
tool in contact like a jam, and caps translation and rotation separately.
A fake oracle stands in for the simulator, so no MuJoCo is needed.

Run from the repo root with
    python -m pytest tests/test_operator_rotation.py -q
    python tests/test_operator_rotation.py
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
for _p in ("serl_launcher", "examples"):
    sys.path.insert(0, str(REPO / _p))

from serl_launcher.aqi.operator_model import (PhaseOracle,  # noqa: E402
                                              SyntheticOperator,
                                              make_operator_theta)

ACTION_SCALE = 0.01      # env units per command unit (sim: 1 unit ~ 100 mm/s)
ROT_SCALE = 0.05         # rad per env unit on the rotation dims (sim env)


class _FakeOracle(PhaseOracle):
    lateral_dims = (0, 1)
    vertical_dim = 2
    rotation_dims = (3, 4, 5)

    def __init__(self, lat=(0.0, 0.0), h=30.0, rot=(0.0, 0.0, 0.0)):
        self.lat, self.h, self.rot = np.array(lat, float), float(h), np.array(rot, float)

    def lateral_error(self, obs):
        return self.lat.copy()

    def height(self, obs):
        return self.h

    def in_contact(self, obs):
        return self.h <= 4.0

    def jammed(self, obs):
        return self.in_contact(obs) and float(np.linalg.norm(self.lat)) > 5.0

    def commit_threshold(self):
        return 4.0

    def approach_height(self):
        return 25.0

    def action_dim(self):
        return 6

    def orientation_error(self, obs):
        return self.rot.copy()

    def rot_rad_per_env_unit(self):
        return ROT_SCALE


def _op(oracle, tier="expert", rotation=True, seed=0):
    th = make_operator_theta(tier, rng=np.random.default_rng(1))
    return SyntheticOperator(oracle, th, rng=np.random.default_rng(seed),
                             action_scale=ACTION_SCALE, rotation_control=rotation)


def test_off_leaves_rotation_dims_alone_and_matches_the_default():
    orc = _FakeOracle(lat=(2.0, -1.0), h=20.0, rot=(0.05, -0.02, 0.1))
    a, b = _op(orc, rotation=False), _op(orc, rotation=False)
    default = SyntheticOperator(orc, a.th, rng=np.random.default_rng(0), action_scale=ACTION_SCALE)
    assert not a.rotation_control
    for _ in range(50):
        u = a._intent_phase(None)
        assert np.all(u[3:] == 0.0)                     # intent never commands rotation
        assert np.array_equal(b.act(None), default.act(None))


def test_rotation_command_points_at_the_target_and_is_capped():
    orc = _FakeOracle(rot=(0.0, 0.0, 0.05))
    op = _op(orc)
    u = op._intent_phase(None)
    want = op.ROT_GAIN * 0.05 / (ACTION_SCALE * ROT_SCALE)
    assert abs(u[5] - want) < 1e-9 and u[3] == 0.0 and u[4] == 0.0
    orc.rot = np.array([1.0, -1.0, 0.0])                # large twist -> per-axis cap
    u = op._intent_phase(None)
    assert u[3] == op.th.v_max and u[4] == -op.th.v_max


def test_no_commit_while_twisted():
    orc = _FakeOracle(lat=(0.0, 0.0), h=20.0, rot=(0.0, 0.0, 0.06))
    op = _op(orc)
    op.phase = "align"
    op._premature = False
    for _ in range(30):
        op._intent_phase(None)
    assert op.phase == "align"                          # aligned laterally, not in yaw
    orc.rot[:] = 0.0
    op._intent_phase(None)
    assert op.phase == "descend"


def test_twisted_tool_in_contact_is_a_jam():
    orc = _FakeOracle(lat=(0.0, 0.0), h=2.0, rot=(0.0, 0.0, 0.12))
    op = _op(orc)
    op.th.q_rec = 1.0                                   # expert-style recovery
    op.phase = "descend"
    op._intent_phase(None)
    assert op.phase == "retract"
    op2 = _op(orc)
    op2.th.q_rec = 0.0                                  # push through, but keep turning
    op2.phase = "descend"
    u = op2._intent_phase(None)
    assert op2.phase == "descend" and u[2] < 0.0 and u[5] > 0.0
    off = _op(orc, rotation=False)                      # the original operator ignores the twist
    off.phase = "descend"
    off._intent_phase(None)
    assert off.phase == "descend"


def test_closed_loop_removes_the_twist():
    for tier, tol in (("expert", SyntheticOperator.ROT_TOL), ("novice", 2 * SyntheticOperator.ROT_TOL)):
        orc = _FakeOracle(lat=(0.0, 0.0), h=30.0, rot=(0.15, -0.10, 0.25))
        op = _op(orc, tier=tier)
        errs = []
        for _ in range(80):
            a = op.act(None)                            # env units
            orc.rot = orc.rot - a[3:6] * ROT_SCALE      # small-angle kinematics in the action frame
            errs.append(float(np.linalg.norm(orc.rot)))
        assert np.mean(errs[-20:]) < tol, (tier, errs[-5:])
        off = _op(_FakeOracle(rot=(0.15, -0.10, 0.25)), tier=tier, rotation=False)
        assert np.linalg.norm(off.oracle.rot) > 0.3     # without control nothing turns it back


def test_translation_and_rotation_have_separate_caps():
    orc = _FakeOracle(lat=(50.0, 50.0), h=60.0, rot=(1.0, 1.0, 1.0))
    op = _op(orc, tier="novice")
    for _ in range(30):
        a = op.act(None) / ACTION_SCALE
        assert np.linalg.norm(a[:3]) <= op.th.v_max + 1e-6
        assert np.linalg.norm(a[3:]) <= op.th.v_max + 1e-6
    assert np.linalg.norm(a[:3]) > 0.5 * op.th.v_max      # translation is not starved by rotation


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
