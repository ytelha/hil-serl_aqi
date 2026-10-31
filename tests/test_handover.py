"""Tests for the smooth hand-over of the synthetic operator (operator_gate.py,
--synthetic_handover_steps). Off, the wrapper behaves exactly as before and
adds no info key; on, the operator's motion starts from the robot's command,
the executed action ramps at takeover and at hand-back, and what ran after a
hand-back is reported so the replay buffer stores it.

Without gymnasium (this PC) a minimal stand-in for gym.Wrapper is installed;
with gymnasium the real one is used.

Run from the repo root with
    python -m pytest tests/test_handover.py -q
    python tests/test_handover.py
"""
import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parent.parent
for _p in ("serl_launcher", "examples"):
    sys.path.insert(0, str(REPO / _p))

try:
    import gymnasium
    _real_gym = hasattr(gymnasium.Wrapper, "step")    # False for a stub module
except ImportError:
    _real_gym = False
if not _real_gym:            # minimal stand-in: only what the wrapper uses
    _gym = types.ModuleType("gymnasium")
    _spaces = types.ModuleType("gymnasium.spaces")

    class _GymEnv:
        @property
        def unwrapped(self):
            return self

    class _Wrapper(_GymEnv):
        def __init__(self, env):
            self.env = env

        @property
        def unwrapped(self):
            return self.env.unwrapped

    _spaces.Box = lambda *a, **k: SimpleNamespace(args=a, kwargs=k)
    _spaces.Dict = lambda *a, **k: SimpleNamespace(args=a, kwargs=k)
    _gym.Env, _gym.Wrapper, _gym.spaces = _GymEnv, _Wrapper, _spaces
    sys.modules["gymnasium"], sys.modules["gymnasium.spaces"] = _gym, _spaces

from serl_launcher.aqi.operator_gate import SyntheticOperatorWrapper  # noqa: E402
from serl_launcher.aqi.operator_model import (PhaseOracle,  # noqa: E402
                                              SyntheticOperator,
                                              make_operator_theta)

SCALE = 0.01


class _Oracle(PhaseOracle):
    def lateral_error(self, obs):
        return np.array([0.0, 8.0])

    def height(self, obs):
        return 30.0

    def in_contact(self, obs):
        return False

    def jammed(self, obs):
        return False

    def commit_threshold(self):
        return 4.0

    def approach_height(self):
        return 25.0

    def action_dim(self):
        return 3


class _Env:
    def __init__(self):
        self.executed = []

    def reset(self, **kw):
        return np.zeros(3), {}

    def step(self, a):
        self.executed.append(np.array(a))
        return np.zeros(3), 0.0, False, False, {}


class _Monitor:
    last_channel = "corridor"

    def __init__(self):
        self.z, self._t = 0.0, 0

    def reset(self):
        self._t = 0

    def __call__(self, obs, info=None):
        self._t += 1
        return self.z


def _wrapper(n):
    th = make_operator_theta("expert", rng=np.random.default_rng(1))
    th.sigma_mon, th.rho_fa, th.p_miss, th.m_in, th.l_react = 0.0, 0.0, 0.0, 1, 1
    th.delta_in, th.delta_out, th.k_out, th.d_max = 0.5, 0.2, 2, 1000.0
    op = SyntheticOperator(_Oracle(), th, rng=np.random.default_rng(0), action_scale=SCALE)
    w = SyntheticOperatorWrapper(_Env(), op, _Monitor(), rng=np.random.default_rng(2), handover_steps=n)
    w.reset()
    return w


def _episode(w, robot, on=20, off=20):
    """Quiet, trouble for `on` steps, quiet again; returns (info, executed) per step."""
    out = []
    for t in range(5 + on + off):
        w.monitor.z = 0.9 if 5 <= t < 5 + on else 0.0
        info = w.step(robot)[-1]
        out.append((info, w.env.executed[-1].copy()))
    return out


def test_off_is_exactly_as_before():
    w = _wrapper(0)
    robot = np.array([0.3, -0.2, 0.1])
    out = _episode(w, robot)
    assert all("blended_action" not in i for i, _ in out)
    back = [k for k, (i, _) in enumerate(out) if i["synthetic_operator"]["handback"]][0]
    assert np.allclose(out[back][1], robot) and np.allclose(out[back + 1][1], robot)   # abrupt: policy at once


def test_motion_starts_from_the_robot_and_ramps_over():
    n = 5
    robot = np.array([0.6, -0.4, 0.2])
    w = _wrapper(n)
    out = _episode(w, robot)
    took = [k for k, (i, _) in enumerate(out) if i["synthetic_operator"]["takeover_started"]][0]
    first = out[took][1]
    assert np.linalg.norm(first - robot) < 0.5 * np.linalg.norm(robot)    # close to the robot, not to rest
    off = _episode(_wrapper(0), robot)
    took0 = [k for k, (i, _) in enumerate(off) if i["synthetic_operator"]["takeover_started"]][0]
    assert np.linalg.norm(first - robot) < np.linalg.norm(off[took0][1] - robot)   # a smaller jump than without
    assert "intervene_action" in out[took][0]                              # it is a takeover step
    assert np.allclose(out[took][0]["intervene_action"], first)            # stored as it ran


def test_hand_back_is_blended_and_reported():
    n = 4
    robot = np.array([0.5, 0.5, -0.5])
    w = _wrapper(n)
    out = _episode(w, robot)
    back = [k for k, (i, _) in enumerate(out) if i["synthetic_operator"]["handback"]][0]
    last_op = w._last_op_action.copy()
    weights = []
    for k in range(back, back + n):
        info, ran = out[k]
        assert "intervene_action" not in info                              # the policy's step again
        assert np.allclose(info["blended_action"], ran)                    # reported for the buffer
        d = last_op - robot
        j = int(np.argmax(np.abs(d)))
        weights.append((ran[j] - robot[j]) / d[j])
    want = [(n - i) / (n + 1.0) for i in range(n)]
    assert np.allclose(weights, want, atol=1e-5), (weights, want)          # linear ramp down
    info, ran = out[back + n]
    assert "blended_action" not in info and np.allclose(ran, robot)        # then the policy alone


def test_reset_clears_the_ramp_and_no_randomness_is_added():
    w = _wrapper(4)
    _episode(w, np.ones(3) * 0.2)
    w.reset()
    assert w._ramp_in == 0 and w._ramp_out == 0 and w._last_op_action is None
    a, b = _wrapper(0), _wrapper(6)
    robot = np.array([0.2, 0.2, 0.2])
    _episode(a, robot), _episode(b, robot)
    assert a.op.rng.bit_generator.state == b.op.rng.bit_generator.state    # same random stream as off
    assert a.gate.rng.bit_generator.state == b.gate.rng.bit_generator.state


def test_flag_is_defined_and_the_training_loop_stores_what_ran():
    src = (REPO / "examples" / "train_aqi_rlpd.py").read_text(encoding="utf-8")
    assert "synthetic_handover_steps" in set(re.findall(r'flags\.DEFINE_\w+\(\s*"(\w+)"', src))
    assert 'actions = info.pop("blended_action", exec_action)' in src


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"{len(tests)} tests passed")
