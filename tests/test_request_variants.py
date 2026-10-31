"""Tests for the two request variants of the synthetic operator
(operator_gate.py): the OR arm (--synthetic_request_or: requests are added to
the operator's own monitor) and held requests (--synthetic_request_hold_steps:
a request waits for a slow operator). Off, the wrapper behaves exactly as
before; on, the campaign runner passes them per arm, only to the modes that
have requests.

Without gymnasium (this PC) a minimal stand-in for gym.Wrapper is installed;
with gymnasium the real one is used.

Run from the repo root with
    python -m pytest tests/test_request_variants.py -q
    python tests/test_request_variants.py
"""
import re
import sys
import tempfile
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

from serl_launcher.aqi.operator_gate import (MonitoringSignal,  # noqa: E402
                                             SyntheticOperatorWrapper)
from serl_launcher.aqi.operator_model import (SyntheticOperator,  # noqa: E402
                                              make_operator_theta)
from serl_launcher.aqi.operator_model import PhaseOracle  # noqa: E402


class _Oracle(PhaseOracle):
    """A peg hovering far above the hole: no channel of the monitor fires
    unless the test sets `z` through a stand-in monitor."""

    def lateral_error(self, obs):
        return np.zeros(2)

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
    """z_task under the test's control."""
    last_channel = "corridor"

    def __init__(self):
        self.z, self._t = 0.0, 0

    def reset(self):
        self._t = 0

    def __call__(self, obs, info=None):
        self._t += 1
        return self.z


def _wrapper(l_react=15, p_miss=0.0, **kw):
    th = make_operator_theta("expert", rng=np.random.default_rng(1))
    th.sigma_mon, th.rho_fa, th.p_miss, th.m_in = 0.0, 0.0, p_miss, 1
    th.l_react, th.delta_in, th.delta_out, th.k_out, th.d_max = l_react, 0.5, 0.2, 2, 1000.0
    op = SyntheticOperator(_Oracle(), th, rng=np.random.default_rng(0), action_scale=0.01)
    w = SyntheticOperatorWrapper(_Env(), op, _Monitor(), rng=np.random.default_rng(2), **kw)
    w.reset()
    return w


def _run(w, requests):
    """Feed one request flag per step; returns the steps with an operator in control."""
    over = []
    a = np.zeros(3)
    for r in requests:
        w.notify_request(bool(r))
        w.step(a)
        over.append(w.gate.taking_over)
    return over


def test_off_keeps_the_old_behaviour():
    a, b = _wrapper(respond_to_request=True), _wrapper(respond_to_request=True, request_or_monitor=False,
                                                       request_hold_steps=0)
    a.monitor.z = b.monitor.z = 0.9                        # the monitor is ignored when requests replace it
    assert _run(a, [0] * 40) == _run(b, [0] * 40) == [False] * 40
    pulse = [1] * 10 + [0] * 30
    c = _wrapper(l_react=15, respond_to_request=True)
    assert not any(_run(c, pulse))                         # a short pulse lapses before a slow reaction
    d = _wrapper(l_react=5, respond_to_request=True)
    assert any(_run(d, pulse))                             # a quick operator answers it


def test_or_arm_steps_in_unasked_and_still_answers():
    w = _wrapper(l_react=3, respond_to_request=True, request_or_monitor=True, log_takeover_channels=True)
    w.monitor.z = 0.9
    assert any(_run(w, [0] * 10))                          # unasked, on its own monitor
    w = _wrapper(l_react=3, respond_to_request=True, request_or_monitor=True)
    w.monitor.z = 0.0
    assert any(_run(w, [1] * 10))                          # asked, with a quiet monitor


def test_or_arm_names_the_cause():
    w = _wrapper(l_react=3, respond_to_request=True, request_or_monitor=True, log_takeover_channels=True)
    w.monitor.z = 0.9
    causes = []
    for _ in range(10):
        w.notify_request(False)
        _o, _r, _d, _t, info = w.step(np.zeros(3))
        if "takeover_channel" in info["synthetic_operator"]:
            causes.append(info["synthetic_operator"]["takeover_channel"])
    assert causes == ["corridor"]
    w = _wrapper(l_react=3, respond_to_request=True, request_or_monitor=True, log_takeover_channels=True)
    causes = []
    for _ in range(10):
        w.notify_request(True)
        _o, _r, _d, _t, info = w.step(np.zeros(3))
        if "takeover_channel" in info["synthetic_operator"]:
            causes.append(info["synthetic_operator"]["takeover_channel"])
    assert causes == ["request"]


def test_a_held_request_reaches_a_slow_operator():
    pulse = [1] * 2 + [0] * 40
    assert not any(_run(_wrapper(l_react=15, respond_to_request=True), pulse))
    held = _wrapper(l_react=15, respond_to_request=True, request_hold_steps=30)
    over = _run(held, pulse)
    assert any(over)
    assert over.index(True) >= 15                          # reaction delay still applies
    assert held._held_left == 0                            # cleared by the takeover


def test_a_held_request_times_out_and_a_miss_stays_a_miss():
    w = _wrapper(l_react=3, p_miss=1.0, respond_to_request=True, request_hold_steps=8)
    stim = []
    a = np.zeros(3)
    for r in [1] + [0] * 20:
        w.notify_request(bool(r))
        w.step(a)
        stim.append(w._ext_request)
    assert not w.gate.taking_over                          # p_miss = 1: never answered
    assert stim[:8] == [1.0] * 8 and stim[8:] == [0.0] * 13    # on for exactly the hold, then off
    w = _wrapper(l_react=50, respond_to_request=True, request_hold_steps=5)
    a = np.zeros(3)
    on = []
    for _ in range(12):                                    # a standing request cannot extend the hold ...
        w.notify_request(True)
        on.append(w._held_left)
        w.step(a)
    assert on[0] == 5 and on[4] == 1 and on[5] == 5        # ... it re-arms only after the hold has ended


def test_no_randomness_is_added():
    a = _wrapper(respond_to_request=True)
    b = _wrapper(respond_to_request=True, request_hold_steps=7, request_or_monitor=True)
    s0a, s0b = a.gate.rng.bit_generator.state, b.gate.rng.bit_generator.state
    assert s0a == s0b
    a.gate.rng.normal(), b.gate.rng.normal()
    w = _wrapper(respond_to_request=True, request_hold_steps=7)
    n = w.gate.rng.bit_generator.state["state"]["state"]
    w.notify_request(True)
    assert w.gate.rng.bit_generator.state["state"]["state"] == n   # notify draws nothing


def test_campaign_runner_passes_them_per_arm():
    import run_campaign as rc
    d = Path(tempfile.mkdtemp())
    base = (REPO / "examples" / "campaigns" / "active_ratematch.yaml").read_text(encoding="utf-8")
    text = base.replace("  logged:      {mode: logged}\n",
                        "  logged:      {mode: logged}\n"
                        "  active-or:   {mode: active, query_target_rate: 0.55, query_cooldown_steps: 5, synthetic_request_or: true}\n"
                        "  active-hold: {mode: active, query_target_rate: 0.55, query_cooldown_steps: 5, synthetic_request_hold_steps: 30}\n"
                        ).replace("  - logged                         # engineered geometric monitor (~60%)\n",
                                  "  - logged\n  - active-or\n  - active-hold\n")
    (d / "c.yaml").write_text(text, encoding="utf-8")
    argv = ["--campaign", "unused", "--config", str(d / "c.yaml")]
    parser = rc.build_arg_parser()
    args = parser.parse_args(argv)
    args.ops_per_tier_map = rc.apply_config(args, rc.load_config(str(d / "c.yaml")),
                                            rc._explicit_dests(parser, argv)) or None
    ckpt = Path("runs/x/checkpoints/c")
    cmd = lambda arm: rc._actor_cmd(args, ckpt, arm, "expert", "exp00", Path("t.json"), 0, "r")   # noqa: E731
    assert "--synthetic_request_or" in cmd("active-or")
    assert cmd("active-hold")[cmd("active-hold").index("--synthetic_request_hold_steps") + 1] == "30"
    for arm in ("active-q55", "logged", "hilserl"):
        assert not [t for t in cmd(arm) if "request_or" in t or "request_hold" in t], arm
    src = (REPO / "examples" / "train_aqi_rlpd.py").read_text(encoding="utf-8")
    defined = set(re.findall(r'flags\.DEFINE_\w+\(\s*"(\w+)"', src))
    assert {"synthetic_request_or", "synthetic_request_hold_steps"} <= defined


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"{len(tests)} tests passed")
