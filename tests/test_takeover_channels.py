"""Tests for the takeover-channel logging (--log_takeover_channels,
operator_gate.py): the monitor names the channel behind z, off leaves the
wrapper's info exactly as before, on adds takeover_channel / handback_kind
and nothing else, and the operator acts the same either way.

Without gymnasium (this PC) a minimal stand-in for gym.Env / gym.Wrapper is
installed, so the toy insertion env still runs; with gymnasium the real one
is used.

Run from the repo root with
    python -m pytest tests/test_takeover_channels.py -q
    python tests/test_takeover_channels.py
"""
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
if not _real_gym:            # minimal stand-in: only what the wrapper and toy env use
    _gym = types.ModuleType("gymnasium")
    _spaces = types.ModuleType("gymnasium.spaces")

    class _Env:
        @property
        def unwrapped(self):
            return self

    class _Wrapper(_Env):
        def __init__(self, env):
            self.env = env

        @property
        def unwrapped(self):
            return self.env.unwrapped

        def reset(self, **kw):
            return self.env.reset(**kw)

        def step(self, action):
            return self.env.step(action)

    _spaces.Box = lambda *a, **k: SimpleNamespace(args=a, kwargs=k)
    _spaces.Dict = lambda *a, **k: SimpleNamespace(args=a, kwargs=k)
    _gym.Env, _gym.Wrapper, _gym.spaces = _Env, _Wrapper, _spaces
    sys.modules["gymnasium"], sys.modules["gymnasium.spaces"] = _gym, _spaces

from serl_launcher.aqi.operator_gate import (MonitoringSignal,  # noqa: E402
                                             SyntheticOperatorWrapper)
from serl_launcher.aqi.operator_model import (SyntheticOperator,  # noqa: E402
                                              make_operator_theta)
from serl_launcher.aqi.toy_insertion import (ToyInsertionEnv,  # noqa: E402
                                             ToyInsertionOracle)

CHANNELS = {"time", "corridor", "stagnation", "hover", "off_bore", "orientation", "force",
            "force_excess", "value", "sigma", "false_alarm", "request"}
HANDBACKS = {"signal", "time_cap", "latch_success", "episode_end"}
NEW_KEYS = {"takeover_channel", "handback_kind"}


def _run(log, tier="novice", seed=0, steps=600, **kw):
    orc = ToyInsertionOracle()
    op = SyntheticOperator(orc, make_operator_theta(tier),
                           rng=np.random.default_rng(seed))
    env = SyntheticOperatorWrapper(
        ToyInsertionEnv(seed=seed), op, MonitoringSignal(orc),
        rng=np.random.default_rng(seed + 1), log_takeover_channels=log, **kw)
    env.reset()
    rng = np.random.default_rng(seed + 2)
    out = []
    for _ in range(steps):
        # a poor policy that needs help: it descends while drifting sideways
        a = np.array([rng.uniform(-40.0, 40.0), -25.0])
        obs, rew, term, trunc, info = env.step(a)
        out.append((obs["state"].copy(), rew, term, trunc,
                    dict(info["synthetic_operator"])))
        if term or trunc:
            env.reset()
    return out


def test_monitor_names_the_largest_channel():
    class Orc:
        def __init__(self, h, err):
            self.h, self.err = h, err

        def height(self, obs):
            return self.h

        def lateral_error(self, obs):
            return np.array([self.err, 0.0])

        def approach_height(self):
            return 20.0
    m = MonitoringSignal(Orc(h=10.0, err=1.5), corridor_mm=1.0)
    m.reset()
    assert m.last_channel == "time"
    assert m(None) == 1.0 and m.last_channel == "corridor"
    m = MonitoringSignal(Orc(h=-1.0, err=5.0), corridor_mm=1.0)
    m.reset()
    assert m(None) == 1.0 and m.last_channel == "off_bore"
    m = MonitoringSignal(Orc(h=50.0, err=0.0), force_norm=10.0)
    m.reset()
    assert abs(m(None, {"f_z": 8.0}) - 0.8) < 1e-12
    assert m.last_channel == "force"
    m = MonitoringSignal(None, budget_steps=10)
    m.reset()
    for _ in range(4):
        z = m(None)
    assert abs(z - 0.2) < 1e-12 and m.last_channel == "time"


def test_off_info_unchanged_and_on_adds_only_the_new_keys():
    off, on = _run(False), _run(True)
    assert len(off) == len(on)
    n_trig = 0
    for (s0, r0, te0, tr0, i0), (s1, r1, te1, tr1, i1) in zip(off, on):
        assert np.array_equal(s0, s1) and r0 == r1 and te0 == te1 and tr0 == tr1
        assert not NEW_KEYS & set(i0)
        assert set(i1) - set(i0) <= NEW_KEYS
        assert {k: v for k, v in i1.items() if k not in NEW_KEYS} == i0
        n_trig += int(i0["takeover_started"])
    assert n_trig > 0                       # the check saw takeovers


def test_every_takeover_has_a_channel_and_every_handback_a_kind():
    seen_ch, seen_hb = set(), set()
    for tier in ("expert", "intermediate", "novice"):
        for (_s, _r, term, trunc, i) in _run(True, tier=tier):
            if i["takeover_started"]:
                assert i["takeover_channel"] in CHANNELS
                seen_ch.add(i["takeover_channel"])
                assert (i["takeover_channel"] == "false_alarm") == i["false_alarm"]
            else:
                assert "takeover_channel" not in i
            if i["handback"]:                       # the gate handed back
                assert i["handback_kind"] in ("signal", "time_cap")
            if "handback_kind" in i:
                assert i["handback_kind"] in HANDBACKS
                seen_hb.add(i["handback_kind"])
    assert seen_ch and seen_hb


def test_takeover_cause_is_the_most_common_channel_in_the_window():
    stub = SimpleNamespace(
        respond_to_request=False,
        op=SimpleNamespace(th=SimpleNamespace(m_in=2, l_react=1)),
        _chan_hist=["force"] * 10 + ["time", "corridor", "corridor", "time",
                                     "corridor"])
    cause = SyntheticOperatorWrapper._takeover_cause
    assert cause(stub, "trigger") == "corridor"     # last 4: c, c, t, c
    assert cause(stub, "trigger_fa") == "false_alarm"
    stub._chan_hist = []
    assert cause(stub, "trigger") == ""
    stub.respond_to_request = True
    assert cause(stub, "trigger") == "request"


class _SucceedKey(ToyInsertionEnv):
    """The toy env with the sim env's success key ('succeed'), which the
    latching modes listen to."""

    def _info(self, jam_event=False):
        info = super()._info(jam_event)
        info["succeed"] = info["succeeded"]
        return info


def test_latch_released_by_success_is_logged():
    orc = ToyInsertionOracle()
    op = SyntheticOperator(orc, make_operator_theta("expert"),
                           rng=np.random.default_rng(0))
    env = SyntheticOperatorWrapper(
        _SucceedKey(seed=0), op, MonitoringSignal(orc),
        rng=np.random.default_rng(1), intervention_mode="always_latch",
        log_takeover_channels=True)
    env.reset()
    kinds = []
    for _ in range(1500):
        _o, _r, term, trunc, i = env.step(np.array([30.0, -25.0]))
        info = i["synthetic_operator"]
        if "handback_kind" in info:
            kinds.append(info["handback_kind"])
        if term or trunc:
            env.reset()
    assert "latch_success" in kinds
    assert "signal" not in kinds and "time_cap" not in kinds   # latched


def test_one_channel_and_one_handback_per_takeover():
    """Per episode: takeovers (the actor's intervention_count, the rising
    edges of intervene_action) = channels logged = hand-backs logged, in
    every intervention mode."""
    for mode in ("signal", "always_latch", "latch_until_autonomous_success"):
        for tier in ("expert", "novice"):
            orc = ToyInsertionOracle()
            op = SyntheticOperator(orc, make_operator_theta(tier),
                                   rng=np.random.default_rng(3))
            env = SyntheticOperatorWrapper(
                _SucceedKey(seed=3), op, MonitoringSignal(orc),
                rng=np.random.default_rng(4), intervention_mode=mode,
                log_takeover_channels=True)
            env.reset()
            rng = np.random.default_rng(5)
            n = {"takeovers": 0, "channels": 0, "handbacks": 0}
            prev, total = False, 0
            for _ in range(1200):
                a = np.array([rng.uniform(-40.0, 40.0), rng.uniform(-35.0, 5.0)])
                _o, _r, term, trunc, info = env.step(a)
                used = "intervene_action" in info
                n["takeovers"] += int(used and not prev)
                prev = used
                n["channels"] += int("takeover_channel" in info["synthetic_operator"])
                n["handbacks"] += int("handback_kind" in info["synthetic_operator"])
                if term or trunc:
                    assert n["takeovers"] == n["channels"] == n["handbacks"], (mode, tier, n)
                    total += n["takeovers"]
                    n = dict.fromkeys(n, 0)
                    prev = False
                    env.reset()
            assert total > 0, (mode, tier)


def test_episode_end_during_a_takeover_is_logged():
    found = False
    for seed in range(4):
        for (_s, _r, term, trunc, i) in _run(True, tier="novice", seed=seed,
                                             steps=900):
            if i.get("handback_kind") == "episode_end":
                assert term or trunc
                found = True
    assert found


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
