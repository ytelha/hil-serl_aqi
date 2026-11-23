"""Tests for the fixed-budget run instrumentation (serl_launcher/utils/run_guard.py; extension study, phase 3):
checkpoints on environment steps, the divergence guard, the local learner metrics, the careless counts and E2HiL's
careless label.

    python tests/test_run_guard.py        (numpy only; no GPU, no simulator)
"""
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "serl_launcher"))

from serl_launcher.utils import run_guard as rg      # noqa: E402


def test_flatten_metrics_nested_and_arrays():
    info = {"critic": {"critic_loss": np.float32(1.5), "q": np.array([1.0, 3.0])},
            "actor": {"actor_loss": 2.0, "e2hil_careless_keep_frac": np.array(0.25)},
            "temperature": np.array([0.1]), "name": "x", "empty": np.array([])}
    flat = rg.flatten_metrics(info)
    assert flat == {"critic/critic_loss": 1.5, "critic/q": 2.0, "actor/actor_loss": 2.0,
                    "actor/e2hil_careless_keep_frac": 0.25, "temperature": 0.1}
    assert rg.nonfinite_keys(flat) == []
    bad = rg.flatten_metrics({"a": np.array([1.0, np.nan]), "b": float("inf"), "c": 1.0})
    assert rg.nonfinite_keys(bad) == ["a", "b"]
    assert rg.json_safe(bad) == {"a": "nan", "b": "inf", "c": 1.0}


def test_divergence_guard_writes_marker_with_last_finite():
    with tempfile.TemporaryDirectory() as d:
        g = rg.DivergenceGuard(d, 100)
        assert g.on and g.due(0) and g.due(200) and not g.due(150)
        assert not g.check(100, {"critic_loss": 1.0, "temperature": 0.2}, "pretrain")
        assert not rg.diverged(d)
        # a non-finite diagnostic alone never trips the guard (only the core losses / temperature / Q / entropy)
        assert not g.check(100, {"critic": {"critic_loss": 1.0}, "actor": {"e2hil_abs_c_hi": float("inf")},
                                 "temperature": 0.2}, "pretrain")
        assert not rg.diverged(d)
        assert g.check(200, {"critic_loss": float("nan"), "temperature": 0.2}, "online",
                       {"online_updates": 50, "env_received": 60})
        assert rg.diverged(d)
        p = json.loads(Path(d, rg.DIVERGED_FILE).read_text())
        assert p["step"] == 200 and p["phase"] == "online" and p["nonfinite"] == ["critic_loss"]
        assert p["nonfinite_all"] == ["critic_loss"]
        # the last finite step is the one with the (non-core) inf diagnostic: recorded as it was
        assert p["last_finite_step"] == 100 and p["last_finite_metrics"]["critic/critic_loss"] == 1.0
        assert p["last_finite_metrics"]["actor/e2hil_abs_c_hi"] == "inf"
        assert p["metrics"]["critic_loss"] == "nan" and p["online_updates"] == 50
        g2 = rg.DivergenceGuard(d, 100)
        g2.reset()                                  # a new learner start drops the old marker
        assert not rg.diverged(d)
    off = rg.DivergenceGuard(None, 100)
    assert not off.on and not off.due(100)
    assert not rg.DivergenceGuard("x", 0).due(0)


def test_core_metrics_by_name_in_any_group():
    flat = rg.flatten_metrics({"critic": {"critic_loss": float("nan"), "critic_lr": float("nan")},
                               "actor": {"actor_loss": 1.0, "aqi_bc_loss": float("inf")},
                               "temperature": {"temperature_loss": float("nan")}})
    assert rg.core_nonfinite(flat) == ["actor/aqi_bc_loss", "critic/critic_loss", "temperature/temperature_loss"]


def test_metrics_log_lines():
    with tempfile.TemporaryDirectory() as d:
        m = rg.MetricsLog(d, 50)
        assert m.due(0) and m.due(100) and not m.due(75)
        m.write(50, {"actor": {"e2hil_careless_keep_frac": 0.8}, "x": float("nan")}, "online", {"online_updates": 3})
        rows = [json.loads(x) for x in Path(d, rg.METRICS_FILE).read_text().splitlines()]
        assert rows == [{"step": 50, "phase": "online", "t": rows[0]["t"], "online_updates": 3,
                         "metrics": {"actor/e2hil_careless_keep_frac": 0.8, "x": "nan"}}]


def test_env_checkpoints_schedule_and_final():
    with tempfile.TemporaryDirectory() as d:
        e = rg.EnvCheckpoints(d, 2000, 1.0)
        saved = []
        for n in range(1, 19700):                    # online updates under U = 1; the actor stopped at 19 699
            if e.due(n):
                saved.append(e.record(5000 + n - 1, n, n + 150)["env_steps"])
        assert saved == list(range(2000, 18001, 2000))
        # at the stop the learner has done 19 699 updates but received 19 850 steps: it catches up first
        assert e.catch_up(19699, 19850, 0)
        assert not e.catch_up(19850, 19850, 151)
        assert not e.catch_up(19700, 19850, 2000)    # at most `limit` extra updates
        assert not e.catch_up(19700, float("inf"), 0)
        row = e.record(5000 + 19849, 19850, 19850, final=True)
        assert row["env_steps"] == 19850 and row["final"]
        rows = [json.loads(x) for x in Path(d, rg.ENV_CKPT_FILE).read_text().splitlines()]
        assert [r["env_steps"] for r in rows] == list(range(2000, 18001, 2000)) + [19850]
        assert rows[0]["learner_step"] == 6999 and rows[0]["online_updates"] == 2000 and rows[0]["env_received"] == 2150
    e2 = rg.EnvCheckpoints("x", 1000, 2.0)            # U = 2: the k-th after 2 * k * 1000 updates
    assert not e2.due(1999) and e2.due(2000)
    try:
        rg.EnvCheckpoints("x", 2000, 0.0)
        raise AssertionError("needs a fixed update ratio")
    except ValueError:
        pass
    assert not rg.EnvCheckpoints("x", 0, 0.0).on      # off: no ratio needed


def test_careless_counter_and_label():
    c = rg.CarelessCounter()
    # takeover 1 careful (3 steps), takeover 2 careless (2 steps), robot steps in between
    seq = [(True, True, "attentive"), (True, False, "attentive"), (True, False, "attentive"), (False, False, None),
           (True, True, "inattentive"), (True, False, "inattentive"), (False, False, "inattentive")]
    for iv, onset, cond in seq:
        c.step(iv, onset, cond)
    assert c.record() == {"careless_takeovers": 1, "careless_steps": 2}
    c.reset()
    assert c.record() == {"careless_takeovers": 0, "careless_steps": 0}
    assert rg.e2hil_label(False, "inattentive") == 1
    assert rg.e2hil_label(True, "attentive") == 2
    assert rg.e2hil_label(True, None) == 2             # consistent operator: every takeover is label 2
    assert rg.e2hil_label(True, "inattentive") == 3


def test_would_stop_payload():
    p = rg.would_stop_payload(6100, 31, 0.8, 0.6, 2, 2000, [[4000, 0.6], [6100, 0.8]], [[0.0, 0.0]])
    assert p["step"] == 6100 and p["mode"] == "observe" and p["mini_eval_history"][-1] == [6100, 0.8]
    with tempfile.TemporaryDirectory() as d:
        rg.write_json(os.path.join(d, rg.WOULD_STOP_FILE), p)
        assert json.loads(Path(d, rg.WOULD_STOP_FILE).read_text())["step"] == 6100
        assert not Path(d, rg.WOULD_STOP_FILE + ".tmp").exists()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
