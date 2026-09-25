"""Tests for examples/quality_signal_audit.py (numpy only; runs anywhere).

The key test builds synthetic logs with KNOWN structure: episode success is
decided by WHEN the takeover happens (late = timeout failure) and by a real
quality signal (d_value); late takeovers run to the operator's d_max cap.
The audit must then show -t_int as predictive on all segments but not once
timing is controlled, while d_value stays predictive.

Run from the repo root with
    python -m pytest tests/test_quality_audit.py -q
    python tests/test_quality_audit.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "examples"))

import quality_signal_audit as qa             # noqa: E402

T = 300          # episode budget (steps)


def _write_run(root, name, arm, tier, d_max, segments, episodes,
               score_mode="behavioral"):
    d = Path(root) / name
    d.mkdir(parents=True)
    op = f"{tier[:3]}00"
    meta = {"run_name": name, "operator_id": op, "mode": "quality",
            "config": {"arm": arm, "quality_score_mode": score_mode,
                       "synthetic_operator": {"tier": tier, "d_max": d_max,
                                              "operator_id": op}}}
    (d / "meta.json").write_text(json.dumps(meta))
    with open(d / "segments.jsonl", "w") as f:
        for s in segments:
            f.write(json.dumps(s) + "\n")
    with open(d / "episodes.jsonl", "w") as f:
        for e in episodes:
            f.write(json.dumps(e) + "\n")


def _leaky_world(root, n_ep=400, seed=0):
    """One run per tier. success = early takeover AND good correction."""
    rng = np.random.default_rng(seed)
    for tier, d_max in (("expert", 3.8), ("intermediate", 5.5),
                        ("novice", 8.0)):
        cap = int(np.ceil(d_max * 10))
        segs, eps = [], []
        for ep in range(n_ep):
            start = int(rng.integers(0, 260))
            quality = rng.normal()
            late = start > 140
            if late:           # operator cannot hand back: runs to the cap
                length = min(cap, T - start)
            else:
                length = int(rng.integers(3, cap - 4))
            end = start + length
            success = (not late) and quality > -0.8
            ep_len = min(T, end + int(rng.integers(1, 40))) if success else T
            segs.append({
                "episode": ep, "start_step": start, "length": length,
                "episode_success": success, "requested": False,
                "excluded": False, "q_int_mean": max(quality, 0.0),
                "a_hat_mean": rng.normal(), "a_hat_at_trigger": rng.normal(),
                "sigma_mean": abs(rng.normal()),
                "features": {"d_value": quality + 0.3 * rng.normal(),
                             "d_progress": rng.normal(),
                             "f_peak": abs(rng.normal()),
                             "jerk_mean": abs(rng.normal()),
                             "t_int": length}})
            eps.append({"episode": ep, "length": ep_len, "success": success})
        _write_run(root, f"quality-beh_{tier[:3]}00_s0_actor", "quality-beh",
                   tier, d_max, segs, eps)


def test_auroc_primitives():
    assert qa.auroc([1, 2, 3], [False, True, True]) == 1.0
    assert qa.auroc([3, 2, 1], [False, True, True]) == 0.0
    assert qa.auroc([1, 1, 1, 1], [0, 1, 0, 1]) == 0.5          # all ties
    assert qa.auroc([1, 2], [True, True]) != qa.auroc([1, 2], [1, 1])  # nan
    s = np.random.default_rng(0).normal(size=50)
    y = np.random.default_rng(1).random(50) < 0.5
    assert abs(qa.stratified_auroc(s, y, np.zeros(50)) - qa.auroc(s, y)) < 1e-12
    # a score that only encodes the stratum is uninformative within strata
    strata = np.repeat([0, 1], 25)
    lab = np.r_[np.ones(20, bool), np.zeros(5, bool),
                np.ones(5, bool), np.zeros(20, bool)]
    assert qa.auroc(-strata, lab) > 0.7
    assert qa.stratified_auroc(-strata, lab, strata) == 0.5


def test_termination_classification():
    with tempfile.TemporaryDirectory() as tmp:
        cap = 39                                  # ceil(3.834 * 10)
        segs = [
            {"episode": 0, "start_step": 10, "length": cap,
             "episode_success": False, "features": {}},        # at cap
            {"episode": 0, "start_step": 100, "length": cap - 3,
             "episode_success": False, "features": {}},        # tolerance
            {"episode": 1, "start_step": 10, "length": 12,
             "episode_success": True, "features": {}},         # handback
            {"episode": 1, "start_step": 60, "length": 20,
             "episode_success": True, "features": {}},         # ep end
        ]
        eps = [{"episode": 0, "length": 300}, {"episode": 1, "length": 80}]
        _write_run(tmp, "r", "quality", "expert", 3.834, segs, eps)
        rows = qa.load_segments(tmp, cap_tolerance=2, horizon=50)
        assert [r["cap_steps"] for r in rows] == [cap] * 4
        assert [r["ended"] for r in rows] == ["cap", "handback", "handback",
                                              "episode_end"]
        # released at step 22, success at 80: 58 > horizon 50 -> False
        assert rows[2]["post_release_success"] is False
        assert rows[3]["post_release_success"] is None


def test_detects_timing_leakage_and_keeps_real_signal():
    with tempfile.TemporaryDirectory() as tmp:
        _leaky_world(tmp)
        out = qa.make_audit(log_root=tmp, out=os.path.join(tmp, "audit"),
                            n_boot=100, plots=False)
        res = json.load(open(os.path.join(out, "audit.json")))
        g = res["arms"]["quality-beh"]
        t = g["auroc"]["-t_int"]
        dv = g["auroc"]["d_value"]
        time_only = g["auroc"]["-start_step (time only)"]["auroc"]
        # -t_int looks predictive overall ...
        assert t["auroc"] > 0.7, t
        # ... but only through timing: clean and stratified collapse to ~0.5
        assert 0.35 < t["auroc_clean"] < 0.65, t
        assert t["auroc_stratified"] < t["auroc"] - 0.1, t
        assert time_only >= t["auroc"] - 0.03
        # the genuine quality signal survives every control
        assert dv["auroc_clean"] > 0.7 and dv["auroc_stratified"] > 0.65, dv
        ci = t["ci95"]
        assert ci[0] < t["auroc"] < ci[1]
        # takeover length tracks the d_max cap across tiers
        term = g["termination"]
        assert term["novice"]["mean_cap"] > term["expert"]["mean_cap"]
        assert term["all"]["frac_cap"] > 0.2
        text = open(os.path.join(out, "summary.md")).read()
        assert "Timing alone explains" in text
        assert "## Arm `quality-beh`" in text
        assert os.path.exists(os.path.join(out, "segments.csv"))


def test_no_segments_returns_none():
    with tempfile.TemporaryDirectory() as tmp:
        assert qa.make_audit(log_root=tmp, out=os.path.join(tmp, "o"),
                             plots=False) is None


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
