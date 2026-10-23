"""Tests for the campaign report (examples/run_campaign.py, campaign_plots.py):
the headline states the ICC change with the right verb and an unsigned
magnitude, the header counts operators per tier from the roster, and the
frontier figure places every method at its measured intervention rate. Run
from the repo root with
    python -m pytest tests/test_report_render.py -q
    python tests/test_report_render.py
"""
import sys
from collections import namedtuple
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for _p in ("serl_launcher", "examples"):
    sys.path.insert(0, str(REPO / _p))

import run_campaign as rc                     # noqa: E402
import campaign_plots as cp                   # noqa: E402

TIERS = ["expert", "intermediate", "novice"]
Op = namedtuple("Op", "operator_id tier")


def _sr(per_op):
    """{op: [sr seed0, sr seed1]} -> sr_by_method entry (two operators per tier)."""
    tier = {"e0": "expert", "e1": "expert", "i0": "intermediate", "i1": "intermediate",
            "n0": "novice", "n1": "novice"}
    return {op: {"tier": tier[op], "sr": {str(s): v for s, v in enumerate(vals)}}
            for op, vals in per_op.items()}


# baseline with a clear operator effect; "flat" removes it, "steep" makes it
# stronger, "same" repeats the baseline exactly
SR = {
    "logged": _sr({"e0": [0.9, 0.8], "e1": [0.8, 0.9], "i0": [0.6, 0.5],
                   "i1": [0.5, 0.6], "n0": [0.3, 0.2], "n1": [0.2, 0.3]}),
    "flat": _sr({"e0": [0.7, 0.4], "e1": [0.4, 0.7], "i0": [0.7, 0.4],
                 "i1": [0.4, 0.7], "n0": [0.7, 0.4], "n1": [0.4, 0.7]}),
    "steep": _sr({"e0": [1.0, 1.0], "e1": [1.0, 1.0], "i0": [0.5, 0.5],
                  "i1": [0.5, 0.5], "n0": [0.0, 0.0], "n1": [0.0, 0.0]}),
}
SR["same"] = SR["logged"]
METHODS = ["logged", "flat", "steep", "same"]


def _meta(ops_per_tier):
    return {"campaign_dir": "/runs/x", "exp_name": "sim_peg_insertion", "peg_shape": "cuboid",
            "grid_sha256": "0" * 64, "n_grid_trials": 9, "n_operators": 9,
            "ops_per_tier": ops_per_tier, "n_seeds": 2, "steps": 35000}


def _headline(md):
    return md.split("## Headline", 1)[1].strip().splitlines()


def test_headline_verbs_and_unsigned_magnitudes():
    report = rc.compute_dependency(SR, METHODS, TIERS, "logged")
    d = {m: report["methods"][m]["delta_icc_vs_baseline"] for m in METHODS}
    assert d["flat"] > 0 and d["steep"] < 0 and d["same"] == 0
    lines = _headline(rc.render_report_md(report, _meta(2), TIERS))
    assert lines[0] == (f"- `flat` reduces operator-dependency ICC by {d['flat']:.3f} vs `logged` "
                        f"(ICC {rc._fmt(report['methods']['logged']['icc'])} -> "
                        f"{rc._fmt(report['methods']['flat']['icc'])}).")
    assert lines[1].startswith(f"- `steep` increases operator-dependency ICC by {-d['steep']:.3f} ")
    assert lines[2].startswith("- `same` leaves operator-dependency ICC unchanged vs `logged`")
    assert not any(" by -" in l or " by +" in l for l in lines)


def test_change_below_rounding_counts_as_unchanged():
    report = rc.compute_dependency(SR, ["logged", "flat"], TIERS, "logged")
    report["methods"]["flat"]["delta_icc_vs_baseline"] = -0.0004
    assert "leaves operator-dependency ICC unchanged" in _headline(
        rc.render_report_md(report, _meta(2), TIERS))[0]


def test_header_uniform_and_per_tier_counts():
    report = rc.compute_dependency(SR, METHODS, TIERS, "logged")
    md = rc.render_report_md(report, _meta(3), TIERS)
    assert "- operators: 9 (3/tier x 3 tiers), seeds/op: 2, cap: 35000 env steps/cell" in md
    md = rc.render_report_md(report, _meta({"expert": 3, "intermediate": 3, "novice": 2}), TIERS)
    assert "- operators: 9 (expert 3, intermediate 3, novice 2), seeds/op: 2" in md


def test_roster_tier_counts_ignore_the_cli_default():
    roster = [Op(f"{t[:3]}{i:02d}", t) for t in TIERS for i in range(3)]
    assert rc._roster_tier_counts(roster, TIERS) == (3, {"expert": 3, "intermediate": 3, "novice": 3})
    ops, per = rc._roster_tier_counts(roster[:-1], TIERS)
    assert ops == per == {"expert": 3, "intermediate": 3, "novice": 2}


def _ivn_report():
    """Two methods with matched cells; hilserl has no aqi logs but a measured rate."""
    sr = {"logged": _sr({"e0": [0.8, 0.6]}), "hilserl": _sr({"e0": [1.0, 0.8]})}
    pairs = [{"operator": "e0", "tier": "expert", "seed": s, "baseline_rate": b, "method_rate": m,
              "delta": m - b} for s, (b, m) in enumerate([(0.50, 0.60), (0.54, 0.40)])]
    return {"baseline": "logged",
            "methods": {"logged": {"mean_sr": 0.7}, "hilserl": {"mean_sr": 0.9}},
            "sr_by_method": sr,
            "interventions": {"baseline": "logged",
                              "methods": {"logged": {"n_cells": 2, "rate_mean": 0.52},
                                          "hilserl": {"n_cells": 2, "rate_mean": 0.50}},
                              "paired": {"hilserl": {"pairs": pairs}}}}


def test_frontier_uses_the_measured_rate_of_every_method():
    pts = cp._frontier_points(_ivn_report())
    assert set(pts) == {"logged", "hilserl"}
    rates, srs, rate_mean, sr_mean = pts["hilserl"]
    assert rate_mean == 0.50 and sorted(rates) == [0.40, 0.60]
    assert abs(sr_mean - 0.9) < 1e-9 and sorted(srs) == [0.8, 1.0]
    assert sorted(pts["logged"][0]) == [0.50, 0.54]


def test_frontier_skipped_without_interventions():
    rep = _ivn_report()
    del rep["interventions"]
    assert cp._frontier_points(rep) == {}
    assert cp._fig_frontier(None, rep, None) is False


def test_frontier_figure_renders():
    try:
        import matplotlib
    except ImportError:                      # plots are optional, like in analyze
        return
    import tempfile
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "frontier.png"
        assert cp._fig_frontier(plt, _ivn_report(), out) is not False
        assert out.stat().st_size > 0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
