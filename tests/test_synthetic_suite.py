"""Tests for the synthetic-operator test suite (examples/campaigns/synthetic_suite/,
examples/synthetic_suite.py): the ladder and leave-one-out structure of the
configs, that every config passes the campaign runner, that the confirmation
keeps the thesis base configuration, and the driver's limits, run guards and
report on fake campaign folders. No JAX, no MuJoCo, no training.

Run from the repo root with
    python -m pytest tests/test_synthetic_suite.py -q
    python tests/test_synthetic_suite.py
"""
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for _p in ("serl_launcher", "examples"):
    sys.path.insert(0, str(REPO / _p))

import synthetic_suite as ss                  # noqa: E402

SW = ("rotation", "box", "channel", "convergence", "utd_ratio")
LADDER = ["L0_original", "L1_rotation", "L2_box", "L3_channel", "L4_evalconv", "L5_full"]
LOO = {"X_nobox": "box", "X_nochannel": "channel", "X_noeval": "convergence"}


def _sw(name):
    return ss.switches(ss.load_yaml(ss.config_path(name)))


def _diff(a, b):
    return [k for k in SW if a[k] != b[k]]


def _strip(cfg):
    """A config without the tested switches and the per-setting tag."""
    cfg = json.loads(json.dumps(cfg))
    cfg.get("task", {}).pop("rpy_box", None)
    for k in ("synthetic_rotation_control", "synthetic_orientation_monitor", "synthetic_orientation_tilt_rad",
              "synthetic_orientation_yaw_rad"):
        cfg.get("aqi", {}).pop(k, None)
    for k in ("converge_mode", "converge_eval_every", "converge_eval_consecutive", "utd_ratio"):
        cfg.get("budget", {}).pop(k, None)
    cfg.get("logging", {}).pop("wandb_tags", None)
    return cfg


def test_ladder_adds_one_switch_per_step_and_loo_removes_one():
    steps = [_sw(n) for n in LADDER]
    assert not any([steps[0]["rotation"], steps[0]["box"], steps[0]["channel"], steps[0]["utd_ratio"]])
    assert steps[0]["convergence"] == "rolling"
    for prev, cur, want in zip(steps, steps[1:], ["rotation", "box", "channel", "convergence", "utd_ratio"]):
        assert _diff(prev, cur) == [want], (want, _diff(prev, cur))
    full = steps[-1]
    for name, gone in LOO.items():
        assert _diff(full, _sw(name)) == [gone], name
    for n in LADDER + list(LOO):
        sw = _sw(n)
        assert sw["takeover_logging"] and sw["methods"] == ["hilserl"] and sw["peg_shape"] == "cuboid"
        assert not sw["channel"] or sw["rotation"]          # the channel needs rotation control


def test_pilots_identical_apart_from_the_switches():
    base = _strip(ss.load_yaml(ss.config_path("L0_original")))
    for n in LADDER[1:] + list(LOO):
        assert _strip(ss.load_yaml(ss.config_path(n))) == base, n
    assert base["operators"] == {"expert": 1, "intermediate": 1, "novice": 1, "jitter": 0.15, "sample_seed": 0}
    assert base["replicates"] == {"seeds": 1, "seed_base": 0}
    assert base["budget"]["steps_cap"] == 30000


def test_confirmation_keeps_the_thesis_base():
    full, ref = (ss.load_yaml(ss.config_path(n)) for n in ("C_full", "C_reference"))
    assert _strip(full) == _strip(ref)
    assert _diff(_sw("C_reference"), _sw("L0_original")) == []
    assert _diff(_sw("C_full"), _sw("L5_full")) == []
    b2 = ss.load_yaml(REPO / "examples" / "campaigns" / "rq4_ensemble_convergence.yaml")
    for blk in ("operators", "demos", "replicates", "method_overrides", "eval"):
        assert ref[blk] == b2[blk], blk
    for k in ("steps_cap", "converge_success", "converge_window", "converge_min_steps", "converge_eval_threshold",
              "demo_pretrain", "demo_pretrain_steps", "learning_rate"):
        assert ref["budget"][k] == b2["budget"][k], k
    for k in ("critic_ensemble_size", "critic_subsample_size", "per_member_targets"):
        assert ref["aqi"][k] == b2["aqi"][k], k
    assert ref["methods"] == ["hilserl", "logged"]          # + the asking arm, to be copied in


def test_every_config_passes_the_campaign_runner():
    import run_campaign as rc
    for n in ss.ORDER:
        d = Path(tempfile.mkdtemp()) / n
        with contextlib.redirect_stdout(io.StringIO()):
            rc.main(["--campaign", str(d), "--config", str(ss.config_path(n)), "--plan"])
        m = json.loads((d / "campaign.json").read_text())
        sw = _sw(n)
        assert ("sim_rpy_box" in m) == bool(sw["box"]), n
        assert ("synthetic_orientation_monitor" in m) == sw["channel"], n
        assert ("utd_ratio" in m) == (sw["utd_ratio"] > 0), n
        assert m.get("log_takeover_channels") is True, n


def test_limits_from_the_sweep():
    rows = [dict(shape="cuboid", axis="tilt", tier="expert", tolerance_deg=6.0),
            dict(shape="cuboid", axis="yaw", tier="expert", tolerance_deg=15.0),
            dict(shape="cuboid", axis="tilt", tier="novice", tolerance_deg=2.0),
            dict(shape="cylinder", axis="tilt", tier="expert", tolerance_deg=None)]
    lim, warn = ss.limits_from_sweep(rows)
    assert lim["cuboid"] == {"tilt_rad": 0.1047, "yaw_rad": 0.2618}
    assert lim["cylinder"] == {"tilt_rad": 0.0, "yaw_rad": 0.0} and len(warn) == 1


def _args(**kw):
    a = dict(wandb=None, extra=[], dry=True, keep_going=False, default_limits=False, without_active=False)
    a.update(kw)
    return type("A", (), a)()


def test_run_passes_limits_only_to_channel_settings_and_guards():
    root = Path(tempfile.mkdtemp())
    (root / "stage0").mkdir()
    (root / "stage0" / "limits.json").write_text(json.dumps(
        {"limits": {"cuboid": {"tilt_rad": 0.09, "yaw_rad": 0.0}}}))
    cmd, sw, lim = ss.campaign_cmd(_args(), "L3_channel", root)
    assert "--synthetic_orientation_tilt_rad" in cmd and "0.09" in cmd and "--synthetic_orientation_yaw_rad" not in cmd
    cmd, *_ = ss.campaign_cmd(_args(wandb="online"), "L2_box", root)
    assert not any("orientation" in c for c in cmd) and cmd[-2:] == ["--wandb", "online"]
    with contextlib.redirect_stdout(io.StringIO()):
        a = _args(names=["L3_channel"], root=str(Path(tempfile.mkdtemp())))
        assert ss.cmd_run(a) == 2                                       # no stage-0 limits
        a.default_limits = True
        assert ss.cmd_run(a) == 0
        assert ss.cmd_run(_args(names=["C_full"], root=str(root))) == 2  # asking arm missing
        assert ss.cmd_run(_args(names=["C_full"], root=str(root), without_active=True)) == 0
        camp = root / "L2_box"
        camp.mkdir()
        (camp / "suite_setting.json").write_text(json.dumps({"config": "x", "config_sha256": "old", "limits": None}))
        assert ss.cmd_run(_args(names=["L2_box"], root=str(root))) == 2  # another config in this folder


def _fake_cell(camp, cid, tier, eps, sr, conv=None, evals=(), minutes="12:34"):
    d = camp / "checkpoints" / cid
    d.mkdir(parents=True)
    (d / ".done").write_text(json.dumps({"cell": cid, "method": "hilserl", "operator": cid.split("_")[1],
                                         "tier": tier, "seed": 0, "converged": conv is not None,
                                         "converged_step": conv}))
    (d / "episodes.jsonl").write_text("\n".join(json.dumps(e) for e in eps) + "\n")
    if evals:
        (d / "convergence_evals.jsonl").write_text("\n".join(json.dumps(e) for e in evals) + "\n")
    (camp / "eval_runs" / cid).mkdir(parents=True)
    (camp / "eval_runs" / cid / "cell.json").write_text(json.dumps({"outcomes": {"sr": sr}}))
    (camp / "logs").mkdir(exist_ok=True)
    (camp / "logs" / f"{cid}.actor.log").write_text(f" 10%|#   | 3000/30000 [00:05<10:00, 8it/s]\r"
                                                    f" 40%|####| 12000/30000 [{minutes}<20:00, 8it/s]\n")


def _ep(step, length, success, n_int, steps_int, ch=None, hb=None):
    e = dict(step=step, length=length, success=success, intervention_count=n_int, intervention_steps=steps_int)
    if ch is not None:
        e.update(takeover_channels=ch, handbacks=hb)
    return e


def test_report_on_fake_campaigns():
    root = Path(tempfile.mkdtemp())
    eps0 = [_ep(300, 300, False, 2, 150, {"corridor": 2}, {"time_cap": 2}),
            _ep(600, 300, False, 1, 100, {"stagnation": 1}, {"signal": 1}),
            _ep(12000, 80, True, 0, 0, {}, {})]
    eps1 = [_ep(300, 300, False, 1, 60, {"orientation": 1}, {"signal": 1}),
            _ep(450, 150, True, 0, 0, {}, {}), _ep(9000, 90, True, 0, 0, {}, {})]
    _fake_cell(root / "L0_original", "hilserl_exp00_s0", "expert", eps0, 0.78, minutes="1:02:03")
    _fake_cell(root / "L1_rotation", "hilserl_exp00_s0", "expert", eps1, 0.89, conv=9000,
               evals=[{"step": 4000, "sr": 0.4}, {"step": 6000, "sr": 0.8}])
    m0 = ss.cell_metrics(root / "L0_original", root / "L0_original" / "checkpoints" / "hilserl_exp00_s0")
    assert m0["first_unaided_success"] == 12000 and m0["early_timeouts"] == 1.0 and m0["early_success"] == 0.0
    assert m0["early_time_cap_handbacks"] == round(2 / 3, 3) and m0["actor_minutes"] == 62.0
    assert m0["operator_share"] == round(250 / 680, 3) and m0["ruler_sr"] == 0.78
    m1 = ss.cell_metrics(root / "L1_rotation", root / "L1_rotation" / "checkpoints" / "hilserl_exp00_s0")
    assert m1["first_unaided_success"] == 450 and m1["early_orientation_takeovers"] == 1.0
    assert m1["mini_evals"] == 2 and m1["last_mini_eval_sr"] == 0.8 and m1["actor_minutes"] == 12.6
    with contextlib.redirect_stdout(io.StringIO()):
        assert ss.cmd_report(type("A", (), dict(root=str(root), names=[]))()) == 0
    text = (root / "report.md").read_text(encoding="utf-8")
    assert "## Against the original operator (L0)" in text
    assert "| L1_rotation | -11550 | -0.67 | 0.11 |" in text
    assert (root / "report_cells.csv").read_text(encoding="utf-8").count("hilserl_exp00_s0") == 2
    assert "| L1_rotation | L0_original | hilserl | 1 | 0.11 | 1 / 0 / 0 | 1.00 |" in text


def test_sign_test():
    assert ss.sign_test_p(0, 0) is None
    assert abs(ss.sign_test_p(13, 3) - 0.02127) < 1e-4          # the asking-rate campaign's q55 contrast
    assert ss.sign_test_p(5, 5) == 1.0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
