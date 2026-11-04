"""Tests for the shorter ruler: which checkpoints are kept, the campaign's flags per arm, the mixed-selection guard,
and the re-selection of finished cells from their trials.

Run from the repo root with
    python -m pytest tests/test_ruler_select.py -q
    python tests/test_ruler_select.py
"""
import contextlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for _p in ("serl_launcher", "examples"):
    sys.path.insert(0, str(REPO / _p))

import ruler_select as rs  # noqa: E402
import ruler_reselect as rr  # noqa: E402

CAMP = REPO / "examples" / "campaigns"
NEW_FLAGS = ("--select_min_step", "--select_stride", "--select_newest")
STEPS = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 10400]


def test_keep_checkpoints():
    assert rs.keep_checkpoints(STEPS) == STEPS                                  # off: every checkpoint
    assert rs.keep_checkpoints(STEPS, min_step=5000) == [6000, 7000, 8000, 9000, 10000, 10400]
    assert rs.keep_checkpoints([1000, 3000], min_step=5000) == [3000]          # stopped early: the newest
    assert rs.keep_checkpoints(STEPS, stride=2) == [1000, 3000, 5000, 7000, 9000, 10400]   # newest always scored
    assert rs.keep_checkpoints(STEPS, newest=4) == [8000, 9000, 10000, 10400]
    assert rs.keep_checkpoints(STEPS, min_step=5000, newest=8) == [6000, 7000, 8000, 9000, 10000, 10400]
    assert rs.keep_checkpoints(STEPS, min_step=5000, stride=2, newest=2) == [9000, 10400]
    assert rs.keep_checkpoints([]) == []
    assert rs.describe() is None and rs.describe(0, 1, 0) is None
    assert rs.describe(5000, 1, 8) == {"min_step": 5000, "stride": 1, "newest": 8}
    assert not rs.is_active(None, None, None) and rs.is_active(newest=8)


def test_best_window_as_eval_ruler():
    pts = [{"step": s, "sr": v} for s, v in zip(STEPS, [0, 0, 0, 0, 0, 0.1, 1, 1, 0.5, 1, 1])]
    sr, steps = rs.best_window(pts, 3)       # three windows reach 5/6: the earliest wins
    assert abs(sr - 5 / 6) < 1e-12 and steps == [7000, 8000, 9000]
    tie = [{"step": s, "sr": v} for s, v in zip([1, 2, 3, 4], [1, 1, 1, 1])]
    assert rs.best_window(tie, 2) == (1.0, [1, 2])                             # a tie goes to the earliest window
    nan = [{"step": 1, "sr": float("nan")}, {"step": 2, "sr": 0.5}]
    assert rs.best_window(nan, 3) == (0.5, [2])                                 # NaN skipped, window shrinks
    assert rs.best_window([], 3) == (None, None)
    assert rs.best_single(tie) == (1.0, 1) and rs.mean_all(pts) == sum(p["sr"] for p in pts) / len(pts)


def _parse(path, extra=()):
    import run_campaign as rc
    argv = ["--campaign", "unused", "--config", str(path), *extra]
    parser = rc.build_arg_parser()
    args = parser.parse_args(argv)
    args.ops_per_tier_map = rc.apply_config(args, rc.load_config(str(path)), rc._explicit_dests(parser, argv)) or None
    return rc, args


def _eval_cmd(rc, args, arm, campaign):
    args.dry_run = True
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc.eval_cell(args, Path(campaign), rc.Cell(arm, "exp00", "expert", 0))
    line = [x for x in out.getvalue().splitlines() if "DRY: python" in x][0]
    return re.sub(r"\x1b\[[0-9;]*m", "", line).split("DRY: python ", 1)[1].split()


def test_campaign_flags_off_by_default_and_per_arm_when_on():
    tmp = tempfile.mkdtemp(prefix="ruler_")
    cfg = CAMP / "synthetic_suite" / "C_full.yaml"           # hilserl opts out of pre-training, the others do 5000
    rc, args = _parse(cfg)
    for arm in args.methods.split(","):
        cmd = _eval_cmd(rc, args, arm, tmp)
        assert not any(f in cmd for f in NEW_FLAGS), (arm, cmd)
        assert rc._ruler_settings(args) is None
    rc, args = _parse(cfg, ["--eval_skip_pretrain", "--eval_newest", "8"])
    hil = _eval_cmd(rc, args, "hilserl", tmp)
    assert "--select_min_step" not in hil and hil[hil.index("--select_newest") + 1] == "8"
    other = [a for a in args.methods.split(",") if a != "hilserl"][0]
    cmd = _eval_cmd(rc, args, other, tmp)
    assert cmd[cmd.index("--select_min_step") + 1] == "5000" and "--select_stride" not in cmd
    assert rc._ruler_settings(args) == {"skip_pretrain": True, "stride": 1, "newest": 8}
    shutil.rmtree(tmp, ignore_errors=True)


def test_yaml_keys():
    tmp = tempfile.mkdtemp(prefix="ruler_")
    src = (CAMP / "inconsistent_operator" / "two_arm_confirm_new.yaml").read_text(encoding="utf-8")
    assert "\neval:\n" in src
    cfg = Path(tmp) / "c.yaml"
    cfg.write_text(src.replace("\neval:\n", "\neval:\n  skip_pretrain: true\n  stride: 2\n  newest: 6\n"), encoding="utf-8")
    rc, args = _parse(cfg)
    assert (args.eval_skip_pretrain, args.eval_stride, args.eval_newest) == (True, 2, 6)
    cmd = _eval_cmd(rc, args, "drop-hind", tmp)
    for f, v in zip(NEW_FLAGS, ("5000", "2", "6")):
        assert cmd[cmd.index(f) + 1] == v, f
    shutil.rmtree(tmp, ignore_errors=True)


def _cell_json(path, sr_by_step, select="best_window", k=3, filt=None):
    pts = [{"step": s, "sr": v, "n_trials": 3} for s, v in sr_by_step.items()]
    bw, bws = rs.best_window(pts, k)
    trials = []
    for s, v in sr_by_step.items():
        n_ok = round(v * 3)
        trials += [{"trial": i, "checkpoint_step": s, "success": i < n_ok, "steps": 20 if i < n_ok else 300,
                    "time_s": 2.0 if i < n_ok else None, "f_peak": 1.0, "f_mean": 0.1, "intervened": False}
                   for i in range(3)]
    win = [t for t in trials if t["checkpoint_step"] in bws]
    cj = {"cell_id": path.name, "select": select, "used_certified": False, "best_window_sr": bw,
          "best_window_steps": bws, "sr_per_checkpoint": pts, "f_max": 30.0,
          "eval_checkpoint_steps": list(sr_by_step),
          "outcomes": {"sr": sum(t["success"] for t in win) / len(win)}}
    if filt:
        cj["checkpoint_filter"] = filt
    path.mkdir(parents=True, exist_ok=True)
    (path / "cell.json").write_text(json.dumps(cj), encoding="utf-8")
    (path / "trials.jsonl").write_text("".join(json.dumps(t) + "\n" for t in trials), encoding="utf-8")


def _manifest(camp, cell, pretrain):
    d = camp / "checkpoints" / cell
    d.mkdir(parents=True, exist_ok=True)
    flag = "--demo_pretrain" if pretrain else "--nodemo_pretrain"
    (d / "manifest.json").write_text(json.dumps({"reproduce": [
        f"# 1) learner\npython train_aqi_rlpd.py --learner {flag} --demo_pretrain_steps {pretrain or 0} --seed 0"]}))


def _fake_campaign(tmp):
    camp = Path(tmp) / "camp"
    late = {1000: 0, 2000: 0, 3000: 0, 4000: 0, 5000: 0, 6000: 1 / 3, 7000: 1, 8000: 1, 9000: 1}
    early = {1000: 0, 2000: 1, 3000: 1, 4000: 1, 5000: 0, 6000: 0, 7000: 0, 8000: 1 / 3, 9000: 2 / 3}
    for cell, srs, pre in (("logged_exp00_s0", late, 5000), ("hilserl_exp00_s0", early, 0)):
        _cell_json(camp / "eval_runs" / cell, srs)
        _manifest(camp, cell, pre)
    return camp


def test_reselect_report_and_apply_only_on_copied_cells():
    tmp = tempfile.mkdtemp(prefix="ruler_")
    camp = _fake_campaign(tmp)
    r = rr.reselect_cell(str(camp), "logged_exp00_s0", (True, 1, 4, None), 3)
    assert r["kept"] == [6000, 7000, 8000, 9000] and r["old"] == r["new"] == 1.0
    h = rr.reselect_cell(str(camp), "hilserl_exp00_s0", (True, 1, 4, None), 3)
    assert h["want"] == (0, 1, 4) and h["kept"] == [6000, 7000, 8000, 9000]
    assert h["old"] == 1.0 and abs(h["new"] - 1 / 3) < 1e-9                      # its best window was early
    before = {c: (camp / "eval_runs" / c / "cell.json").read_text() for c in ("logged_exp00_s0", "hilserl_exp00_s0")}
    with contextlib.redirect_stdout(io.StringIO()):
        assert rr.main(["--campaign", str(camp), "--skip_pretrain", "--newest", "4"]) == 0     # read-only
        assert rr.main(["--campaign", str(camp), "--newest", "4", "--apply"]) == 2              # not a copy
    assert {c: (camp / "eval_runs" / c / "cell.json").read_text() for c in before} == before
    (camp / "copied_cells.json").write_text(json.dumps({"cells": ["hilserl_exp00_s0"]}))
    with contextlib.redirect_stdout(io.StringIO()):
        assert rr.main(["--campaign", str(camp), "--skip_pretrain", "--newest", "4", "--apply"]) == 0
    assert (camp / "eval_runs" / "logged_exp00_s0" / "cell.json").read_text() == before["logged_exp00_s0"]
    new = json.loads((camp / "eval_runs" / "hilserl_exp00_s0" / "cell.json").read_text())
    assert (camp / "eval_runs" / "hilserl_exp00_s0" / "cell.full_ruler.json").read_text() == before["hilserl_exp00_s0"]
    assert new["checkpoint_filter"]["newest"] == 4 and new["checkpoint_filter"]["n_saved"] == 9
    assert new["best_window_steps"] == [7000, 8000, 9000] and abs(new["outcomes"]["sr"] - 1 / 3) < 1e-9
    assert new["outcomes"]["n_trials"] == 9 and "reselected" in new
    with contextlib.redirect_stdout(io.StringIO()):                              # applying twice changes nothing
        assert rr.main(["--campaign", str(camp), "--skip_pretrain", "--newest", "4", "--apply"]) == 0
    assert json.loads((camp / "eval_runs" / "hilserl_exp00_s0" / "cell.json").read_text()) == new
    shutil.rmtree(tmp, ignore_errors=True)


def test_mixed_selection_is_found_before_training():
    tmp = tempfile.mkdtemp(prefix="ruler_")
    camp = _fake_campaign(tmp)
    cfg = CAMP / "synthetic_suite" / "C_full.yaml"
    rc, args = _parse(cfg)
    cells = [rc.Cell("logged", "exp00", "expert", 0), rc.Cell("hilserl", "exp00", "expert", 0)]
    assert rc.check_ruler_filter(args, camp, cells) == []                      # both scored on every checkpoint
    rc, args = _parse(cfg, ["--eval_skip_pretrain", "--eval_newest", "4"])
    mixed = rc.check_ruler_filter(args, camp, cells)
    assert {m[0] for m in mixed} == {"logged_exp00_s0", "hilserl_exp00_s0"}
    (camp / "copied_cells.json").write_text(json.dumps({"cells": ["logged_exp00_s0", "hilserl_exp00_s0"]}))
    with contextlib.redirect_stdout(io.StringIO()):
        assert rr.main(["--campaign", str(camp), "--skip_pretrain", "--newest", "4", "--apply"]) == 0
    assert rc.check_ruler_filter(args, camp, cells) == []                      # re-selected: consistent again
    shutil.rmtree(tmp, ignore_errors=True)


def test_verify_a_fresh_short_run_against_the_original():
    tmp = tempfile.mkdtemp(prefix="ruler_")
    camp = _fake_campaign(tmp)
    orig = camp / "eval_runs" / "logged_exp00_s0"
    fresh = Path(tmp) / "fresh"
    fresh.mkdir()
    lines = [json.loads(x) for x in (orig / "trials.jsonl").read_text().splitlines()]
    short = [t for t in lines if t["checkpoint_step"] > 5000]
    (fresh / "trials.jsonl").write_text("".join(json.dumps(t) + "\n" for t in short))
    with contextlib.redirect_stdout(io.StringIO()) as out:
        assert rr.main(["--verify", str(fresh), "--against", str(orig)]) == 0
    assert "12 trials on 4 checkpoints" in out.getvalue() and "IDENTICAL" in out.getvalue()
    (camp / "logs").mkdir()
    (camp / "logs" / "logged_exp00_s0.eval.log").write_text(
        "# source /r/examples/experiments/sim_peg_insertion/sim_env.sh && exec python eval_ruler.py --exp_name x "
        "--checkpoint_path /r/runs/c/checkpoints/logged_exp00_s0 --out /r/runs/c/eval_runs/logged_exp00_s0 "
        "--argmax --select best_window --noprefer_certified\n[ruler] ...\n")
    with contextlib.redirect_stdout(io.StringIO()) as out:
        assert rr.main(["--campaign", str(camp), "--skip_pretrain", "--newest", "8", "--check_cmd", "logged_exp00_s0"]) == 0
    cmd = out.getvalue().strip()
    assert cmd.startswith("cd /r/examples && SIM_DISABLE_SPACEMOUSE=1 SIM_RPY_BOX='' bash -c 'source ")
    assert "&& exec python eval_ruler.py" in cmd and "/r/runs/c/eval_runs" not in cmd
    assert cmd.endswith("--out /tmp/ruler_check/logged_exp00_s0 --select_min_step 5000 --select_newest 8'")
    short[0] = dict(short[0], success=not short[0]["success"], steps=17)
    (fresh / "trials.jsonl").write_text("".join(json.dumps(t) + "\n" for t in short))
    with contextlib.redirect_stdout(io.StringIO()) as out:
        assert rr.main(["--verify", str(fresh), "--against", str(orig)]) == 1
    assert "NOT identical" in out.getvalue()
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
