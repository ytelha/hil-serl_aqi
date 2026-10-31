"""Tests for the novice test: its config against the oracle campaign's, the cell copy, the report's fixed reading.

Run from the repo root with
    python -m pytest tests/test_novice_ablation.py -q
    python tests/test_novice_ablation.py
"""
import contextlib
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for _p in ("serl_launcher", "examples"):
    sys.path.insert(0, str(REPO / _p))

import copy_campaign_cells as cc  # noqa: E402
import novice_ablation_report as nar  # noqa: E402

CAMP = REPO / "examples" / "campaigns"
REUSED = ("logged", "quality-oracle")
NEW = ("oracle-nofilter", "oracle-nobc", "filter-only", "bc-only")


def _parse(name):
    import run_campaign as rc
    argv = ["--campaign", "unused", "--config", str(CAMP / f"{name}.yaml")]
    parser = rc.build_arg_parser()
    args = parser.parse_args(argv)
    args.ops_per_tier_map = rc.apply_config(args, rc.load_config(argv[3]), rc._explicit_dests(parser, argv)) or None
    return rc, args


def _cmds(rc, args, arm):
    ck = Path("runs/x/checkpoints/c")
    return (rc._learner_cmd(args, ck, Path("d.pkl"), arm, 0, "r"),
            rc._actor_cmd(args, ck, arm, "novice", "nov00", Path("t.json"), 0, "r"))


def test_config_equals_the_oracle_campaign_apart_from_arms_and_tier():
    import yaml
    a = yaml.safe_load((CAMP / "quality_novice_ablation.yaml").read_text(encoding="utf-8"))
    o = yaml.safe_load((CAMP / "aqi_oracle_quality.yaml").read_text(encoding="utf-8"))
    for blk in ("task", "demos", "replicates", "aqi", "eval", "gpu", "logging", "analysis"):
        assert a[blk] == o[blk], blk
    assert {k: v for k, v in a["budget"].items() if k != "total_hours"} \
        == {k: v for k, v in o["budget"].items() if k != "total_hours"}
    assert a["operators"]["novice"] == o["operators"]["novice"] == 3 and "expert" not in a["operators"]
    assert a["methods"] == list(REUSED + NEW)
    for arm in REUSED:
        assert a["arms"][arm] == o["arms"][arm], arm


def test_reference_arms_get_the_oracle_campaigns_commands():
    rc, ab = _parse("quality_novice_ablation")
    rc2, or_ = _parse("aqi_oracle_quality")
    for arm in REUSED:
        assert _cmds(rc, ab, arm) == _cmds(rc2, or_, arm), arm


def test_new_arms_differ_only_by_the_intended_flags():
    rc, ab = _parse("quality_novice_ablation")
    rc.validate_arms(ab, ab.methods.split(","))
    base = _cmds(rc, ab, "quality-oracle")[0]

    def diff(arm):
        c = _cmds(rc, ab, arm)[0]
        # the flag values that changed, as {flag: new value}; --arm and the run's name always differ
        out = {}
        for i, x in enumerate(c):
            if x.startswith("--") and x not in ("--arm",):
                v = c[i + 1] if i + 1 < len(c) and not c[i + 1].startswith("--") else True
                old = base[base.index(x) + 1] if x in base and base.index(x) + 1 < len(base) else None
                if x not in base or (v is not True and v != old):
                    out[x] = v
        return out
    assert diff("oracle-nofilter") == {"--quality_filters": "none"}
    assert diff("oracle-nobc") == {"--quality_filters": "none", "--bc_loss_weight": "0.0"}
    assert diff("filter-only") == {"--quality_score_mode": "uniform", "--bc_loss_weight": "0.0"}
    assert diff("bc-only") == {"--quality_score_mode": "uniform", "--quality_filters": "none"}


def _fake_source(root):
    src = Path(root) / "src"
    (src / "operators").mkdir(parents=True)
    (src / "eval_grid.json").write_text('{"sha256": "g"}')
    for arm in REUSED + ("aqi-oracle",):
        for op in ("nov00", "exp00"):
            for s in (0, 1):
                cid = f"{arm}_{op}_s{s}"
                d = src / "checkpoints" / cid
                (d / "checkpoint_1000").mkdir(parents=True)
                (d / "checkpoint_1000" / "state").write_text("big")
                (d / "wandb").mkdir()
                (d / ".done").write_text(json.dumps({"converged": True}))
                (d / "episodes.jsonl").write_text("{}\n")
                (src / "eval_runs" / cid).mkdir(parents=True)
                (src / "eval_runs" / cid / "cell.json").write_text(json.dumps({"outcomes": {"sr": 0.5}}))
                (src / "aqi_logs" / f"{cid}_actor").mkdir(parents=True)
                (src / "aqi_logs" / f"{cid}_actor" / "segments.jsonl").write_text("{}\n")
                (src / "logs").mkdir(exist_ok=True)
                (src / "logs" / f"{cid}.actor.log").write_text("x")
                (src / "demos" / f"{op}_s{s}").mkdir(parents=True, exist_ok=True)
                (src / "demos" / f"{op}_s{s}" / "demos.pkl").write_text(f"demos {op} {s}")
        (src / "operators" / "nov00.json").write_text('{"tier": "novice"}')
        (src / "operators" / "exp00.json").write_text('{"tier": "expert"}')
    return src


def _tree_hash(root):
    h = hashlib.sha256()
    for d, sub, files in sorted(os.walk(root)):
        sub.sort()
        for f in sorted(files):
            h.update(f.encode())
            h.update(open(os.path.join(d, f), "rb").read())
    return h.hexdigest()


def test_copy_selects_results_not_policies_and_leaves_the_source_alone():
    tmp = tempfile.mkdtemp(prefix="copycells_")
    src = _fake_source(tmp)
    before = _tree_hash(src)
    dst = Path(tmp) / "dst"
    args = ["--src", str(src), "--dst", str(dst), "--arms", "logged,quality-oracle", "--operators", "nov00"]
    with contextlib.redirect_stdout(io.StringIO()):
        assert cc.main(args + ["--dry"]) == 0
        assert not dst.exists()
        assert cc.main(args) == 0
    assert _tree_hash(src) == before
    cells = sorted(p.name for p in (dst / "checkpoints").iterdir())
    assert cells == ["logged_nov00_s0", "logged_nov00_s1", "quality-oracle_nov00_s0", "quality-oracle_nov00_s1"]
    c = dst / "checkpoints" / "logged_nov00_s0"
    assert (c / ".done").exists() and (c / "episodes.jsonl").exists()
    assert not (c / "checkpoint_1000").exists() and not (c / "wandb").exists()
    assert (dst / "eval_runs" / "logged_nov00_s0" / "cell.json").exists()
    assert (dst / "aqi_logs" / "logged_nov00_s0_actor" / "segments.jsonl").exists()
    assert (dst / "logs" / "logged_nov00_s0.actor.log").exists()
    assert (dst / "operators" / "nov00.json").exists() and not (dst / "operators" / "exp00.json").exists()
    assert (dst / "demos" / "nov00_s1" / "demos.pkl").read_text() == "demos nov00 1"
    assert (dst / "eval_grid.json").exists()
    rec = json.loads((dst / "copied_cells.json").read_text())
    assert len(rec["cells"]) == 4 and "operators/nov00.json" in rec["sha256"]
    with contextlib.redirect_stdout(io.StringIO()):
        assert cc.main(args) == 2                                     # already there


def test_copy_refuses_what_it_should():
    tmp = tempfile.mkdtemp(prefix="copycells_")
    src = _fake_source(tmp)
    dst = Path(tmp) / "dst"
    base = ["--src", str(src), "--dst", str(dst), "--operators", "nov00"]
    with contextlib.redirect_stdout(io.StringIO()):
        assert cc.main(base + ["--arms", "logged,no-such-arm"]) == 2
        os.remove(src / "checkpoints" / "logged_nov00_s0" / ".done")
        assert cc.main(base + ["--arms", "logged"]) == 2              # not finished
        (src / "checkpoints" / "logged_nov00_s0" / ".done").write_text("{}")
        os.remove(src / "eval_runs" / "logged_nov00_s0" / "cell.json")
        assert cc.main(base + ["--arms", "logged"]) == 2              # not evaluated
        (src / "eval_runs" / "logged_nov00_s0" / "cell.json").write_text("{}")
        dst.mkdir()
        (dst / "campaign.json").write_text("{}")
        assert cc.main(base + ["--arms", "logged"]) == 2              # already started
        assert cc.main(["--src", str(src), "--dst", str(src), "--arms", "logged"]) == 2
    assert not (dst / "checkpoints").exists()


def _srs(**kw):
    d = {"logged": 0.65, "quality-oracle": 0.32, "oracle-nofilter": 0.65, "oracle-nobc": 0.65,
         "filter-only": 0.65, "bc-only": 0.65}
    d.update({k.replace("_", "-"): v for k, v in kw.items()})
    return d


def test_reading_follows_the_rule():
    low = 0.3
    assert nar.reading(_srs(filter_only=low))[0] == "filter"
    assert nar.reading(_srs(bc_only=low, oracle_nofilter=low))[0] == "bc"
    assert nar.reading(_srs(filter_only=low, bc_only=low))[0] == "filter_and_bc"
    assert nar.reading(_srs(oracle_nobc=low))[0] == "score"
    assert nar.reading(_srs(oracle_nobc=low, oracle_nofilter=low))[0] == "score"
    assert nar.reading(_srs(oracle_nofilter=low))[0] == "score_and_bc"
    assert nar.reading(_srs())[0] == "filter_with_score"
    assert nar.reading(_srs(bc_only=low))[0] == "mixed"
    assert nar.reading(_srs(filter_only=low, oracle_nofilter=low))[0] == "mixed"
    assert nar.reading(_srs(logged=0.40))[0] == "no_gap"
    assert nar.reading({"logged": 0.6})[0] == "incomplete"


def _fake_campaign(root, sr_by_arm):
    c = Path(root) / "camp"
    for arm, sr in sr_by_arm.items():
        for op in ("nov00", "nov01", "nov02"):
            for s in (0, 1):
                cid = f"{arm}_{op}_s{s}"
                (c / "eval_runs" / cid).mkdir(parents=True)
                (c / "eval_runs" / cid / "cell.json").write_text(json.dumps(
                    {"outcomes": {"sr": sr}, "sr_per_checkpoint": [{"sr": 0.0}, {"sr": min(1.0, sr + 0.1)}]}))
                (c / "checkpoints" / cid).mkdir(parents=True)
                (c / "checkpoints" / cid / ".done").write_text(json.dumps({"converged": s == 0}))
                (c / "aqi_logs" / f"{cid}_actor").mkdir(parents=True)
                drop = arm in ("quality-oracle", "filter-only")
                (c / "aqi_logs" / f"{cid}_actor" / "segments.jsonl").write_text("\n".join(json.dumps(
                    {"length": 10, "episode_success": i == 0, "excluded": bool(drop and i > 0),
                     "exclusion_reason": "outcome_filter" if drop and i > 0 else "", "q_int_mean": 0.0}) for i in range(4)))
    return c


def test_report_on_a_fake_campaign():
    tmp = tempfile.mkdtemp(prefix="novice_")
    c = _fake_campaign(tmp, _srs(filter_only=0.3))
    with contextlib.redirect_stdout(io.StringIO()):
        assert nar.main(["--campaign", str(c)]) == 0
    text = (c / "report" / "novice_ablation.md").read_text(encoding="utf-8")
    assert "**Reading (`filter`):**" in text and "| filter-only |" in text and "30.0 %" in text
    res = json.loads((c / "report" / "novice_ablation.json").read_text())
    v = res["arms"]["filter-only"]
    assert v["cells"] == 6 and v["converged"] == 3 and abs(v["dropped"] - 0.75) < 1e-9 and abs(v["last_sr"] - 0.4) < 1e-9
    assert v["vs_logged"]["pairs"] == 6 and v["vs_logged"]["losses"] == 6 and abs(v["vs_logged"]["mean"] + 0.35) < 1e-9
    assert res["arms"]["logged"]["vs_logged"]["ties"] == 6
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert nar.main(["--count", str(c / "aqi_logs")]) == 0
    assert "| filter-only | nov | 24 | 75 % | 75 % | 75 % | 0 % | outcome_filter 18 |" in out.getvalue()
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
