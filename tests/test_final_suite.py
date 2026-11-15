"""Tests for the final study (examples/campaigns/final/, `synthetic_suite.py --suite final`, docs/final_study.md):
every config passes the campaign runner; every arm uses only flags the training script defines; the design reaches
the commands (the four asking x grading cells on one pipeline, the controls, the careless operator at 40 % in every
arm, held requests, the stronger novice, the ensemble base, no outcome filter, no imitation); checkpoint retention;
the simulator guard; crashed runs are marked and retried once; the unattended run decides without stopping and never
pushes. No JAX, no MuJoCo, no training.

Run from the repo root with
    python -m pytest tests/test_final_suite.py -q
    python tests/test_final_suite.py
"""
import contextlib
import io
import json
import re
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
for _p in ("serl_launcher", "examples", "franka_sim"):
    sys.path.insert(0, str(REPO / _p))

import ckpt_retention as cr                    # noqa: E402
import run_campaign as rc                      # noqa: E402
import synthetic_suite as ss                   # noqa: E402

FINAL = REPO / "examples" / "campaigns" / "final"
ss.configure("final")
FACTORIAL = ("keepall", "asks-15", "beh-discard", "aqi-15")
ARMS = ["hilserl", "keepall", "asks-15", "beh-discard", "aqi-15", "perfect-filter", "random-15"]


def _parse(name, extra=()):
    cfg_p = str(FINAL / f"{name}.yaml")
    argv = ["--campaign", "unused", "--config", cfg_p, *extra]
    parser = rc.build_arg_parser()
    args = parser.parse_args(argv)
    args.ops_per_tier_map = rc.apply_config(args, rc.load_config(cfg_p), rc._explicit_dests(parser, argv)) or None
    rc.validate_arms(args, args.methods.split(","))
    return args


def _defined(script):
    text = (REPO / "examples" / script).read_text(encoding="utf-8")
    return set(re.findall(r'flags\.DEFINE_\w+\(\s*"(\w+)"', text))


def _flags(cmd):
    return {t[2:].split("=", 1)[0] for t in cmd if isinstance(t, str) and t.startswith("--")}


def _cmds(args, arm, tier="expert", op="exp00"):
    ck = Path("runs/x/checkpoints/c")
    return (rc._learner_cmd(args, ck, Path("d.pkl"), arm, 0, "r"),
            rc._actor_cmd(args, ck, arm, tier, op, Path("t.json"), 0, "r"))


def _eval_cmd(args, arm):
    args.dry_run = True
    seen = []
    orig = rc._yellow
    rc._yellow = seen.append
    try:
        rc.eval_cell(args, Path(tempfile.mkdtemp()), rc.Cell(arm, "exp00", "expert", 0))
    finally:
        rc._yellow = orig
    return seen[0]


def _patch(**kw):
    old = {k: getattr(ss, k) for k in kw}
    for k, v in kw.items():
        setattr(ss, k, v)
    return old


# ---------------------------------------------------------------- the suite and the configs
def test_suite_groups_order_and_settings():
    assert ss.resolve(["smoke"]) == ["F0_smoke"] and ss.resolve(["main"]) == ["F1_hard", "F2_standard"]
    assert ss.ORDER == ["F0_smoke", "F1_hard", "F2_standard"]
    assert set(ss.REF_ARMS) == {"hilserl", "keepall"} and "tests/test_final_suite.py" in ss.TESTS
    d = ss.SUITES["final"]
    assert d["ruler_check"] is False and d["operator_tiers"] == "expert,intermediate,s=0.3"
    assert [ss.planned_cells(n) for n in ss.ORDER] == [6, 84, 84]


def test_every_config_passes_the_campaign_runner():
    want = {"F0_smoke": ({"expert": 1, "novice": 1}, 3, 3), "F1_hard": ({"expert": 2, "intermediate": 2, "novice": 2}, 7, 3),
            "F2_standard": ({"expert": 2, "intermediate": 2, "novice": 2}, 7, 6)}
    for n, (ops, n_arms, demos) in want.items():
        d = Path(tempfile.mkdtemp()) / n
        with contextlib.redirect_stdout(io.StringIO()):
            rc.main(["--campaign", str(d), "--config", str(ss.config_path(n)), "--plan"])
        m = json.loads((d / "campaign.json").read_text())
        assert {k: v for k, v in m["operators_per_tier"].items() if v} == ops, n
        assert m["synthetic_rotation_control"] is True and m["utd_ratio"] == 1.0 and m["log_takeover_channels"], n
        assert m["synthetic_inconsistent"] == {"bad_share": 0.4, "persistence": 0.7}, n
        assert len(m["methods"]) == n_arms and (m.get("converge_mode") or {}).get("mode") == "eval", n
        assert int(_parse(n).demos) == demos, n


def test_F1_and_F2_differ_only_in_the_demonstrations():
    a, b = vars(_parse("F1_hard")), vars(_parse("F2_standard"))
    diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert diff <= {"demos", "config", "wandb_tags"}, diff
    assert int(a["demos"]) == 3 and int(b["demos"]) == 6


def test_every_arm_uses_only_defined_flags():
    defined = _defined("train_aqi_rlpd.py")
    for n in ss.ORDER:
        args = _parse(n)
        for arm in args.methods.split(","):
            for cmd in _cmds(args, arm):
                assert cmd[0] == "train_aqi_rlpd.py", (n, arm)
                used = _flags(cmd)
                neg = {u for u in used if u.startswith("no") and u[2:] in defined}
                assert not (used - defined - neg), (n, arm, used - defined - neg)


# ---------------------------------------------------------------- the design reaches the commands
def test_the_four_cells_differ_only_in_the_two_switches():
    args = _parse("F1_hard")
    assert args.methods.split(",") == ARMS
    s = {arm: [" ".join(c) for c in _cmds(args, arm)] for arm in ARMS}
    for arm in ("keepall", "beh-discard", "perfect-filter"):
        assert "--aqi_mode quality" in s[arm][0], arm               # the operator decides
    for arm in ("asks-15", "aqi-15", "random-15"):
        assert "--aqi_mode aqi" in s[arm][0], arm                   # the robot asks; same quality pipeline
    for arm in ("asks-15", "aqi-15"):
        assert "--query_target_rate 0.15" in s[arm][0] and "--trigger_terms uncertainty" in s[arm][0], arm
    assert "--trigger_terms none" in s["random-15"][0] and "--random_request_rate 0.05" in s["random-15"][0]
    for arm in ("keepall", "asks-15", "random-15"):
        assert "--quality_filters none" in s[arm][0], arm           # keep every correction
    for arm in ("beh-discard", "aqi-15"):
        assert "--quality_filters score_filter" in s[arm][0] and "--quality_drop_fraction 0.4" in s[arm][0], arm
    assert "--quality_filters oracle_filter" in s["perfect-filter"][0]

    def strip(cmd):                                                  # everything but the two switches and names
        cmd = re.sub(r"--aqi_mode \S+|--query_target_rate \S+|--query_cooldown_steps \S+|--quality_filters \S+|"
                     r"--quality_drop_fraction \S+|--arm \S+|--run_name \S+|--trigger_terms \S+|--trigger_combiner \S+|"
                     r"--synthetic_request_hold_steps \S+",
                     "", cmd)
        return " ".join(cmd.split())
    base = strip(s["keepall"][0])
    for arm in ("asks-15", "beh-discard", "aqi-15"):
        assert strip(s[arm][0]) == base, arm


def test_careless_operator_held_requests_and_no_filters_or_imitation():
    args = _parse("F1_hard")
    for arm in ARMS:
        lc, ac = (" ".join(c) for c in _cmds(args, arm))
        assert "--synthetic_inconsistent --synthetic_bad_share 0.4 --synthetic_bad_persistence 0.7" in ac, arm
        assert "outcome_filter" not in lc + ac and "safety_veto" not in lc + ac, arm
        if "--bc_loss_weight" in lc:
            assert "--bc_loss_weight 0.0" in lc, arm
        assert "--utd_ratio 1" in lc and "--utd_ratio 1" in ac, arm
    for arm in ("asks-15", "aqi-15", "random-15"):
        assert "--synthetic_request_hold_steps 30" in " ".join(_cmds(args, arm)[1]), arm


def test_ensemble_base_and_hilserl_reference():
    args = _parse("F1_hard")
    for arm in ARMS[1:]:
        lc = " ".join(_cmds(args, arm)[0])
        assert "--critic_ensemble_size 10 " in lc and "--critic_subsample_size 2 " in lc, arm
        assert "--demo_pretrain --demo_pretrain_steps 5000" in lc and "--learning_rate" not in lc, arm
    hl = " ".join(_cmds(args, "hilserl")[0])
    assert "--aqi_mode hilserl" in hl and "--nodemo_pretrain" in hl and "--learning_rate" not in hl


def test_checkpoint_retention_and_ruler():
    for n in ss.ORDER:
        args = _parse(n)
        for arm in args.methods.split(","):
            lc = " ".join(_cmds(args, arm)[0])
            assert "--checkpoint_keep 8 --checkpoint_keep_every 2000 --checkpoint_keep_until 17000" in lc, (n, arm)
            ev = _eval_cmd(args, arm)
            assert "--select best_window" in ev and "--select_newest" not in ev, (n, arm, ev)
            assert ("--select_min_step" in ev) == (arm != "hilserl"), (n, arm, ev)    # warm-start checkpoints skipped
    off = _parse("F1_hard", extra=["--checkpoint_keep_every", "0"])                       # the disk fallback
    assert "--checkpoint_keep_every" not in " ".join(_cmds(off, "keepall")[0])
    other = (REPO / "examples" / "campaigns" / "study" / "B_quality.yaml")
    parser = rc.build_arg_parser()
    argv = ["--campaign", "unused", "--config", str(other)]
    a = parser.parse_args(argv)
    rc.apply_config(a, rc.load_config(str(other)), rc._explicit_dests(parser, argv))
    assert "--checkpoint_keep_every" not in " ".join(_cmds(a, "keepall")[0])                 # earlier campaigns unchanged


def test_the_stronger_novice():
    roster = rc.sample_operators(["expert", "intermediate", "novice"], {"expert": 2, "intermediate": 2, "novice": 2},
                                 0, 0.15, tier_s_exec={"novice": 0.3}, tier_s_mon={"novice": 0.3})
    nov = [t for t in roster if t.tier == "novice"]
    assert len(nov) == 2 and all(abs(t.s_exec - 0.3) < 1e-9 and abs(t.s_mon - 0.3) < 1e-9 for t in nov)
    from serl_launcher.aqi.operator_model import EXEC_ANCHORS, MON_ANCHORS
    # between the old novice and the expert, before jitter: the reaction delay and the monitoring threshold
    th = rc.theta_from_scale(0.3, 0.3)
    assert MON_ANCHORS["expert"]["l_react"] <= th.l_react < MON_ANCHORS["novice"]["l_react"]
    assert MON_ANCHORS["expert"]["delta_in"] < th.delta_in < MON_ANCHORS["novice"]["delta_in"]
    assert EXEC_ANCHORS["expert"]["sigma0"] < th.sigma0 < EXEC_ANCHORS["novice"]["sigma0"]
    a = _parse("F1_hard")
    assert a.tier_s_exec == {"novice": 0.3} and a.tier_s_mon == {"novice": 0.3}


# ---------------------------------------------------------------- checkpoint retention, guard, crashes
def test_steps_to_keep_and_prune():
    steps = list(range(1000, 26000, 1000))                           # learner 1 000 .. 25 000
    kept = cr.steps_to_keep(steps, 8, 2000, 17000)
    assert kept == set(range(18000, 26000, 1000)) | set(range(2000, 18000, 2000))
    assert cr.steps_to_keep(steps, 8) == set(range(18000, 26000, 1000))
    d = Path(tempfile.mkdtemp())
    for s in steps:
        (d / f"checkpoint_{s}").mkdir()
    (d / "checkpoint_26000.orbax-checkpoint-tmp-1").mkdir()          # a save in progress is left alone
    (d / "certified").mkdir()
    gone = cr.prune_checkpoints(str(d), 8, 2000, 17000)
    assert sorted(gone) == sorted(set(steps) - kept)
    assert cr.checkpoint_steps(str(d)) == sorted(kept)
    assert (d / "checkpoint_26000.orbax-checkpoint-tmp-1").exists() and (d / "certified").exists()


def test_learner_saves_through_the_retention_helper():
    src = (REPO / "examples" / "train_aqi_rlpd.py").read_text(encoding="utf-8")
    assert src.count("_save_run_checkpoint(agent.state") == 4
    assert "keep=FLAGS.checkpoint_keep" not in src.replace("keep=(10 ** 9 if every > 0 else FLAGS.checkpoint_keep)", "")


def test_simulator_guard():
    import importlib.util                                           # numpy only: no MuJoCo import
    spec = importlib.util.spec_from_file_location(
        "orientation_box", REPO / "franka_sim" / "franka_sim" / "envs" / "orientation_box.py")
    ob = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ob)
    safe_action, safe_quat = ob.safe_action, ob.safe_quat
    q, used = safe_quat([0, 0, 0, 2.0], [0, 0, 0, 1])
    assert np.allclose(q, [0, 0, 0, 1]) and not used
    q, used = safe_quat([0, 0, 0, 0], [0, 0.6, 0, 0.8])
    assert np.allclose(q, [0, 0.6, 0, 0.8]) and used
    q, used = safe_quat([np.nan, 0, 0, 1], [0, 0, 0, 0])
    assert np.allclose(q, [0, 0, 0, 1]) and used
    a, bad = safe_action([0.1, np.nan, np.inf, 0, 0, 0, 0])
    assert bad and np.allclose(a, [0.1, 0, 0, 0, 0, 0, 0])
    a, bad = safe_action([0.1, 0.2, 0, 0, 0, 0, 0])
    assert not bad
    env_src = (REPO / "franka_sim" / "franka_sim" / "envs" / "panda_peg_insert_gym_env.py").read_text(encoding="utf-8")
    assert "self._des_quat, bad_new = safe_quat(new_quat, prev_quat)" in env_src
    assert "Rotation.from_quat(self._des_quat)" not in env_src       # never read back unchecked


def test_actor_crash_detection_and_eval_skip():
    d = Path(tempfile.mkdtemp())
    log = d / "a.log"
    log.write_text("step 1\nTraceback (most recent call last):\n  ...\nValueError: Found zero norm quaternions\n")
    assert rc.actor_crashed(log, -15, True)                          # a crash that hung until the time cap
    log.write_text("all fine\n")
    assert not rc.actor_crashed(log, -15, True)                      # the time cap alone is not a crash
    assert rc.actor_crashed(log, 1, False) and not rc.actor_crashed(log, 0, False)
    camp = d / "camp"
    ck = camp / "checkpoints" / "aqi-15_nov00_s0"
    (ck / "checkpoint_9000").mkdir(parents=True)
    (ck / ".crashed").write_text("{}")
    args = _parse("F1_hard")
    args.dry_run = False
    with contextlib.redirect_stdout(io.StringIO()):
        assert rc.eval_cell(args, camp, rc.Cell("aqi-15", "nov00", "novice", 0)) is False
    assert not (camp / "eval_runs" / "aqi-15_nov00_s0").exists()
    assert ss.crashed_cells(camp) == ["aqi-15_nov00_s0"]
    (ck / ".done").write_text("{}")
    assert ss.crashed_cells(camp) == []


def test_disk_decision_per_campaign():
    st = ss.RunStatus(Path(tempfile.mkdtemp()))
    old_usage = ss.shutil.disk_usage
    try:
        ss.shutil.disk_usage = lambda p: types.SimpleNamespace(free=1000 * 2 ** 30)
        assert ss.disk_decision(st.root, "F1_hard", st) == []
        ss.shutil.disk_usage = lambda p: types.SimpleNamespace(free=50 * 2 ** 30)
        assert ss.disk_decision(st.root, "F2_standard", st) == ["--checkpoint_keep_every", "0"]
    finally:
        ss.shutil.disk_usage = old_usage
    assert "early checkpoints stay" in st.decisions["disk F1_hard"]
    assert "newest checkpoints only" in st.decisions["disk F2_standard"] and st.warnings
    assert ss.disk_needed_gb(["F1_hard"]) == round(84 * 1.6 + ss.DISK_MARGIN_GB, 1)


def test_all_runs_everything_retries_crashes_once_and_never_pushes():
    root = Path(tempfile.mkdtemp()) / "runs" / "final"
    calls = []

    def rec(name):
        def f(a):
            names = list(getattr(a, "names", []) or [])
            calls.append((name, names, list(getattr(a, "extra", []) or [])))
            if name == "run" and names == ["F1_hard"]:
                ck = root / "F1_hard" / "checkpoints" / "aqi-15_nov00_s0"
                ck.mkdir(parents=True, exist_ok=True)
                if sum(1 for c in calls if c[0] == "run" and c[1] == ["F1_hard"]) == 1:
                    (ck / ".crashed").write_text("{}")               # crashes the first time ...
                else:
                    (ck / ".done").write_text("{}")                  # ... and finishes on the retry
            return 0
        return f

    pushed = []
    old = _patch(cmd_check=rec("check"), cmd_stage0=rec("stage0"), cmd_run=rec("run"), cmd_rulercheck=rec("rulercheck"),
                 cmd_audit=rec("audit"), cmd_report=rec("report"), cmd_plots=rec("plots"),
                 export_results=lambda *a, **k: 0, wandb_logged_in=lambda *a, **k: False,
                 parallel_decision=lambda r, n, req: (2, "test: faster side by side"),
                 push_results=lambda *a, **k: pushed.append(1) or 0)
    try:
        a = types.SimpleNamespace(root=str(root), results=str(root.parent.parent / "results" / "final"), parallel=2,
                                  port_base=6100, wandb=None, push=False, push_every_h=6.0, branch="unused")
        with contextlib.redirect_stdout(io.StringIO()):
            ss.cmd_all(a)
    finally:
        _patch(**old)
    names = [(c[0], c[1]) for c in calls]
    assert ("rulercheck", ["F0_smoke"]) not in names                 # no shorter-ruler check in this suite
    runs = [c for c in calls if c[0] == "run"]
    assert [r[1] for r in runs] == [["F0_smoke"], ["F1_hard"], ["F1_hard"], ["F2_standard"]]   # F1 retried once
    st = json.loads((root / "RUN_STATUS.json").read_text())
    assert "retry F1_hard" in st["decisions"] and "as configured" in st["decisions"]["ruler"]
    assert "disk F1_hard" in st["decisions"] and "disk F2_standard" in st["decisions"]
    assert not pushed and not any("still crashed" in w for w in st["warnings"])


def test_unattended_wrapper_never_pushes():
    src = (REPO / "tools" / "run_final_unattended.sh").read_text(encoding="utf-8")
    assert "nohup setsid" in src and "--suite final all --parallel 2" in src
    assert "--push" not in src.replace("never --push", "") and "results/final" in src


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"{len(tests)} tests passed")
