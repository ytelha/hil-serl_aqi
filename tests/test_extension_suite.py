"""Tests for phase 2 of the extension study (examples/campaigns/extension/, `synthetic_suite.py --suite extension`,
franka_sim/envs/task_knobs.py, docs/extension_study.md): the candidate-task configs, the task knobs end to end, the
per-tier skill and quality_gamma flags, the screen and pilot rules, pruning, and the unattended flow.

    python tests/test_extension_suite.py        (numpy only; no GPU, no simulator)
"""
import contextlib
import importlib.util
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for _p in ("serl_launcher", "examples", "franka_sim"):
    sys.path.insert(0, str(REPO / _p))

import run_campaign as rc                      # noqa: E402
import synthetic_suite as ss                   # noqa: E402

EXT = REPO / "examples" / "campaigns" / "extension"
FINAL = REPO / "examples" / "campaigns" / "final"
ss.configure("extension")
PILOTS = ["P1_hex", "P2_tight", "P3_wide", "P4_hexwide"]
SMOKES = ["S1_hex", "S2_tight", "S3_wide", "S4_hexwide"]
KNOBS = {"P1_hex": ("hex", {}), "P2_tight": ("cuboid", {"success_lateral": 0.006}),
         "P3_wide": ("cuboid", {"random_xy": 0.025, "random_rz": 0.2}),
         "P4_hexwide": ("hex", {"random_xy": 0.025, "random_rz": 0.2})}


def _parse(path, extra=()):
    argv = ["--campaign", "unused", "--config", str(path), *extra]
    parser = rc.build_arg_parser()
    args = parser.parse_args(argv)
    args.ops_per_tier_map = rc.apply_config(args, rc.load_config(str(path)), rc._explicit_dests(parser, argv)) or None
    rc.validate_arms(args, args.methods.split(","))
    return args


def _plan(path, extra=()):
    d = Path(tempfile.mkdtemp()) / Path(path).stem
    with contextlib.redirect_stdout(io.StringIO()):
        rc.main(["--campaign", str(d), "--config", str(path), "--plan", *extra])
    return json.loads((d / "campaign.json").read_text())


def _cmds(args, arm, tier="intermediate", op="int00", seed=0):
    ck = Path("runs/x/checkpoints/c")
    return ([str(c) for c in rc._learner_cmd(args, ck, Path("d.pkl"), arm, seed, "r")],
            [str(c) for c in rc._actor_cmd(args, ck, arm, tier, op, Path("t.json"), seed, "r")])


def _patch(**kw):
    old = {k: getattr(ss, k) for k in kw}
    for k, v in kw.items():
        setattr(ss, k, v)
    return old


# ---------------------------------------------------------------- suite and configs
def test_suite_groups_and_sizes():
    assert ss.SUITE == "extension" and ss.ORDER == PILOTS + SMOKES
    assert ss.resolve(["pilot"]) == PILOTS and ss.resolve(["smoke"]) == SMOKES
    assert [ss.planned_cells(n) for n in PILOTS] == [12] * 4 and [ss.planned_cells(n) for n in SMOKES] == [8] * 4
    assert ss.EXT_PILOT == dict(zip(ss.EXT_CANDIDATES, PILOTS)) and ss.EXT_SMOKE == dict(zip(ss.EXT_CANDIDATES, SMOKES))
    assert "tests/test_extension_suite.py" in ss.TESTS and ss.SUITES["extension"]["sweep_shapes"] == "cuboid,hex"


def test_pilots_careful_three_arms_and_their_task():
    for n in PILOTS:
        args = _parse(EXT / f"{n}.yaml")
        assert args.methods.split(",") == ["hilserl", "keepall", "bc"], n
        m = _plan(EXT / f"{n}.yaml")
        assert not m.get("synthetic_inconsistent"), n
        assert {k: v for k, v in m["operators_per_tier"].items() if v} == {"intermediate": 1, "novice": 1}, n
        peg, knobs = KNOBS[n]
        assert m["peg_shape"] == peg and m.get("sim_task_knobs", {}) == knobs, (n, m.get("sim_task_knobs"))
        assert m["synthetic_rotation_control"] and m["utd_ratio"] == 1.0, n
        assert int(args.demos) == 6, n


def test_smokes_careless_and_every_new_arm_type():
    want = {"bc-discard", "bc-reweight", "bc-outcome", "bc-perfect", "asks-15-bc", "aqi", "asks-30", "e2hil"}
    for n in SMOKES:
        args = _parse(EXT / f"{n}.yaml")
        assert set(args.methods.split(",")) == want, n
        m = _plan(EXT / f"{n}.yaml")
        assert m["synthetic_inconsistent"] == {"bad_share": 0.4, "persistence": 0.7}, n
        assert {k: v for k, v in m["operators_per_tier"].items() if v} == {"intermediate": 1}, n
        assert m["steps"] == 8000, n


def test_imitation_and_filter_flags_of_the_new_arms():
    args = _parse(EXT / "S1_hex.yaml")
    exp = {"bc-discard": ("0.1", "0.0", "score_filter"), "bc-reweight": ("0.1", None, "none"),
           "bc-outcome": ("0.1", "0.0", "outcome_filter"), "bc-perfect": ("0.1", "0.0", "oracle_filter"),
           "asks-15-bc": ("0.1", "0.0", "none"), "aqi": ("0.1", "0.0", "score_filter"), "asks-30": ("0.0", None, "none")}
    for arm, (bc, gamma, filt) in exp.items():
        for cmd in _cmds(args, arm):
            assert cmd[cmd.index("--bc_loss_weight") + 1] == bc, (arm, cmd)
            assert (cmd[cmd.index("--quality_gamma") + 1] if "--quality_gamma" in cmd else None) == gamma, arm
            assert cmd[cmd.index("--quality_filters") + 1] == filt, arm
    l15, _ = _cmds(args, "asks-15-bc")
    l30, _ = _cmds(args, "asks-30")
    assert l15[l15.index("--query_target_rate") + 1] == "0.15" and l30[l30.index("--query_target_rate") + 1] == "0.3"


def test_quality_gamma_is_a_trainer_flag_and_absent_from_older_configs():
    text = (REPO / "examples" / "train_aqi_rlpd.py").read_text(encoding="utf-8")
    assert 'flags.DEFINE_float("quality_gamma"' in text
    assert "quality_gamma" in rc.ARM_KEYS and rc._AQI_FLAG_SPEC[-1] == ("quality_gamma", "val")
    old = _parse(FINAL / "F2_standard.yaml")
    for arm in old.methods.split(","):
        for cmd in _cmds(old, arm):
            assert "--quality_gamma" not in cmd, arm


# ---------------------------------------------------------------- task knobs and skills
def _task_knobs_module():
    spec = importlib.util.spec_from_file_location(
        "task_knobs", REPO / "franka_sim" / "franka_sim" / "envs" / "task_knobs.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_task_knobs_reach_every_process_and_default_to_the_env():
    tk = _task_knobs_module()
    env = rc._sim_env(_parse(EXT / "P3_wide.yaml"))
    assert env == {"SIM_RPY_BOX": "0.1,0.1,0.4", "SIM_SUCCESS_LATERAL": "", "SIM_RANDOM_XY": "0.025",
                   "SIM_RANDOM_RZ": "0.2"}, env
    assert tk.task_overrides_from_env(env) == {"random_xy_range": 0.025, "random_rz_range": 0.2}
    env2 = rc._sim_env(_parse(EXT / "P2_tight.yaml"))
    assert tk.task_overrides_from_env(env2) == {"success_lateral": 0.006}
    old = rc._sim_env(_parse(FINAL / "F2_standard.yaml"))           # the final study: all three empty
    assert old["SIM_SUCCESS_LATERAL"] == old["SIM_RANDOM_XY"] == old["SIM_RANDOM_RZ"] == ""
    assert tk.task_overrides_from_env(old) == {} and tk.task_overrides_from_env({}) == {}
    for bad in ("abc", "0", "-0.01"):
        try:
            tk.task_overrides_from_env({"SIM_RANDOM_XY": bad})
            raise AssertionError(bad)
        except ValueError:
            pass
    wrapper = (REPO / "examples" / "experiments" / "sim_peg_insertion" / "wrapper.py").read_text(encoding="utf-8")
    assert "task_overrides_from_env" in wrapper and "kwargs.setdefault(_k, _v)" in wrapper


def test_screen_environment_matches_the_pilot_task():
    for c, n in zip(ss.EXT_CANDIDATES, PILOTS):
        env, shape = ss.ext_task_env(n)
        args = _parse(EXT / f"{n}.yaml")
        run_env = rc._sim_env(args)
        assert shape == KNOBS[n][0] and env["SIM_RPY_BOX"] == "" and env["JAX_PLATFORMS"] == "cpu"
        for k in ("SIM_SUCCESS_LATERAL", "SIM_RANDOM_XY", "SIM_RANDOM_RZ"):
            assert env[k] == run_env[k], (n, k)


def test_reset_yaw_must_fit_the_orientation_box():
    rc._validate_rpy_box(_parse(EXT / "P3_wide.yaml"))              # 0.2 rad within the 0.4 box
    try:
        rc._validate_rpy_box(_parse(EXT / "P3_wide.yaml", ["--sim_random_rz", "0.5"]))
        raise AssertionError("a reset yaw beyond the box must be refused")
    except SystemExit:
        pass


def test_novice_skill_from_the_command_line():
    args = _parse(EXT / "P1_hex.yaml", ["--tier_s_exec", "novice=0.15", "--tier_s_mon", "novice=0.15"])
    assert args.tier_s_exec == {"novice": 0.15} and args.tier_s_mon == {"novice": 0.15}
    assert _parse(EXT / "P1_hex.yaml").tier_s_exec == {"novice": 0.3}             # the YAML fallback
    for s in ("0", "0.15"):
        d = Path(tempfile.mkdtemp()) / "P1_hex"
        with contextlib.redirect_stdout(io.StringIO()):
            rc.main(["--campaign", str(d), "--config", str(EXT / "P1_hex.yaml"), "--plan",
                     "--tier_s_exec", f"novice={s}", "--tier_s_mon", f"novice={s}"])
        th = json.loads((d / "operators" / "nov00.json").read_text())
        assert abs(float(th["s_exec"]) - float(s)) < 1e-9 and abs(float(th["s_mon"]) - float(s)) < 1e-9, th


# ---------------------------------------------------------------- the rules
def _rows(es, er, ec, nov):
    rows = [dict(tier="expert", scenario="solo", condition="attentive", rotation="on", success=es),
            dict(tier="expert", scenario="rescue", condition="attentive", rotation="on", success=er),
            dict(tier="expert", scenario="solo", condition="inattentive", rotation="on", success=ec)]
    rows += [dict(tier=ss._skill_label(s), scenario="solo", condition="attentive", rotation="on", success=v)
             for s, v in nov.items()]
    return rows


def test_screen_rule():
    v = ss.screen_verdict(_rows(0.97, 0.9, 0.3, {0.3: 0.9, 0.15: 0.6, 0.0: 0.45}))
    assert v["passes"] and v["novice_skill"] == 0.15, v                 # 0.6 is closest to 0.55
    v = ss.screen_verdict(_rows(0.97, 0.9, 0.3, {0.3: 0.5, 0.15: 0.6, 0.0: 0.2}))
    assert v["novice_skill"] == 0.3, v                                   # tie 0.05 / 0.05: the earlier skill
    assert not ss.screen_verdict(_rows(0.80, 0.9, 0.3, {0.3: 0.5, 0.15: 0.5, 0.0: 0.5}))["passes"]   # expert weak
    assert not ss.screen_verdict(_rows(0.97, 0.7, 0.3, {0.3: 0.5, 0.15: 0.5, 0.0: 0.5}))["passes"]   # no rescue
    assert not ss.screen_verdict(_rows(0.97, 0.9, 0.7, {0.3: 0.5, 0.15: 0.5, 0.0: 0.5}))["passes"]   # careless ok
    v = ss.screen_verdict(_rows(0.97, 0.9, 0.3, {0.3: 0.95, 0.15: 0.9, 0.0: 0.8}))
    assert not v["passes"] and v["novice_skill"] is None                 # the tiers do not differ
    assert not ss.screen_verdict([])["passes"]


def _cell(camp, cid, curve, final, pre=5000, steps_on_disk=None):
    """A scored cell: cell.json with the per-checkpoint ruler, a manifest with the warm start, checkpoints."""
    ev = Path(camp) / "eval_runs" / cid
    ck = Path(camp) / "checkpoints" / cid
    ev.mkdir(parents=True, exist_ok=True)
    ck.mkdir(parents=True, exist_ok=True)
    cps = [{"step": s + pre, "sr": v, "n_trials": 9} for s, v in curve]
    last = cps[-1]["step"]
    (ev / "cell.json").write_text(json.dumps({"sr_per_checkpoint": cps, "best_window_sr": final,
                                              "best_window_steps": [c["step"] for c in cps[-3:]],
                                              "checkpoint_step": last, "outcomes": {"sr": final}}))
    rep = ["python", "train_aqi_rlpd.py"] + (["--demo_pretrain", "--demo_pretrain_steps", str(pre)] if pre else [])
    (ck / "manifest.json").write_text(json.dumps({"arm": {"name": cid.split("_")[0], "overrides": {}},
                                                  "reproduce": rep}))
    (ck / ".done").write_text("{}")
    for s in (steps_on_disk or [c["step"] for c in cps]):
        d = ck / f"checkpoint_{s}"
        d.mkdir(exist_ok=True)
        (d / "w.msgpack").write_bytes(b"x" * 1000)


def _pilot(root, name, int_at, nov_at, final):
    """keepall cells of a pilot: success rises linearly to int_at / nov_at at 8 000 env steps."""
    camp = Path(root) / name
    for op, at in (("int00", int_at), ("nov00", nov_at)):
        for seed in (0, 1):
            curve = [(s, min(1.0, at * s / 8000.0)) for s in range(2000, 16001, 2000)]
            _cell(camp, f"keepall_{op}_s{seed}", curve, final)
    return camp


def test_budget_success_counts_environment_steps():
    root = Path(tempfile.mkdtemp())
    camp = _pilot(root, "P3_wide", 0.5, 0.2, 0.9)
    cv, fin = ss.cell_curve(camp, "keepall_int00_s0")
    assert cv[0] == (2000, 0.125) and fin == 0.9                         # learner step 7 000 - 5 000 warm start
    assert ss.budget_success(cv, 8000) == 0.5 and ss.budget_success(cv, 1000) == 0.0
    assert ss.budget_success(cv, 99999) == cv[-1][1]                     # a stopped run keeps its last score


def test_pilot_rule_choice_fallback_and_budget():
    root = Path(tempfile.mkdtemp())
    _pilot(root, "P1_hex", 0.9, 0.5, 1.0)          # too easy at 8 000
    _pilot(root, "P3_wide", 0.5, 0.2, 0.9)         # in the band, gap 0.3
    _pilot(root, "P4_hexwide", 0.4, 0.3, 0.85)     # in the band, gap 0.1
    ver = ss.pilot_verdicts(root)
    assert set(ver) == {"hex", "wide", "hexwide"} and not ver["hex"]["qualifies"]
    assert ver["wide"]["qualifies"] and ver["hexwide"]["qualifies"] and abs(ver["wide"]["gap"] - 0.3) < 1e-9
    assert ss.choose_task(ver)[0] == "wide" and ss.choose_task(ver)[2] is None
    b, at = ss.choose_budget(root / "P3_wide")
    assert b == 8000 and at == 0.5
    root2 = Path(tempfile.mkdtemp())
    _pilot(root2, "P1_hex", 0.1, 0.0, 0.6)         # none qualifies: hardest learnable one, cap 40 000
    _pilot(root2, "P2_tight", 0.2, 0.1, 0.4)       # not learnable (final < 0.5)
    c, how, cap = ss.choose_task(ss.pilot_verdicts(root2))
    assert c == "hex" and cap == 40000 and "fallback" in how
    root3 = Path(tempfile.mkdtemp())
    _pilot(root3, "P1_hex", 0.0, 0.0, 0.2)
    assert ss.choose_task(ss.pilot_verdicts(root3))[0] is None


def test_prune_keeps_window_last_and_newest_and_refuses_other_roots():
    root = Path(tempfile.mkdtemp())
    (root / ss.PRUNE_MARKER).write_text("test")
    camp = root / "P3_wide"
    curve = [(s, 0.5) for s in range(2000, 16001, 2000)]                 # learner steps 7 000 ... 21 000
    _cell(camp, "keepall_int00_s0", curve, 0.5, steps_on_disk=list(range(7000, 21001, 2000)) + [22000])
    _cell(camp, "keepall_int00_s1", curve, 0.5)
    (camp / "checkpoints" / "keepall_int00_s1" / ".done").unlink()        # unfinished: untouched
    n, d, freed = ss.prune_campaign(camp, log=None)
    left = sorted(int(p.name.split("_")[1]) for p in (camp / "checkpoints" / "keepall_int00_s0").glob("checkpoint_*"))
    assert left == [17000, 19000, 21000, 22000] and n == 1 and d == 5 and freed == 5000, (left, n, d, freed)
    assert len(list((camp / "checkpoints" / "keepall_int00_s1").glob("checkpoint_*"))) == 8
    assert (camp / "checkpoints" / "keepall_int00_s0" / "manifest.json").exists()
    assert ss.prunable_root(REPO / "runs" / "extension") and ss.prunable_root(REPO / "runs" / "final_twins")
    assert not ss.prunable_root(REPO / "runs" / "final") and not ss.prunable_root(Path(tempfile.mkdtemp()))
    with contextlib.redirect_stdout(io.StringIO()):
        assert ss.cmd_prune(ss._ns(type("A", (), {"root": str(REPO / "runs" / "final"), "results": None})(),
                                   names=[])) == 2


# ---------------------------------------------------------------- the unattended flow
def test_all_screens_pilots_decides_smokes_and_never_pushes():
    root = Path(tempfile.mkdtemp())
    (root / ss.PRUNE_MARKER).write_text("test")
    calls = []

    def fake_screen(a):
        calls.append(("screen",))
        sd = Path(a.root) / "screen"
        sd.mkdir(parents=True, exist_ok=True)
        for c in ss.EXT_CANDIDATES:
            (sd / f"{c}.json").write_text("{}")
        cands = {c: {"passes": c in ("hex", "wide"), "novice_skill": {"hex": 0.15, "wide": 0.3}.get(c),
                     "knobs": {}, "peg_shape": "cuboid"} for c in ss.EXT_CANDIDATES}
        (sd / "screen_decision.json").write_text(json.dumps({"passing": ["hex", "wide"], "candidates": cands}))
        return 0

    def fake_stage0(a):
        calls.append(("stage0",))
        (Path(a.root) / "stage0").mkdir(parents=True, exist_ok=True)
        (Path(a.root) / "stage0" / "limits.json").write_text(json.dumps(
            {"limits": {"cuboid": {"tilt_rad": 0.31, "yaw_rad": 0.14}, "hex": {"tilt_rad": 0.3, "yaw_rad": 0.1}}}))
        return 0

    def fake_run(a):
        n = a.names[0]
        calls.append(("run", n, list(a.extra), a.parallel))
        if n == "P1_hex":
            _pilot(a.root, n, 0.9, 0.5, 1.0)
        elif n == "P3_wide":
            _pilot(a.root, n, 0.5, 0.2, 0.9)
        else:
            _cell(Path(a.root) / n, "aqi_int00_s0", [(2000, 0.2)], 0.2)
        return 0

    old = _patch(cmd_check=lambda a: 0, cmd_stage0=fake_stage0, cmd_screen=fake_screen, cmd_run=fake_run,
                 cmd_audit=lambda a: calls.append(("audit", list(a.names))) or 0,
                 cmd_report=lambda a: calls.append(("report",)) or 0, cmd_plots=lambda a: 0,
                 export_results=lambda a, quiet=False: 0, wandb_logged_in=lambda *x, **k: False,
                 push_results=lambda *x, **k: (_ for _ in ()).throw(AssertionError("never push")))
    try:
        args = type("A", (), {})()
        args.root, args.results, args.wandb, args.parallel, args.port_base, args.push_every_h = (
            str(root), None, None, 2, 6100, 6.0)
        with contextlib.redirect_stdout(io.StringIO()):
            ss.cmd_extension_all(args)
            # a second start repeats neither stage 0 nor the screen
            ss.cmd_extension_all(args)
    finally:
        _patch(**old)
    first = calls[:calls.index(("report",)) + 1]
    runs = [c for c in first if c[0] == "run"]
    assert [c[1] for c in runs] == ["P1_hex", "P3_wide", "S3_wide"], runs
    assert runs[0][2][:4] == ["--tier_s_exec", "novice=0.15", "--tier_s_mon", "novice=0.15"] and runs[0][3] == 2
    assert runs[1][2][:4] == ["--tier_s_exec", "novice=0.3", "--tier_s_mon", "novice=0.3"]
    assert runs[2][2] == [] and ("audit", ["S3_wide"]) in first
    dec = json.loads((root / "decision.json").read_text())
    assert dec["chosen"] == "wide" and dec["novice_skill"] == 0.3 and dec["budget_B_env_steps"] == 8000
    st = json.loads((root / "RUN_STATUS.json").read_text())
    assert st["state"].startswith("finished") and "task of the main study" in st["decisions"]
    assert calls.count(("stage0",)) == 1 and calls.count(("screen",)) == 1          # not repeated on restart
    left = list((root / "P3_wide" / "checkpoints" / "keepall_int00_s0").glob("checkpoint_*"))
    assert len(left) == 3                                                             # pruned after scoring
    shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"{len(tests)} tests passed")
