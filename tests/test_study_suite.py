"""Tests for the comprehensive study (examples/campaigns/study/, `synthetic_suite.py --suite study`,
docs/comprehensive_study.md): every config passes the campaign runner, every arm's commands use only flags
the target script defines, the study's design reaches the commands (critics, asking rates, the careless
operator at 20 % in every arm of B incl. HiRIL, no outcome filter, no imitation, the short warm-up, the
shorter ruler in A and B only), and the driver's side-by-side runs, W&B check and plots. No JAX, no MuJoCo,
no training.

Run from the repo root with
    python -m pytest tests/test_study_suite.py -q
    python tests/test_study_suite.py
"""
import contextlib
import io
import json
import re
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for _p in ("serl_launcher", "examples"):
    sys.path.insert(0, str(REPO / _p))

import run_campaign as rc                      # noqa: E402
import synthetic_suite as ss                   # noqa: E402

STUDY = REPO / "examples" / "campaigns" / "study"
ss.configure("study")


def _parse(name, extra=()):
    cfg_p = str(STUDY / f"{name}.yaml")
    argv = ["--campaign", "unused", "--config", cfg_p, *extra]
    parser = rc.build_arg_parser()
    args = parser.parse_args(argv)
    args.ops_per_tier_map = rc.apply_config(args, rc.load_config(cfg_p), rc._explicit_dests(parser, argv)) or None
    rc.validate_arms(args, args.methods.split(","))
    return args


def _defined(script):
    text = (REPO / "examples" / script).read_text(encoding="utf-8")
    names = set(re.findall(r'flags\.DEFINE_\w+\(\s*"(\w+)"', text))
    if "for _name, _cast in CLI_FIELDS.items()" in text:
        from serl_launcher.hiril.config import CLI_FIELDS
        names |= {f"hiril_{n}" for n in CLI_FIELDS}
    return names


def _flags(cmd):
    return {t[2:].split("=", 1)[0] for t in cmd if isinstance(t, str) and t.startswith("--")}


def _cmds(args, arm):
    ck = Path("runs/x/checkpoints/c")
    return (rc._learner_cmd(args, ck, Path("d.pkl"), arm, 0, "r"),
            rc._actor_cmd(args, ck, arm, "expert", "exp00", Path("t.json"), 0, "r"))


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


def test_suite_groups_and_order():
    assert ss.resolve(["smoke"]) == ["S0_smoke_A", "S0_smoke_B"]
    assert ss.resolve(["main"]) == ["A_consistent", "B_quality"]
    assert ss.ORDER == ["S0_smoke_A", "S0_smoke_B", "A_consistent", "B_quality"]
    assert set(ss.REF_ARMS) == {"hilserl", "ens-k8", "keepall"}
    assert "tests/test_study_suite.py" in ss.TESTS


def test_every_config_passes_the_campaign_runner():
    want = {"S0_smoke_A": ({"expert": 1}, False, 3), "S0_smoke_B": ({"expert": 1}, True, 5),
            "A_consistent": ({"expert": 3, "intermediate": 3}, False, 8), "B_quality": ({"expert": 3}, True, 8)}
    for n, (ops, inc, n_arms) in want.items():
        d = Path(tempfile.mkdtemp()) / n
        with contextlib.redirect_stdout(io.StringIO()):
            rc.main(["--campaign", str(d), "--config", str(ss.config_path(n)), "--plan"])
        m = json.loads((d / "campaign.json").read_text())
        assert {k: v for k, v in m["operators_per_tier"].items() if v} == ops, n
        assert m["synthetic_rotation_control"] is True and m["utd_ratio"] == 1.0 and m["log_takeover_channels"], n
        assert m["sim_rpy_box"] == [0.1, 0.1, 0.4] and "synthetic_orientation_monitor" in m, n
        assert ("synthetic_inconsistent" in m) == inc, n
        if inc:
            assert m["synthetic_inconsistent"] == {"bad_share": 0.2, "persistence": 0.7}, n
        assert len(m["methods"]) == n_arms, n
        assert (m.get("converge_mode") or {}).get("mode") == "eval", n


def test_every_arm_uses_only_defined_flags():
    known = {"train_aqi_rlpd.py": _defined("train_aqi_rlpd.py"), "train_hiril.py": _defined("train_hiril.py")}
    for n in ss.ORDER:
        args = _parse(n)
        for arm in args.methods.split(","):
            for cmd in _cmds(args, arm):
                defined = known[cmd[0]]
                used = _flags(cmd)
                neg = {u for u in used if u.startswith("no") and u[2:] in defined}
                unknown = used - defined - neg
                assert not unknown, (n, arm, cmd[0], unknown)


def test_study_A_critics_and_asking():
    args = _parse("A_consistent")
    assert args.methods.split(",") == ["hilserl", "ens-k4", "ens-k8", "ens-k16", "asks-40", "asks-55", "asks-70",
                                       "random-55"]
    s = {arm: [" ".join(c) for c in _cmds(args, arm)] for arm in args.methods.split(",")}
    for k in (4, 8, 16):
        assert f"--aqi_mode logged" in s[f"ens-k{k}"][0] and f"--critic_ensemble_size {k} " in s[f"ens-k{k}"][0]
    for r in (40, 55, 70):
        lc = s[f"asks-{r}"][0]
        assert "--aqi_mode active" in lc and f"--query_target_rate {r / 100:g}" in lc
        assert "--query_cooldown_steps 5" in lc and "--critic_ensemble_size 8 " in lc
    rl = s["random-55"][0]
    assert "--trigger_terms none" in rl and "--random_request_rate 0.25" in rl
    assert "--nodemo_pretrain" in s["hilserl"][0] and "--learning_rate" not in s["hilserl"][0]
    for arm, (lc, ac) in s.items():
        assert "--synthetic_inconsistent" not in ac, arm          # study A: the consistent operator
        assert "--utd_ratio 1" in lc and "--utd_ratio 1" in ac, arm
        if arm != "hilserl":
            assert "--demo_pretrain --demo_pretrain_steps 5000" in lc and "--learning_rate 0.00015" in lc, arm


def test_study_B_careless_expert_and_the_arms():
    args = _parse("B_quality")
    arms = args.methods.split(",")
    assert arms == ["hilserl", "e2hil", "hiril", "keepall", "beh-discard", "beh-reweight", "oracle-reweight",
                    "perfect-filter"]
    s = {arm: [" ".join(c) for c in _cmds(args, arm)] for arm in arms}
    for arm, (lc, ac) in s.items():
        assert "--synthetic_inconsistent --synthetic_bad_share 0.2 --synthetic_bad_persistence 0.7" in ac, arm
        assert "--synthetic_inconsistent" not in lc, arm
        assert "outcome_filter" not in lc + ac, arm                # no outcome filter anywhere
        if "--bc_loss_weight" in lc:
            assert "--bc_loss_weight 0.0" in lc, arm               # no imitation loss anywhere
    assert s["hiril"][0].startswith("train_hiril.py") and s["hiril"][1].startswith("train_hiril.py")
    assert "--hiril_recalib_period 1000" in s["hiril"][0]
    for arm in ("hilserl", "e2hil", "hiril"):
        assert "--nodemo_pretrain" in s[arm][0], arm                # plain, from scratch, as in the papers
    assert "--aqi_mode e2hil" in s["e2hil"][0]
    q = {a: s[a][0] for a in ("keepall", "beh-discard", "beh-reweight", "oracle-reweight", "perfect-filter")}
    for arm, lc in q.items():
        assert "--aqi_mode quality" in lc and "--quality_warmup_steps 1000" in lc and "--demo_pretrain " in lc, arm
    assert "--quality_filters none" in q["keepall"] and "--per_eta 0.0" in q["keepall"]
    assert "--quality_filters score_filter" in q["beh-discard"] and "--quality_drop_fraction 0.2" in q["beh-discard"]
    assert "--quality_score_mode behavioral" in q["beh-reweight"] and "--per_eta 1.0" in q["beh-reweight"]
    assert "--quality_score_mode oracle" in q["oracle-reweight"] and "--quality_cf_diag" in q["oracle-reweight"]
    assert "--quality_filters oracle_filter" in q["perfect-filter"]


def test_shorter_ruler_in_A_and_B_full_ruler_in_the_smoke():
    for n, short in (("A_consistent", True), ("B_quality", True), ("S0_smoke_A", False), ("S0_smoke_B", False)):
        args = _parse(n)
        for arm in args.methods.split(","):
            cmd = _eval_cmd(args, arm)
            assert ("--select_newest 8" in cmd) == short, (n, arm)
            pre = "--select_min_step" in cmd
            if short:   # pre-training is skipped where an arm has one (5 000 steps); the 2-critic arms have none
                assert pre == (arm not in ("hilserl", "e2hil", "hiril")), (n, arm, cmd)
            else:
                assert not pre, (n, arm)
            if arm == "hiril":
                assert "--agent_type hiril" in cmd


def test_parallel_commands_split_ports_shards_and_memory():
    prep, shards, ana = ss.parallel_cmds(["python", "run_campaign.py", "--config", "c.yaml"], 2, 6100)
    assert prep[-2:] == ["--stages", "sample,demos"] and ana[-2:] == ["--stages", "analyze"]
    assert [e["AQI_TRAINER_PORT"] for _c, e in shards] == ["6100", "6110"]
    for i, (c, _e) in enumerate(shards):
        s = " ".join(c)
        assert f"--shard {i}/2" in s and "--stages train,eval" in s and "--force" in s
        assert "--gpu_mem_learner 0.30" in s and "--gpu_mem_actor 0.10" in s


def test_run_dry_prints_the_shards():
    root = Path(tempfile.mkdtemp())
    (root / "stage0").mkdir()
    (root / "stage0" / "limits.json").write_text(json.dumps({"limits": {"cuboid": {"tilt_rad": 0.384,
                                                                                  "yaw_rad": 0.1396}}}))
    a = type("A", (), dict(names=["A_consistent"], root=str(root), wandb=None, extra=[], dry=True, keep_going=False,
                           default_limits=False, without_active=False, allow_orig=False, score_as_is=False,
                           parallel=2, port_base=6100))()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert ss.cmd_run(a) == 0
    out = buf.getvalue()
    assert "--synthetic_orientation_tilt_rad 0.384" in out and "AQI_TRAINER_PORT=6110" in out


def test_wandb_login_check():
    home = Path(tempfile.mkdtemp())
    assert not ss.wandb_logged_in(env={}, home=home)
    assert ss.wandb_logged_in(env={"WANDB_API_KEY": "x"}, home=home)
    (home / ".netrc").write_text("machine api.wandb.ai\n  login user\n  password secret\n")
    assert ss.wandb_logged_in(env={}, home=home)
    assert set(ss.configs_logging_online()) == set(ss.ORDER)       # every study config logs online


def _cell(arm, op, seed, sr, steps, share, fus=4000):
    return {"cell": f"{arm}_{op}_s{seed}", "arm": arm, "operator": op, "seed": seed, "ruler_sr": sr,
            "env_steps": steps, "operator_share": share, "first_unaided_success": fus, "actor_minutes": 25.0}


def test_plots_on_fake_cells():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("SKIP (no matplotlib)")
        return
    out = Path(tempfile.mkdtemp())
    a_arms = ["hilserl", "ens-k4", "ens-k8", "ens-k16", "asks-40", "asks-55", "asks-70", "random-55"]
    cells = [_cell(a, op, s, 0.8 + 0.01 * i, 9000 + 100 * i, 0.3 + 0.02 * i)
             for i, a in enumerate(a_arms) for op in ("exp00", "int00") for s in (0, 1)]
    made = ss.plot_campaign("A_consistent", cells, a_arms, {"asks-55": 0.01}, None, out, plt)
    assert {p.name for p in made} == {"A_consistent_arms.png", "A_consistent_critics.png", "A_consistent_asking.png"}
    audit = {"n_segments": 100, "n_runs": 9, "scores": {
        "B = z(dV)-z(t)-z(F)": {"auroc_within": {"all": 0.66}}, "-t_int": {"auroc_within": {"all": 0.65}},
        "a_hat_mean": {"auroc_within": {"all": 0.49}}}}
    b_arms = ["hilserl", "keepall", "beh-discard"]
    made = ss.plot_campaign("B_quality", [_cell(a, "exp00", 0, 0.9, 9000, 0.4) for a in b_arms], b_arms, {},
                            audit, out, plt)
    assert {p.name for p in made} == {"B_quality_arms.png", "B_quality_scores.png"}
    assert all(p.stat().st_size > 1000 for p in made)


def test_compact_log_keeps_the_last_progress_update_and_head_and_tail():
    text = "cmd line\n" + "\r".join(f"{i}/100 [00:{i:02d}<]" for i in range(100)) + "\nend\n"
    out = ss.compact_log(text)
    assert out.splitlines() == ["cmd line", "99/100 [00:99<]", "end"]
    long = "\n".join(f"line {i}" for i in range(ss.LOG_HEAD + ss.LOG_TAIL + 500))
    lines = ss.compact_log(long).splitlines()
    assert lines[0] == "line 0" and lines[-1] == f"line {ss.LOG_HEAD + ss.LOG_TAIL + 499}"
    assert len(lines) == ss.LOG_HEAD + ss.LOG_TAIL + 1 and "left out by the export" in lines[ss.LOG_HEAD]


def _fake_run_tree(root):
    camp = root / "B_quality"
    for rel, body in {
            "machine.json": json.dumps({"machine_id": "box / RTX 5090"}),
            "report.md": "# report", "plots/B_quality_arms.png": "png",
            "stage0/tests.json": "{}",
            "B_quality/campaign.json": "{}", "B_quality/report/dependency.json": "{}",
            "B_quality/report/plots/sr_by_tier.png": "png",
            "B_quality/checkpoints/keepall_exp00_s0/.done": json.dumps({"method": "keepall"}),
            "B_quality/checkpoints/keepall_exp00_s0/episodes.jsonl": "{}\n",
            "B_quality/checkpoints/keepall_exp00_s0/checkpoint_9000/manifest.ocdbt": "x" * 5000,
            "B_quality/checkpoints/keepall_exp00_s0/wandb/run-1/files/output.log": "x" * 5000,
            "B_quality/eval_runs/keepall_exp00_s0/cell.json": json.dumps({"outcomes": {"sr": 1.0}}),
            "B_quality/eval_runs/keepall_exp00_s0/trials.jsonl": "{}\n",
            "B_quality/aqi_logs/keepall_exp00_s0_actor/segments.jsonl": "{}\n",
            "B_quality/aqi_logs/keepall_exp00_s0_actor/steps.jsonl": "x" * 5000,
            "B_quality/demos/exp00_s0/demos.pkl": "x" * 5000,
            "B_quality/logs/keepall_exp00_s0.actor.log": "start\n" + "\r".join(["1/9", "9/9 [12:34<]"]) + "\n"}.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    (camp / "checkpoints" / "keepall_exp00_s0" / ".done").write_text(json.dumps({"method": "keepall"}))
    return camp


def test_export_keeps_the_layout_without_heavy_files():
    root = Path(tempfile.mkdtemp()) / "runs" / "study"
    _fake_run_tree(root)
    out = root.parent.parent / "results" / "study"
    a = type("A", (), dict(root=str(root), out=str(out), with_steps=False))()
    with contextlib.redirect_stdout(io.StringIO()):
        assert ss.export_results(a) == 0
    got = {str(p.relative_to(out)).replace("\\", "/") for p in out.rglob("*") if p.is_file()}
    for keep in ("machine.json", "report.md", "plots/B_quality_arms.png", "stage0/tests.json",
                 "B_quality/report/dependency.json", "B_quality/report/plots/sr_by_tier.png",
                 "B_quality/checkpoints/keepall_exp00_s0/.done", "B_quality/checkpoints/keepall_exp00_s0/episodes.jsonl",
                 "B_quality/eval_runs/keepall_exp00_s0/cell.json", "B_quality/eval_runs/keepall_exp00_s0/trials.jsonl",
                 "B_quality/aqi_logs/keepall_exp00_s0_actor/segments.jsonl",
                 "B_quality/logs/keepall_exp00_s0.actor.log", "README.md", "export.json"):
        assert keep in got, keep
    for drop in ("checkpoint_9000", "wandb", "steps.jsonl", "demos.pkl"):
        assert not any(drop in g for g in got), drop
    assert (out / "B_quality/logs/keepall_exp00_s0.actor.log").read_text().splitlines() == ["start", "9/9 [12:34<]"]
    info = json.loads((out / "export.json").read_text())
    assert info["skipped_items"] == 4 and info["campaigns"][0]["campaign"] == "B_quality"
    assert info["campaigns"][0]["cells_done"] == 1 and info["campaigns"][0]["report"]
    # the exported folder re-analyses with the same driver, and is not exported again
    b = type("B", (), dict(root=str(out), no_export=False, dry=False))()
    assert not ss.auto_export(b)
    rows, cells, _m = ss.summarize("B_quality", out / "B_quality")
    assert cells[0]["ruler_sr"] == 1.0 and cells[0]["actor_minutes"] == 12.6
    with contextlib.redirect_stdout(io.StringIO()):
        assert ss.export_results(a) == 0                        # a second export copies nothing new
    assert json.loads((out / "export.json").read_text())["files_written"] == 0


def test_default_results_folder():
    assert ss.default_results(ss.DEFAULT_ROOT) == ss.REPO / "results" / "study"
    other = Path(tempfile.mkdtemp()) / "myruns"
    assert ss.default_results(other) == other.parent / "results" / "myruns"
    assert "results/" in (ss.REPO / ".gitignore").read_text(encoding="utf-8")     # never pushed with the code


def _patch(**fns):
    old = {k: getattr(ss, k) for k in fns}
    for k, v in fns.items():
        setattr(ss, k, v)
    return old


def test_all_runs_every_step_without_stopping_and_decides():
    root = Path(tempfile.mkdtemp()) / "runs" / "study"
    calls = []

    def rec(name, rc=0):
        def f(a):
            calls.append((name, list(getattr(a, "names", []) or []), getattr(a, "parallel", None),
                          list(getattr(a, "extra", []) or []), getattr(a, "wandb", None)))
            if name == "rulercheck":                     # the shorter ruler changes one smoke score
                (root / "S0_smoke_A" / "report").mkdir(parents=True, exist_ok=True)
                (root / "S0_smoke_A" / "report" / "ruler_reselect.json").write_text(json.dumps(
                    {"ens-k16_exp00_s0": {"old": 0.8, "new": 0.7}, "asks-70_exp00_s0": {"old": 1.0, "new": 1.0}}))
            if name == "stage0":
                raise SystemExit(3)                      # a crashing step: recorded, the run goes on
            return rc
        return f

    old = _patch(cmd_check=rec("check", 1), cmd_stage0=rec("stage0"), cmd_run=rec("run"),
                 cmd_rulercheck=rec("rulercheck"), cmd_audit=rec("audit"), cmd_report=rec("report"),
                 cmd_plots=rec("plots"), export_results=lambda *a, **k: 0, wandb_logged_in=lambda *a, **k: False,
                 parallel_decision=lambda r, n, req: (1, "test: not faster side by side"))
    try:
        a = type("A", (), dict(root=str(root), results=str(root.parent / "results"), parallel=2, port_base=6100,
                               wandb=None, push=False, push_every_h=6.0, branch="study-results"))()
        with contextlib.redirect_stdout(io.StringIO()):
            rc = ss.cmd_all(a)
    finally:
        _patch(**old)
    assert rc == 1                                        # warnings: check problems, stage 0 crash
    names = [c[0] for c in calls]
    assert names == ["check", "stage0", "run", "rulercheck", "audit", "report", "run", "run", "audit", "report",
                     "plots"]
    smoke, mainA, mainB = calls[2], calls[6], calls[7]
    assert smoke[1] == ["S0_smoke_A", "S0_smoke_B"] and smoke[2] == 2 and smoke[4] == "offline"
    assert mainA[1] == ["A_consistent"] and mainB[1] == ["B_quality"] and mainA[2] == 1
    assert mainA[3] == ["--no-eval_skip_pretrain", "--eval_newest", "0"] == mainB[3]   # full ruler after a change
    assert calls[4][1] == ["S0_smoke_B"] and calls[8][1] == ["B_quality"]
    st = json.loads((root / "RUN_STATUS.json").read_text())
    assert st["state"] == "finished with warnings"
    assert "FULL ruler" in st["decisions"]["ruler"] and "1 of 2" in st["decisions"]["ruler"]
    assert st["decisions"]["W&B"].startswith("offline") and st["decisions"]["cells side by side"].startswith("1:")
    assert any("stage0" in w for w in st["warnings"]) and (root / "RUN_STATUS.md").exists()


def test_parallel_decision_rules():
    root = Path(tempfile.mkdtemp())
    (root / "S0_smoke_A" / "checkpoints").mkdir(parents=True)
    rows = {"fast": [dict(arm="asks-70", finished=True, env_steps=4000, actor_minutes=4000 / (4.0 * 60)),
                     dict(arm="random-55", finished=True, env_steps=4000, actor_minutes=4000 / (4.2 * 60)),
                     dict(arm="ens-k16", finished=True, env_steps=4000, actor_minutes=100.0)],   # not an 8-critic arm
            "slow": [dict(arm="asks-70", finished=True, env_steps=4000, actor_minutes=4000 / (3.0 * 60))],
            "broken": [dict(arm="asks-70", finished=False, env_steps=0, actor_minutes=None),
                       dict(arm="random-55", finished=False, env_steps=0, actor_minutes=None)]}
    for case, want in (("fast", 2), ("slow", 1), ("broken", 1)):
        old = _patch(summarize=lambda n, camp, _c=case: ([], rows[_c], None))
        try:
            n, why = ss.parallel_decision(root, ["S0_smoke_A"], 2)
        finally:
            _patch(**old)
        assert n == want, (case, why)
    assert ss.parallel_decision(root, ["S0_smoke_A"], 1)[0] == 1


def test_ruler_changes_counts_only_compared_cells():
    root = Path(tempfile.mkdtemp())
    (root / "S0_smoke_B" / "report").mkdir(parents=True)
    (root / "S0_smoke_B" / "report" / "ruler_reselect.json").write_text(json.dumps(
        {"a": {"old": 1.0, "new": 1.0}, "b": {"old": None, "new": 0.5}, "c": {"old": 0.6, "new": 0.6}}))
    assert ss.ruler_changes(root, ["S0_smoke_A", "S0_smoke_B"]) == (2, 0)


def test_push_results_to_a_local_remote():
    import subprocess
    tmp = Path(tempfile.mkdtemp())
    bare = tmp / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)
    res = tmp / "results" / "study"
    res.mkdir(parents=True)
    (res / "README.md").write_text("x")
    assert ss.push_results(res, branch="study-results", remote_url=str(bare), log=lambda m: None) == 0
    (res / "README.md").write_text("y")
    assert ss.push_results(res, branch="study-results", remote_url=str(bare), log=lambda m: None) == 0
    log = subprocess.run(["git", "--git-dir", str(bare), "log", "--oneline", "study-results"], capture_output=True,
                         text=True).stdout.splitlines()
    assert len(log) == 2
    assert ss.push_results(res, remote_url=str(tmp / "nowhere.git"), log=lambda m: None) != 0   # fails, no exception


def test_unattended_wrapper():
    src = (REPO / "tools" / "run_study_unattended.sh").read_text(encoding="utf-8")
    assert "nohup setsid" in src and "systemd-inhibit" in src and "--suite study all --parallel 2" in src
    assert "--no-push" in src and "tools/setup_workstation.sh" in src


def test_hiril_audit_rows_are_left_out_by_mode():
    src = (REPO / "examples" / "inconsistent_operator_audit.py").read_text(encoding="utf-8")
    assert '"--modes"' in src and 'r.get("mode") in keep' in src
    drv = (REPO / "examples" / "synthetic_suite.py").read_text(encoding="utf-8")
    assert '["--modes", "quality"]' in drv


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
