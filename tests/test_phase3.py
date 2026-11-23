"""Tests for phase 3 of the extension study (examples/campaigns/phase3/, synthetic_suite.py --suite phase3,
examples/phase3_analysis.py, docs/extension_phase3.md).

- the careful twin X2 trains exactly like X1 apart from the careless operator; E2HiL's mask-only arm differs from
  E2HiL only in the critic reduction; the probe trains exactly like P1_hex's keepall run apart from its short budget
  and its logging; the fixed-budget flags reach the phase-3 commands and no earlier config;
- the analysis (success at B, AULC, operator steps until the would-be stop, careless share, E2HiL keep fractions,
  exact sign / Wilcoxon tests, Holm) on a small fixture of run folders;
- the suite's input copy with hash check, the probe report, the smoke checks and the pruning of scored runs.

    python tests/test_phase3.py        (numpy only; no GPU, no simulator)
"""
import contextlib
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
import phase3_analysis as pa                   # noqa: E402

CAMPS = REPO / "examples" / "campaigns"
CARELESS = ["--synthetic_inconsistent", "--synthetic_bad_share", "0.4", "--synthetic_bad_persistence", "0.7"]
GUARD = ["--checkpoint_env_every", "--divergence_guard_every", "--metrics_log_every"]


def _parse(rel, extra=()):
    cfg_p = str(CAMPS / rel)
    argv = ["--campaign", "unused", "--config", cfg_p, *extra]
    parser = rc.build_arg_parser()
    args = parser.parse_args(argv)
    args.ops_per_tier_map = rc.apply_config(args, rc.load_config(cfg_p), rc._explicit_dests(parser, argv)) or None
    rc.validate_arms(args, args.methods.split(","))
    return args


def _cmds(args, arm, tier="intermediate", op="int00", seed=0):
    ck = Path("runs/x/checkpoints/c")
    with contextlib.redirect_stdout(io.StringIO()):
        return ([str(c) for c in rc._learner_cmd(args, ck, Path("d.pkl"), arm, seed, "r")],
                [str(c) for c in rc._actor_cmd(args, ck, arm, tier, op, Path("t.json"), seed, "r")])


def _drop(cmd, flags_with_values):
    out, i = [], 0
    while i < len(cmd):
        if cmd[i] in flags_with_values:
            i += 2
            continue
        out.append(cmd[i])
        i += 1
    return out


def _without(cmd, seq):
    i = cmd.index(seq[0])
    assert cmd[i:i + len(seq)] == seq, cmd[i:i + len(seq)]
    return cmd[:i] + cmd[i + len(seq):]


# ---------------------------------------------------------------------------- configs and commands
def test_planned_runs():
    ss.configure("phase3")
    assert [ss.planned_cells(n) for n in ss.ORDER] == [2, 8, 144, 72]
    assert ss.GROUPS["main"] == ["X1_careless", "X2_careful"]


def test_careful_twin_equals_careless_campaign_but_the_operator():
    x1, x2 = _parse("phase3/X1_careless.yaml"), _parse("phase3/X2_careful.yaml")
    assert x2.methods.split(",") == ["hilserl", "e2hil", "keepall", "bc", "bc-discard", "aqi"]
    for arm in x2.methods.split(","):
        for tier, op, seed in (("expert", "exp01", 0), ("novice", "nov00", 1)):
            l1, a1 = _cmds(x1, arm, tier, op, seed)
            l2, a2 = _cmds(x2, arm, tier, op, seed)
            assert l1 == l2, arm
            assert _without(a1, CARELESS) == a2, arm


def test_e2hil_mask_only_arm():
    x1 = _parse("phase3/X1_careless.yaml")
    le, ae = _cmds(x1, "e2hil")
    lm, am = _cmds(x1, "e2hil-mean")
    assert le[le.index("--e2hil_q_agg") + 1] == "min" and lm[lm.index("--e2hil_q_agg") + 1] == "mean"
    assert _drop(lm, {"--e2hil_q_agg", "--arm"}) == _drop(le, {"--e2hil_q_agg"})
    assert _drop(am, {"--e2hil_q_agg", "--arm"}) == _drop(ae, {"--e2hil_q_agg"})
    for cmd in (le, lm):
        assert "--nodemo_pretrain" in cmd and cmd[cmd.index("--aqi_mode") + 1] == "e2hil"


def test_fixed_budget_flags_reach_phase3_only():
    x1 = _parse("phase3/X1_careless.yaml")
    for arm in x1.methods.split(","):
        lc, ac = _cmds(x1, arm)
        assert [lc[lc.index(f) + 1] for f in GUARD] == ["2000", "100", "200"], arm
        assert ac[ac.index("--converge_mode") + 1] == "observe" and ac[ac.index("--max_steps") + 1] == "20000"
        assert ac[ac.index("--utd_ratio") + 1] == "1"
        assert not any(f in ac for f in GUARD)     # learner-only flags
    assert x1.eval_after_each_cell and x1.eval_prune_keep_last == 3
    for rel, arm in (("final/F2_standard.yaml", "keepall"), ("final/F5_bc_standard.yaml", "bc"),
                     ("extension/P4_hexwide.yaml", "keepall"), ("extension/S4_hexwide.yaml", "e2hil"),
                     ("study/B_quality.yaml", "e2hil")):
        lc, ac = _cmds(_parse(rel), arm)
        assert not any(f in lc + ac for f in GUARD), rel
        assert "observe" not in ac, rel
        a = _parse(rel)
        assert not a.eval_after_each_cell and a.eval_prune_keep_last == 0, rel


def test_probe_equals_p1_keepall_but_budget_and_logging():
    p1, pr = _parse("extension/P1_hex.yaml"), _parse("phase3/X0_probe.yaml")
    assert pr.methods.split(",") == ["keepall", "keepall-rerun"] and pr.seeds == 1
    l1, a1 = _cmds(p1, "keepall")
    for arm in ("keepall", "keepall-rerun"):
        lp, ap = _cmds(pr, arm)
        assert _drop(lp, {"--divergence_guard_every", "--metrics_log_every", "--arm"}) == _drop(l1, {"--arm"}), arm
        assert lp[lp.index("--divergence_guard_every") + 1] == "50" and "--checkpoint_env_every" not in lp
        assert lp[lp.index("--arm") + 1] == arm
        assert _drop(ap, {"--max_steps", "--arm"}) == _drop(a1, {"--max_steps", "--arm"}), arm
        assert ap[ap.index("--max_steps") + 1] == "3000"


# ---------------------------------------------------------------------------- analysis
def _write_run(camp, cid, curve_sr, would_stop=None, eps=None, keep=None, diverged=False, final_env=19850):
    """A run folder: environment-step checkpoints 2 000 ... 18 000 + the final one, their ruler scores, episodes."""
    ck = camp / "checkpoints" / cid
    ev = camp / "eval_runs" / cid
    ck.mkdir(parents=True)
    ev.mkdir(parents=True)
    rows, sr = [], []
    envs = list(range(2000, 18001, 2000)) + [final_env]
    for i, (env, v) in enumerate(zip(envs, curve_sr)):
        ls = 5000 + env - 1
        rows.append({"learner_step": ls, "online_updates": env, "env_steps": env, "final": i == len(envs) - 1})
        sr.append({"step": ls, "sr": v})
    (ck / "env_checkpoints.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (ck / ".done").write_text("{}")
    (ev / "cell.json").write_text(json.dumps({"sr_per_checkpoint": sr, "best_window_sr": max(sr, key=lambda x: x["sr"])["sr"]}))
    eps = eps or [{"step": 299, "intervention_count": 2, "intervention_steps": 50, "length": 300,
                   "careless_takeovers": 1}, {"step": 599, "intervention_count": 1, "intervention_steps": 20,
                                              "length": 300, "careless_takeovers": 0}]
    (ck / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in eps))
    if would_stop is not None:
        (ck / "WOULD_STOP.json").write_text(json.dumps({"step": would_stop}))
    if keep is not None:
        met = {}
        for k, v in keep.items():
            met[f"actor/e2hil_{k}_count"] = 10
            met[f"actor/e2hil_{k}_keep_frac"] = v
        (ck / "learner_metrics.jsonl").write_text("".join(json.dumps(
            {"step": s, "phase": "online", "metrics": met}) + "\n" for s in (200, 400)))
    if diverged:
        (ck / "DIVERGED.json").write_text(json.dumps({"step": 5100, "phase": "online", "nonfinite": ["critic_loss"]}))
        (ck / ".done").unlink()
        (ck / ".crashed").write_text("{}")


def test_wilcoxon_sign_holm():
    assert abs(pa.wilcoxon_exact([1, 2, 3, 4, 5, 6]) - 2 / 64) < 1e-12
    assert abs(pa.sign_test([1, 2, 3, 4, 5, 6]) - 2 / 64) < 1e-12
    assert pa.wilcoxon_exact([0, 0]) == 1.0 and pa.sign_test([]) == 1.0
    # ties: ranks (1.5, 1.5, 3), W+ = 4.5, mean 3; 6 of the 8 sign patterns are at least 1.5 from the mean
    assert pa.wilcoxon_exact([1, -1, 2]) == 0.75
    assert pa.holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


def test_analysis_on_fixture():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        x1, x2 = root / "X1_careless", root / "X2_careful"
        for op in ("exp00", "int00", "nov00"):
            for seed in (0, 1):
                _write_run(x1, f"bc_{op}_s{seed}", [0.1 * i for i in range(10)], would_stop=6100)
                _write_run(x1, f"bc-discard_{op}_s{seed}", [0.1 * i + 0.05 for i in range(10)], would_stop=4100)
                _write_run(x1, f"hilserl_{op}_s{seed}", [0.0] * 10)
                _write_run(x1, f"e2hil_{op}_s{seed}", [0.2] * 10, keep={"intervention": 0.9, "careless": 0.7},
                           diverged=(op == "nov00" and seed == 1))
                _write_run(x2, f"hilserl_{op}_s{seed}", [0.0] * 10)
                _write_run(x2, f"e2hil_{op}_s{seed}", [0.1] * 10)
        runs = pa.read_root(root)
        r = next(x for x in runs if x["cell"] == "bc_int00_s0")
        assert abs(r["success_at_B"] - (0.7 + 0.8 + 0.9) / 3) < 1e-9 and abs(r["AULC"] - 0.45) < 1e-9
        assert r["would_stop"] == 6100 and r["op_steps_to_stop"] == 70 and r["op_steps_total"] == 70
        assert r["careless_share"] == 1 / 3 and r["tier"] == "intermediate"
        h = next(x for x in runs if x["cell"] == "hilserl_int00_s0" and x["campaign"] == "X1_careless")
        assert h["stop_censored"] and h["op_steps_to_stop"] == 70
        bad = next(x for x in runs if x["cell"] == "e2hil_nov00_s1" and x["campaign"] == "X1_careless")
        assert not bad["valid"] and bad["why_not"] == "diverged"
        cs, e2 = pa.analyse(runs)
        c = next(x for x in cs if x["label"].startswith("imitate, discard lowest 40 % vs imitate all"))
        assert c["n"] == 6 and c["better"] == 6 and abs(c["mean_diff"] - 0.05) < 1e-9 and c["holm_p"] is not None
        e = next(x for x in cs if x["label"] == "E2HiL vs HIL-SERL, careless operator")
        assert e["n"] == 5 and e["left_out"] == ["nov00 s1"]
        i = next(x for x in cs if x["label"].startswith("interaction: E2HiL"))
        assert i["n"] == 5 and abs(i["mean_diff"] - 0.1) < 1e-9          # (0.2 - 0) - (0.1 - 0)
        assert e2["e2hil"]["n"] == 5 and e2["e2hil"]["lower"] == 5 and abs(e2["e2hil"]["mean_diff"] + 0.2) < 1e-9
        assert e2["e2hil-mean"]["n"] == 0
        out = root / "res"
        with contextlib.redirect_stdout(io.StringIO()):
            pa.main(["--root", str(root), "--out", str(out)])
        text = (out / "PHASE3_REPORT.md").read_text(encoding="utf-8")
        assert "## Every run" in text and "X1_careless/e2hil_nov00_s1: diverged" in text
        assert (out / "phase3_runs.csv").exists() and (out / "phase3_contrasts.json").exists()


# ---------------------------------------------------------------------------- suite and runner helpers
def test_copy_inputs_checks_hashes():
    ss.configure("phase3")
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d) / "P1_hex", Path(d) / "X0_probe"
        (src / "operators").mkdir(parents=True)
        (src / "demos" / "int00_s0").mkdir(parents=True)
        (src / "operators" / "int00.json").write_text('{"a": 1}')
        (src / "operators" / "nov00.json").write_text('{"b": 1}')
        (src / "demos" / "int00_s0" / "demos.pkl").write_bytes(b"demo")
        (src / "eval_grid.json").write_text("{}")
        items = ["operators/int00.json", "demos/int00_s0", "eval_grid.json"]
        ok, msg = ss.copy_inputs(src, dst, items)
        assert ok, msg
        assert (dst / "demos" / "int00_s0" / "demos.pkl").exists() and not (dst / "operators" / "nov00.json").exists()
        assert json.loads((dst / "inputs_from.json").read_text())["files"] == 3
        (dst / "demos" / "int00_s0" / "demos.pkl").write_bytes(b"other")
        ok, msg = ss.copy_inputs(src, dst, items)
        assert not ok and "demos/int00_s0/demos.pkl" in msg
        ok, msg = ss.copy_inputs(Path(d) / "missing", Path(d) / "y", items)
        assert not ok and "missing" in msg


def test_probe_report_and_smoke_checks():
    ss.configure("phase3")
    with tempfile.TemporaryDirectory() as d:
        probe = Path(d) / "X0_probe"
        for cid, div in (("keepall_int00_s0", True), ("keepall-rerun_int00_s0", False)):
            ck = probe / "checkpoints" / cid
            ck.mkdir(parents=True)
            (ck / "learner_metrics.jsonl").write_text(json.dumps({"step": 5050, "phase": "pretrain",
                                                                  "metrics": {"critic/critic_loss": 1.0}}) + "\n")
        (probe / "diverged").mkdir()
        (probe / "diverged" / "keepall_int00_s0.20261126_100000.json").write_text(json.dumps(
            {"step": 5100, "phase": "pretrain", "nonfinite": ["critic/critic_loss"], "last_finite_step": 5050,
             "last_finite_metrics": {"critic/critic_loss": 1e6, "temperature": 0.01}}))
        with contextlib.redirect_stdout(io.StringIO()):
            assert ss.probe_report(probe) is True
        pj = json.loads((probe / "probe.json").read_text())
        assert [r["diverged"] for r in pj["runs"]] == [False, True]
        assert "REPRODUCED" not in (probe / "PROBE.md").read_text() and "reproduced" in (probe / "PROBE.md").read_text()
        smoke = Path(d) / "X_smoke"
        for arm in ("hilserl", "e2hil", "e2hil-mean", "keepall", "bc-random", "asks-15", "random-15", "aqi"):
            ck = smoke / "checkpoints" / f"{arm}_int00_s0"
            ck.mkdir(parents=True)
            (ck / ".done").write_text("{}")
            (ck / "env_checkpoints.jsonl").write_text("".join(json.dumps({"learner_step": s, "final": s == 4000})
                                                              + "\n" for s in (1000, 2000, 3000, 4000)))
            (ck / "convergence_evals.jsonl").write_text('{"step": 1200, "sr": 0.4}\n')
            (ck / "episodes.jsonl").write_text('{"step": 299, "careless_takeovers": 1}\n')
            met = {"actor/e2hil_careless_keep_frac": 0.8} if arm.startswith("e2hil") else {}
            (ck / "learner_metrics.jsonl").write_text(json.dumps({"step": 200, "metrics": met}) + "\n")
            (smoke / "eval_runs" / f"{arm}_int00_s0").mkdir(parents=True)
            (smoke / "eval_runs" / f"{arm}_int00_s0" / "cell.json").write_text("{}")
        checks = ss.smoke_checks(smoke)
        assert all(ok for _c, ok, _d in checks), checks
        (smoke / "checkpoints" / "e2hil_int00_s0" / "learner_metrics.jsonl").write_text('{"metrics": {}}\n')
        assert not dict((c, ok) for c, ok, _d in ss.smoke_checks(smoke))["E2HiL keep fractions of careless takeover states"]


class _Proc:
    """A process whose wait() times out `n` times before it exits (n None: never)."""

    def __init__(self, n=None, rc=0):
        self.n, self.rc, self.args, self.calls = n, rc, ["x"], 0

    def wait(self, timeout=None):
        self.calls += 1
        if self.n is None or self.calls <= self.n:
            import subprocess
            raise subprocess.TimeoutExpired("x", timeout)
        return self.rc

    def poll(self):
        return 0


def test_wait_actor_stops_after_learner_divergence():
    import subprocess
    with tempfile.TemporaryDirectory() as d:
        ck = Path(d)
        killed = []
        orig = rc._terminate
        rc._terminate = lambda p: killed.append(p) or -15
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                assert rc._wait_actor(_Proc(n=2, rc=0), None, ck, _Proc(), grace=0.05, poll=0.001) == 0
                assert not killed                            # no marker: plain wait
                try:
                    rc._wait_actor(_Proc(), 0.02, ck, _Proc(), grace=10.0, poll=0.001)
                    raise AssertionError("the timeout must still raise")
                except subprocess.TimeoutExpired:
                    pass
                (ck / "DIVERGED.json").write_text("{}")      # learner diverged and exited: actor terminated
                assert rc._wait_actor(_Proc(), None, ck, _Proc(), grace=0.02, poll=0.001) == -15
                assert len(killed) == 1
        finally:
            rc._terminate = orig


def test_prune_scored_cell_keeps_best_window_and_last_three():
    with tempfile.TemporaryDirectory() as d:
        camp = Path(d)
        cell = rc.Cell("bc", "int00", "intermediate", 0)
        ck = camp / "checkpoints" / cell.id
        for s in (6999, 8999, 10999, 12999, 14999, 16999, 18999, 20999, 22999, 24849):
            (ck / f"checkpoint_{s}").mkdir(parents=True)
        (ck / ".done").write_text("{}")
        (camp / "eval_runs" / cell.id).mkdir(parents=True)
        (camp / "eval_runs" / cell.id / "cell.json").write_text(json.dumps(
            {"best_window_steps": [8999, 10999, 12999], "checkpoint_step": 10999}))
        with contextlib.redirect_stdout(io.StringIO()):
            gone = rc.prune_scored_cell(camp, cell, 3)
        left = sorted(int(p.name[11:]) for p in ck.iterdir() if p.name.startswith("checkpoint_"))
        assert gone == 4 and left == [8999, 10999, 12999, 20999, 22999, 24849]
        assert rc.prune_scored_cell(camp, cell, 0) == 0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
