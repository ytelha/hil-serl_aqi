"""Tests for phases 0 and 1 of the extension study: the twins of the final study (examples/campaigns/final/F3-F6,
tools/run_final_twins_unattended.sh, docs/extension_study.md).

A twin must train exactly like the final study's `keepall` arm of its source campaign, apart from the deliberate
differences: F3/F4 never careless; F5 adds imitation (weight 0.1, gamma 0) and one discard filter; F6 adds imitation
and is never careless. Every learner, actor and evaluation command is compared flag by flag.

    python tests/test_final_twins.py        (numpy only; no GPU, no simulator)
"""
import contextlib
import io
import json
import re
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for _p in ("serl_launcher", "examples", "franka_sim"):
    sys.path.insert(0, str(REPO / _p))

import run_campaign as rc                      # noqa: E402
import synthetic_suite as ss                   # noqa: E402

FINAL = REPO / "examples" / "campaigns" / "final"
ss.configure("final")
SOURCE = {"F3_careful_hard": "F1_hard", "F4_careful_standard": "F2_standard",
          "F5_bc_standard": "F2_standard", "F6_bc_careful_standard": "F2_standard"}
CARELESS = ["--synthetic_inconsistent", "--synthetic_bad_share", "0.4", "--synthetic_bad_persistence", "0.7"]
CELLS = [("expert", "exp00", 0), ("intermediate", "int01", 1), ("novice", "nov00", 0), ("novice", "nov01", 1)]
# arm -> (filter, drop fraction) of the imitation arms; all have bc 0.1 and gamma 0
BC_ARMS = {"bc": ("none", None), "bc-discard": ("score_filter", "0.4"), "bc-random": ("random_filter", "0.4"),
           "bc-perfect": ("oracle_filter", None)}


def _parse(name, extra=()):
    cfg_p = str(FINAL / f"{name}.yaml")
    argv = ["--campaign", "unused", "--config", cfg_p, *extra]
    parser = rc.build_arg_parser()
    args = parser.parse_args(argv)
    args.ops_per_tier_map = rc.apply_config(args, rc.load_config(cfg_p), rc._explicit_dests(parser, argv)) or None
    rc.validate_arms(args, args.methods.split(","))
    return args


def _cmds(args, arm, tier, op, seed):
    ck = Path("runs/x/checkpoints/c")
    return ([str(c) for c in rc._learner_cmd(args, ck, Path("d.pkl"), arm, seed, "r")],
            [str(c) for c in rc._actor_cmd(args, ck, arm, tier, op, Path("t.json"), seed, "r")])


def _eval_cmd(args, arm, tier, op, seed):
    args.dry_run = True
    seen = []
    orig = rc._yellow
    rc._yellow = seen.append
    try:
        rc.eval_cell(args, Path("runs/x"), rc.Cell(arm, op, tier, seed))
    finally:
        rc._yellow = orig
    return seen[0]


def _flags(cmd):
    """{flag: value or True} of a command; the script name is kept under '_script'."""
    out, i = {"_script": cmd[0]}, 1
    while i < len(cmd):
        t = cmd[i]
        if t.startswith("--"):
            if i + 1 < len(cmd) and not cmd[i + 1].startswith("--"):
                out[t] = cmd[i + 1]
                i += 2
                continue
            out[t] = True
        i += 1
    return out


def _without_careless(cmd):
    i = cmd.index(CARELESS[0])
    assert cmd[i:i + len(CARELESS)] == CARELESS, cmd[i:i + len(CARELESS)]
    return cmd[:i] + cmd[i + len(CARELESS):]


def _plan(name):
    d = Path(tempfile.mkdtemp()) / name
    with contextlib.redirect_stdout(io.StringIO()):
        rc.main(["--campaign", str(d), "--config", str(ss.config_path(name)), "--plan"])
    return json.loads((d / "campaign.json").read_text())


def _expected_bc(base, arm):
    """The source keepall command's flags with the deliberate changes of an imitation arm."""
    filt, drop = BC_ARMS[arm]
    want = dict(base)
    assert want["--bc_loss_weight"] == "0.0" and want["--quality_filters"] == "none"
    want["--arm"] = arm                                   # the arm's name (run label)
    want["--bc_loss_weight"] = "0.1"
    want["--quality_filters"] = filt
    want["--quality_gamma"] = "0.0"
    if drop is not None:
        want["--quality_drop_fraction"] = drop
    return want


# ---------------------------------------------------------------- the configs
def test_arms_and_operator_condition():
    want = {"F3_careful_hard": (["keepall"], False), "F4_careful_standard": (["keepall"], False),
            "F5_bc_standard": (["bc", "bc-discard", "bc-random", "bc-perfect"], True),
            "F6_bc_careful_standard": (["bc"], False)}
    for name, (arms, careless) in want.items():
        args = _parse(name)
        assert args.methods.split(",") == arms, name
        m = _plan(name)
        assert bool(m.get("synthetic_inconsistent")) == careless, (name, m.get("synthetic_inconsistent"))
        assert ss.planned_cells(name) == 12 * len(arms), name


def test_careful_keepall_equals_the_final_keepall_without_the_careless_flags():
    for new in ("F3_careful_hard", "F4_careful_standard"):
        a, b = _parse(new), _parse(SOURCE[new])
        for tier, op, seed in CELLS:
            (ln, an), (lo, ao) = _cmds(a, "keepall", tier, op, seed), _cmds(b, "keepall", tier, op, seed)
            assert ln == lo, (new, tier, "learner")
            assert an == _without_careless(ao), (new, tier, "actor")


def test_imitation_arms_differ_from_the_final_keepall_only_as_intended():
    for new in ("F5_bc_standard", "F6_bc_careful_standard"):
        a, b = _parse(new), _parse(SOURCE[new])
        for arm in a.methods.split(","):
            for tier, op, seed in CELLS:
                (ln, an), (lo, ao) = _cmds(a, arm, tier, op, seed), _cmds(b, "keepall", tier, op, seed)
                assert _flags(ln) == _expected_bc(_flags(lo), arm), (new, arm, tier, "learner")
                base_actor = ao if new == "F5_bc_standard" else _without_careless(ao)
                assert _flags(an) == _expected_bc(_flags(base_actor), arm), (new, arm, tier, "actor")
                if new == "F6_bc_careful_standard":
                    assert "--synthetic_inconsistent" not in an, (new, arm)


def test_evaluation_commands_are_the_same():
    for new, old in SOURCE.items():
        a, b = _parse(new), _parse(old)
        for arm in a.methods.split(","):
            for tier, op, seed in CELLS:
                ea = _eval_cmd(a, arm, tier, op, seed).replace(f"{arm}_{op}", "CELL")
                eb = _eval_cmd(b, "keepall", tier, op, seed).replace(f"keepall_{op}", "CELL")
                assert ea == eb, (new, arm, tier)


def test_settings_differ_only_in_methods_operator_and_tags():
    allowed = {"config", "methods", "wandb_tags", "synthetic_inconsistent", "arms", "method_overrides",
               "analysis_baseline", "baseline"}
    for new, old in SOURCE.items():
        a, b = vars(_parse(new)), vars(_parse(old))
        diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
        assert diff <= allowed, (new, diff - allowed)


def test_same_roster_budget_and_ruler_as_the_source():
    keys = ("operators_per_tier", "demos", "filtered", "steps", "learner_max_steps", "utd_ratio",
            "synthetic_rotation_control", "synthetic_orientation_monitor", "sim_rpy_box", "converge_mode",
            "log_takeover_channels", "ruler_checkpoints", "grid_sha256", "n_grid_trials", "seeds", "sample_seed",
            "jitter", "peg_shape")
    for new, old in SOURCE.items():
        mn, mo = _plan(new), _plan(old)
        assert mn["operators"] == mo["operators"], new
        for k in keys:
            assert mn.get(k) == mo.get(k), (new, k, mn.get(k), mo.get(k))
        assert "sim_task_knobs" not in mn, new                      # the standard task, untouched


# ---------------------------------------------------------------- the runners
def test_twins_runner_pairs_hash_check_skip_and_never_push():
    text = (REPO / "tools" / "run_final_twins_unattended.sh").read_text(encoding="utf-8")
    assert "\r" not in text
    assert "--push" not in text and "git push" not in text
    assert not re.search(r"\brm\s", text)                                   # it deletes nothing itself
    assert re.search(r'^SRC="runs/final"$', text, re.M) and re.search(r'^DST="runs/final_twins"$', text, re.M)
    assert re.search(r'^RES="results/final_twins"$', text, re.M)
    for pair in SOURCE.items():
        assert f'"{pair[1]}:{pair[0]}"' in text, pair
    assert "sha256sum" in text and "SKIPPED" in text                        # a twin with other inputs is skipped
    assert '"${SUITE[@]}" prune' in text                                    # pruning through the suite's guard
    body = text.split("\nshift\n", 1)[1]
    assert body.count("exit 2") == 1 and "no .venv" in body                 # the only stop: nothing can run


def test_pipeline_order_and_never_push():
    text = (REPO / "tools" / "run_extension_unattended.sh").read_text(encoding="utf-8")
    assert "\r" not in text and "--push" not in text and "git push" not in text
    assert not re.search(r"\brm\s", text)
    i_cpu = text.index("--suite extension --no_export stage0")
    i_twins = text.index("bash tools/run_final_twins_unattended.sh --here")
    i_wait = text.index('wait "${CPU_PID}"')
    i_ext = text.index("--suite extension all")
    assert i_cpu < i_twins < i_wait < i_ext


def test_the_final_study_itself_is_unchanged():
    assert ss.ORDER == ["F0_smoke", "F1_hard", "F2_standard"]
    assert ss.resolve(["main"]) == ["F1_hard", "F2_standard"]
    for name, n in (("F1_hard", 84), ("F2_standard", 84)):
        assert ss.planned_cells(name) == n


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"{len(tests)} tests passed")
