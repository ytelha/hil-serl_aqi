"""Tests for the careful-operator control of the final study (examples/campaigns/final/F3_careful_hard.yaml,
F4_careful_standard.yaml, tools/run_final_careful_unattended.sh, docs/final_careful_control.md).

The control must train exactly like the final study's `keepall` arm, with one difference: the operator is never
careless. So every learner, actor and evaluation command of F3 (F4) must equal F1's (F2's) keepall command without the
three careless-operator flags, and the roster, demonstrations, ruler, budget and checkpoints must be the same.

    python tests/test_final_careful.py        (numpy only; no GPU, no simulator)
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
PAIRS = {"F3_careful_hard": "F1_hard", "F4_careful_standard": "F2_standard"}
CARELESS = ["--synthetic_inconsistent", "--synthetic_bad_share", "0.4", "--synthetic_bad_persistence", "0.7"]
CELLS = [("expert", "exp00", 0), ("intermediate", "int01", 1), ("novice", "nov00", 0), ("novice", "nov01", 1)]


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
    return (rc._learner_cmd(args, ck, Path("d.pkl"), arm, seed, "r"),
            rc._actor_cmd(args, ck, arm, tier, op, Path("t.json"), seed, "r"))


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


def _without_careless(cmd):
    cmd = [str(c) for c in cmd]
    i = cmd.index(CARELESS[0])
    assert cmd[i:i + len(CARELESS)] == CARELESS, cmd[i:i + len(CARELESS)]
    return cmd[:i] + cmd[i + len(CARELESS):]


def _plan(name):
    d = Path(tempfile.mkdtemp()) / name
    with contextlib.redirect_stdout(io.StringIO()):
        rc.main(["--campaign", str(d), "--config", str(ss.config_path(name)), "--plan"])
    return json.loads((d / "campaign.json").read_text())


def test_one_arm_keepall_and_the_operator_never_careless():
    for new in PAIRS:
        args = _parse(new)
        assert args.methods.split(",") == ["keepall"], new
        m = _plan(new)
        assert not m.get("synthetic_inconsistent"), (new, m.get("synthetic_inconsistent"))
        assert m["methods"] == ["keepall"] and ss.planned_cells(new) == 12, new


def test_training_commands_equal_the_final_keepall_without_the_careless_flags():
    for new, old in PAIRS.items():
        a, b = _parse(new), _parse(old)
        for tier, op, seed in CELLS:
            (ln, an), (lo, ao) = _cmds(a, "keepall", tier, op, seed), _cmds(b, "keepall", tier, op, seed)
            assert [str(c) for c in ln] == [str(c) for c in lo], (new, tier, "learner")
            assert not any(str(c) in CARELESS[::2] for c in an), (new, tier, "careless flag in the control")
            assert [str(c) for c in an] == _without_careless(ao), (new, tier, "actor")


def test_evaluation_commands_are_the_same():
    for new, old in PAIRS.items():
        a, b = _parse(new), _parse(old)
        for tier, op, seed in CELLS:
            assert _eval_cmd(a, "keepall", tier, op, seed) == _eval_cmd(b, "keepall", tier, op, seed), (new, tier)


def test_settings_differ_only_in_methods_operator_and_tags():
    for new, old in PAIRS.items():
        a, b = vars(_parse(new)), vars(_parse(old))
        diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
        allowed = {"config", "methods", "wandb_tags", "synthetic_inconsistent", "arms", "method_overrides",
                   "analysis_baseline", "baseline"}
        assert diff <= allowed, (new, diff - allowed)
        assert a["demos"] == b["demos"] and a["checkpoint_keep"] == b["checkpoint_keep"], new


def test_same_roster_budget_and_ruler_as_the_final_study():
    for new, old in PAIRS.items():
        mn, mo = _plan(new), _plan(old)
        assert mn["operators"] == mo["operators"], new                       # same sampled operators
        for k in ("operators_per_tier", "demos", "filtered", "steps", "learner_max_steps", "utd_ratio",
                  "synthetic_rotation_control", "synthetic_orientation_monitor", "sim_rpy_box", "converge_mode",
                  "log_takeover_channels", "ruler_checkpoints", "grid_sha256", "n_grid_trials", "seeds",
                  "sample_seed", "jitter", "peg_shape"):
            assert mn.get(k) == mo.get(k), (new, k, mn.get(k), mo.get(k))


def test_unattended_wrapper_separate_folders_hash_check_and_never_pushes():
    text = (REPO / "tools" / "run_final_careful_unattended.sh").read_text(encoding="utf-8")
    assert "\r" not in text
    assert "--push" not in text and "git push" not in text
    assert re.search(r'^DST="runs/final_careful"$', text, re.M) and re.search(r'^RES="results/final_careful"$', text, re.M)
    assert '--root "${DST}" --results "${RES}"' in text and "--suite final" in text
    assert not re.search(r"\brm\s", text)                                    # nothing is deleted anywhere
    assert "sha256sum" in text and "checkpoint_keep_every" not in text       # inputs checked; no disk fallback
    for pair in ('"F1_hard:F3_careful_hard"', '"F2_standard:F4_careful_standard"'):
        assert pair in text


def test_the_final_study_itself_is_unchanged():
    assert ss.ORDER == ["F0_smoke", "F1_hard", "F2_standard"]
    assert ss.resolve(["main"]) == ["F1_hard", "F2_standard"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"{len(tests)} tests passed")
