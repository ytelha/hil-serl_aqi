"""Tests for named campaign ARMS in examples/run_campaign.py.

An arm = a base --aqi_mode + per-arm overrides (YAML `arms:`), so several
variants of one mode can share a campaign. Run from the repo root with
    python -m pytest tests/test_campaign_arms.py -q
    python tests/test_campaign_arms.py
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for _p in ("serl_launcher", "examples"):
    sys.path.insert(0, str(REPO / _p))

import intervention_curves as ic              # noqa: E402
import run_campaign as rc                     # noqa: E402

CAMPAIGNS = REPO / "examples" / "campaigns"


def _parse(config, extra=()):
    argv = ["--campaign", "unused", "--config", str(CAMPAIGNS / config),
            *extra]
    parser = rc.build_arg_parser()
    args = parser.parse_args(argv)
    cfg = rc.load_config(str(CAMPAIGNS / config))
    args.ops_per_tier_map = rc.apply_config(
        args, cfg, rc._explicit_dests(parser, argv)) or None
    return args


def _cmds(args, arm):
    ckpt = Path("ckpt")
    lcmd = rc._learner_cmd(args, ckpt, Path("d.pkl"), arm, 0, "r")
    acmd = rc._actor_cmd(args, ckpt, arm, "expert", "exp00", Path("t.json"),
                         0, "r")
    return " ".join(lcmd), " ".join(acmd)


def _flag(cmd, name):
    toks = cmd.split()
    return toks[toks.index(name) + 1] if name in toks else None


def test_rq4_config_still_means_the_same():
    """method_overrides (the pre-arms mechanism) keeps working unchanged."""
    args = _parse("rq4_ensemble_convergence.yaml")
    rc.validate_arms(args, args.methods.split(","))
    hl, ha = _cmds(args, "hilserl")
    ll, la = _cmds(args, "logged")
    assert "--nodemo_pretrain" in hl and "--learning_rate" not in hl
    assert "--demo_pretrain " in ll and _flag(ll, "--learning_rate") == "0.00015"
    for cmd in (ll, la):
        assert _flag(cmd, "--critic_ensemble_size") == "8"
        assert _flag(cmd, "--critic_subsample_size") == "1"
    for cmd in (hl, ha, ll, la):
        assert "--arm" not in cmd.split()          # plain arms: no --arm


def test_sweep_arms_resolve_per_arm():
    args = _parse("rq4_attribution_sweep.yaml")
    methods = args.methods.split(",")
    rc.validate_arms(args, methods)
    expect = {  # arm: (K, M, pretrain, LR flag)
        "ens-rq3": ("10", "2", False, None),
        "ens-m1": ("10", "1", False, None),
        "ens-m1-pt": ("10", "1", True, None),
        "ens-fix-k10": ("10", "1", True, "0.00015"),
        "ens-fix-k8": ("8", "1", True, "0.00015"),
    }
    for arm, (k, m, pt, lr) in expect.items():
        lcmd, acmd = _cmds(args, arm)
        for cmd in (lcmd, acmd):
            assert _flag(cmd, "--aqi_mode") == "logged", arm
            assert _flag(cmd, "--arm") == arm, arm
            assert _flag(cmd, "--critic_ensemble_size") == k, arm
            assert _flag(cmd, "--critic_subsample_size") == m, arm
        assert (("--demo_pretrain " in lcmd + " ") == pt
                and ("--nodemo_pretrain" in lcmd) == (not pt)), arm
        assert _flag(lcmd, "--learning_rate") == lr, arm
        # warm-start arms get the pretrain budget added to the READY wait
        extra = rc._ready_timeout(args, arm) - args.ready_timeout
        assert extra == (5000 / 2.5 if pt else 0.0), arm
    hl, _ = _cmds(args, "hilserl")
    assert _flag(hl, "--aqi_mode") == "hilserl" and "--arm" not in hl.split()
    assert set(rc._arms_summary(args)) == set(methods)


def test_precedence_arms_over_method_overrides_over_campaign():
    args = _parse("rq4_ensemble_convergence.yaml")
    args.arms = {"logged": {"learning_rate": 0.0002}}
    # method_overrides has no logged entry; the arm wins over campaign 1.5e-4
    assert rc._arm_val(args, "logged", "learning_rate", 0.00015) == 0.0002
    args.method_overrides = {"logged": {"learning_rate": 0.0001}}
    assert rc._arm_val(args, "logged", "learning_rate", 0.00015) == 0.0002
    args.arms = {}
    assert rc._arm_val(args, "logged", "learning_rate", 0.00015) == 0.0001
    assert rc._arm_val(args, "aqi", "learning_rate", 0.00015) == 0.00015


def test_validation_fails_fast():
    base = _parse("rq4_ensemble_convergence.yaml")
    bad_cases = [
        (["quality_beh"], {}),                                  # "_" in name
        (["quality-x"], {}),                                    # no mode
        (["q-x"], {"q-x": {"mode": "nonsense"}}),               # bad mode
        (["q-x"], {"q-x": {"mode": "quality", "steps": 5}}),    # not per-arm
        (["q-x"], {"q-x": {"mode": "quality",
                           "per_standardize": "maybe"}}),       # bad bool
        (["logged"], {"logged": {"mode": "quality"}}),          # remapped mode
    ]
    for methods, arms in bad_cases:
        args = argparse.Namespace(**vars(base))
        args.arms = arms
        try:
            rc.validate_arms(args, methods)
        except SystemExit:
            continue
        raise AssertionError(f"accepted {methods} {arms}")


def test_conditions_and_manifest_record_the_arm():
    args = _parse("rq4_attribution_sweep.yaml")
    rows = {r["choice"]: r for r in rc._method_conditions("ens-m1-pt", args)}
    assert rows["aqi_mode"]["value"] == "logged"
    assert "critic_subsample_size=1" in rows["arm"]["note"]
    assert rows["critic_subsample_size"]["value"] == "1"
    assert rows["critic_ensemble_size"]["value"] == "10"
    assert rows["demo_pretrain"]["value"] == "5000 steps"
    assert rows["learning_rate"]["value"] == "3e-4 (built-in)"
    summary = rc._arms_summary(args)["ens-fix-k8"]
    assert summary == {"aqi_mode": "logged", "overrides": {
        "critic_ensemble_size": 8, "critic_subsample_size": 1,
        "demo_pretrain": True, "demo_pretrain_steps": 5000,
        "learning_rate": 0.00015}}


def test_empty_quality_filters_means_none():
    args = _parse("rq4_ensemble_convergence.yaml")
    args.arms = {"q-nofilt": {"mode": "quality", "quality_filters": []}}
    lcmd, _ = _cmds(args, "q-nofilt")
    assert _flag(lcmd, "--quality_filters") == "none"


def test_cell_id_parsing_handles_arm_names():
    assert ic._method_of("quality-beh_exp00_s0") == "quality-beh"
    assert ic._method_of("ens-fix-k8_nov02_s1") == "ens-fix-k8"
    assert ic._method_of("hilserl_int01_s3") == "hilserl"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
