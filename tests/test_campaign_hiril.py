"""Tests for the HiRIL pipeline's campaign integration (run_campaign.py,
eval_ruler.py, provenance.py). No JAX needed for the checks themselves.

The key check is STATIC: every flag the campaign passes to train_hiril.py /
eval_ruler.py must be defined there -- an unknown flag would only surface as
an absl error on the training machine.

Run from the repo root with
    python -m pytest tests/test_campaign_hiril.py -q
    python tests/test_campaign_hiril.py
"""

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for _p in ("serl_launcher", "examples"):
    sys.path.insert(0, str(REPO / _p))

import provenance as prov                     # noqa: E402
import run_campaign as rc                     # noqa: E402
from serl_launcher.hiril.config import CLI_FIELDS, HiRILConfig  # noqa: E402

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


def _defined_flags(script):
    """absl flag names defined in a script (static parse), incl. the
    generated --hiril_<name> flags of train_hiril.py."""
    text = (REPO / "examples" / script).read_text(encoding="utf-8")
    names = set(re.findall(r'flags\.DEFINE_\w+\(\s*"(\w+)"', text))
    if "for _name, _cast in CLI_FIELDS.items()" in text:
        names |= {f"hiril_{n}" for n in CLI_FIELDS}
    return names


def _used_flags(cmd):
    out = set()
    for tok in cmd:
        if isinstance(tok, str) and tok.startswith("--"):
            name = tok[2:].split("=", 1)[0]
            out.add(name)
    return out


def _cmds(args, arm):
    ckpt = Path("runs/x/checkpoints/c")
    return (rc._learner_cmd(args, ckpt, Path("d.pkl"), arm, 0, "r"),
            rc._actor_cmd(args, ckpt, arm, "expert", "exp00", Path("t.json"),
                          0, "r"))


def test_hiril_commands_use_the_hiril_pipeline_and_known_flags():
    args = _parse("hiril_comparison.yaml")
    rc.validate_arms(args, args.methods.split(","))
    defined = _defined_flags("train_hiril.py")
    lcmd, acmd = _cmds(args, "hiril")
    for cmd in (lcmd, acmd):
        assert cmd[0] == "train_hiril.py"
        used = _used_flags(cmd)
        booleans = {u[2:] for u in used if u.startswith("no")
                    and u[2:] in defined}
        unknown = used - defined - {f"no{b}" for b in booleans}
        assert not unknown, f"flags not defined in train_hiril.py: {unknown}"
        assert "aqi_mode" not in used and "learning_rate" not in used
        s = " ".join(cmd)
        assert "--hiril_tau_risk 0.0" in s and "--hiril_lambda_bc 1.0" in s
        assert "--arm hiril" in s
    hl, _ = _cmds(args, "hilserl")
    assert hl[0] == "train_aqi_rlpd.py"
    assert not any(t.startswith("--hiril_") for t in hl)


def test_eval_gets_the_agent_type_and_the_flag_exists():
    assert "agent_type" in _defined_flags("eval_ruler.py")
    args = _parse("hiril_comparison.yaml")
    args.dry_run = True
    captured = []
    orig = rc._yellow
    rc._yellow = captured.append
    try:
        for arm in ("hiril", "hilserl"):
            rc.eval_cell(args, Path("unused_campaign"),
                         rc.Cell(arm, "exp00", "expert", 0))
    finally:
        rc._yellow = orig
    assert "--agent_type hiril" in captured[0]
    assert "--agent_type" not in captured[1]


def test_per_arm_hiril_knobs_and_validation():
    args = _parse("hiril_comparison.yaml")
    args.arms = {"hiril-noepi": {"mode": "hiril",
                                 "hiril_du_mode": "aleatoric",
                                 "hiril_lambda_bc": 0.0}}
    rc.validate_arms(args, ["hilserl", "hiril", "hiril-noepi"])
    lcmd, _ = _cmds(args, "hiril-noepi")
    s = " ".join(lcmd)
    assert "--hiril_du_mode aleatoric" in s and "--hiril_lambda_bc 0.0" in s
    assert "--arm hiril-noepi" in s
    rows = {r["choice"]: r for r in rc._method_conditions("hiril-noepi", args)}
    assert rows["pipeline"]["value"] == "hiril"
    assert rows["du_mode"]["value"] == "aleatoric"
    assert rows["tau_risk"]["note"].startswith("calibrated")
    for bad in ({"h-x": {"mode": "hiril", "hiril_du_mode": "nope"}},
                {"hiril": {"mode": "quality"}}):
        a = argparse.Namespace(**vars(args))
        a.arms = bad
        try:
            rc.validate_arms(a, list(bad))
        except SystemExit:
            continue
        raise AssertionError(f"accepted {bad}")


def test_paper_ablations_as_arms_emit_valid_flags():
    """Fig. 6 / 8 ablations of the paper, each as a named arm."""
    args = _parse("hiril_comparison.yaml")
    args.arms = {
        "hiril-no-epi": {"mode": "hiril", "hiril_du_mode": "aleatoric"},
        "hiril-no-alea": {"mode": "hiril", "hiril_du_mode": "epistemic"},
        "hiril-no-rd": {"mode": "hiril", "hiril_use_rd": False},
        "hiril-no-shaping": {"mode": "hiril",
                             "hiril_intervention_penalty": 0.0},
        "hiril-no-bc": {"mode": "hiril", "hiril_lambda_bc": 0.0},
    }
    rc.validate_arms(args, list(args.arms))
    defined = _defined_flags("train_hiril.py")
    for arm in args.arms:
        for cmd in _cmds(args, arm):
            toks = [t for t in cmd if isinstance(t, str)]
            for t in toks:
                if t.startswith("--"):
                    name = t[2:]
                    assert name in defined or (name.startswith("no")
                                               and name[2:] in defined), t
    lcmd, _ = _cmds(args, "hiril-no-rd")
    assert "--nohiril_use_rd" in lcmd
    assert "False" not in lcmd          # never "--flag False" for absl bools
    args.hiril_use_rd = True            # campaign-wide True, arm overrides
    assert "--nohiril_use_rd" in _cmds(args, "hiril-no-rd")[0]
    assert "--hiril_use_rd" in _cmds(args, "hiril")[0]


def test_inert_knobs_are_reported():
    args = _parse("hiril_comparison.yaml")
    args.arms = {"hiril-x": {"mode": "hiril", "quality_score_mode":
                             "behavioral", "learning_rate": 0.001},
                 "logged-x": {"mode": "logged", "hiril_lambda_bc": 2.0}}
    warned = []
    orig = rc._yellow
    rc._yellow = warned.append
    try:
        rc.validate_arms(args, ["hiril-x", "logged-x"])
    finally:
        rc._yellow = orig
    text = "\n".join(warned)
    assert "hiril-x" in text and "quality_score_mode" in text
    assert "learning_rate" in text
    assert "logged-x" in text and "hiril_lambda_bc" in text


def test_yaml_hiril_block_rejects_unknown_keys():
    args = _parse("hiril_comparison.yaml")
    try:
        rc.apply_config(args, {"hiril": {"tau_rsik": 1.0}}, set())
    except SystemExit:
        return
    raise AssertionError("accepted a misspelled hiril key")


def test_provenance_rows_cover_every_knob():
    d = HiRILConfig().as_dict()
    rows = prov.hiril_condition_rows(d)
    assert rows[0]["choice"] == "pipeline"
    assert {r["choice"] for r in rows[1:]} == set(d)
    assert "hiril" in prov.MODE_ONE_LINER


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
