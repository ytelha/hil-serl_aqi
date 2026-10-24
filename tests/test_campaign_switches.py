"""Tests for the campaign-wide platform switches in run_campaign.py:
convergence by the mini-eval (converge_mode eval), a fixed update ratio
(utd_ratio) and takeover-channel logging (log_takeover_channels). Off, every
command is exactly as before; on, the flags reach the right process, are
defined in train_aqi_rlpd.py, and pipeline (HiRIL / E2HiL) arms refuse them.
No JAX needed.

Run from the repo root with
    python -m pytest tests/test_campaign_switches.py -q
    python tests/test_campaign_switches.py
"""
import re
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for _p in ("serl_launcher", "examples"):
    sys.path.insert(0, str(REPO / _p))

import run_campaign as rc                     # noqa: E402

CAMPAIGNS = REPO / "examples" / "campaigns"
NEW_FLAGS = {"converge_mode", "converge_eval_every", "converge_eval_consecutive",
             "utd_ratio", "log_takeover_channels"}


def _parse(config, extra=(), yaml_text=None):
    if yaml_text is not None:
        d = Path(tempfile.mkdtemp())
        base = (CAMPAIGNS / config).read_text(encoding="utf-8")
        (d / "c.yaml").write_text(base + "\n" + yaml_text, encoding="utf-8")
        config = d / "c.yaml"
    else:
        config = CAMPAIGNS / config
    argv = ["--campaign", "unused", "--config", str(config), *extra]
    parser = rc.build_arg_parser()
    args = parser.parse_args(argv)
    cfg = rc.load_config(str(config))
    args.ops_per_tier_map = rc.apply_config(
        args, cfg, rc._explicit_dests(parser, argv)) or None
    return args


def _cmds(args, arm):
    ckpt = Path("runs/x/checkpoints/c")
    return (rc._learner_cmd(args, ckpt, Path("d.pkl"), arm, 0, "r"),
            rc._actor_cmd(args, ckpt, arm, "expert", "exp00", Path("t.json"),
                          0, "r"))


def _used(cmd):
    return {t[2:] for t in cmd if isinstance(t, str) and t.startswith("--")}


def _defined():
    text = (REPO / "examples" / "train_aqi_rlpd.py").read_text(encoding="utf-8")
    return set(re.findall(r'flags\.DEFINE_\w+\(\s*"(\w+)"', text))


def test_off_by_default_commands_unchanged():
    for config in ("rq1_baseline.yaml", "rq4_ensemble_convergence.yaml",
                   "rotfix_pilot_base.yaml"):
        args = _parse(config)
        assert rc._platform_switches(args) == []
        for arm in args.methods.split(","):
            lcmd, acmd = _cmds(args, arm)
            assert not (_used(lcmd) | _used(acmd)) & NEW_FLAGS, (config, arm)


def test_cli_switches_reach_the_right_process():
    args = _parse("rq1_baseline.yaml", extra=[
        "--converge_mode", "eval", "--converge_eval_threshold", "0.6",
        "--converge_eval_every", "1500", "--converge_eval_consecutive", "3",
        "--utd_ratio", "1.0", "--log_takeover_channels"])
    arm = args.methods.split(",")[0]
    lcmd, acmd = _cmds(args, arm)
    a, l_ = " ".join(acmd), " ".join(lcmd)
    assert "--converge_mode eval --converge_eval_every 1500 " \
           "--converge_eval_consecutive 3" in a
    assert "--converge_eval_threshold 0.6" in a
    assert "--utd_ratio 1.0" in a and "--utd_ratio 1.0" in l_
    assert "--log_takeover_channels" in a
    assert "converge_mode" not in l_ and "log_takeover" not in l_
    defined = _defined()
    for cmd in (lcmd, acmd):
        used = _used(cmd)
        booleans = {u[2:] for u in used if u.startswith("no") and u[2:] in defined}
        unknown = used - defined - {f"no{b}" for b in booleans}
        assert not unknown, f"flags not defined in train_aqi_rlpd.py: {unknown}"


def test_yaml_keys():
    args = _parse("rq1_baseline.yaml", yaml_text=(
        "budget:\n  converge_mode: eval\n  converge_eval_threshold: 0.6\n"
        "  converge_eval_every: 2500\n  converge_eval_consecutive: 2\n"
        "  utd_ratio: 0.5\n"
        "logging:\n  takeover_channels: true\n"))
    assert args.converge_mode == "eval" and args.converge_eval_every == 2500
    assert args.utd_ratio == 0.5 and args.log_takeover_channels is True
    assert sorted(rc._platform_switches(args)) == [
        "converge_mode", "log_takeover_channels", "utd_ratio"]
    # an explicit CLI value wins over the YAML
    args = _parse("rq1_baseline.yaml", extra=["--converge_mode", "rolling"],
                  yaml_text="budget:\n  converge_mode: eval\n")
    assert args.converge_mode == "rolling"


def test_pipeline_arms_refuse_the_switches():
    for extra in (["--utd_ratio", "1.0"], ["--log_takeover_channels"],
                  ["--converge_mode", "eval", "--converge_eval_threshold", "0.6"]):
        args = _parse("hiril_comparison.yaml", extra=extra)
        try:
            _cmds(args, "hiril")
        except SystemExit as exc:
            assert "pipeline" in str(exc)
        else:
            raise AssertionError(f"hiril arm accepted {extra}")
    _cmds(_parse("hiril_comparison.yaml"), "hiril")      # off: fine


def test_run_readme_names_the_convergence_rule():
    import provenance as prov
    rolling = prov.render_run_readme({"termination": {
        "converged": True, "step": 9000, "rolling_success": 0.9, "window": 20,
        "autonomous": True}})
    assert "rolling autonomous success 0.9 over 20 episodes" in rolling
    ev = prov.render_run_readme({"termination": {
        "converged": True, "step": 9000, "rolling_success": None, "window": None,
        "autonomous": None, "mode": "eval", "mini_eval_threshold": 0.6,
        "consecutive": 2}})
    assert "mini-eval SR >= 0.6 in 2 evaluations in a row" in ev
    assert "None" not in ev


def test_invalid_values_fail_before_any_work():
    d = tempfile.mkdtemp()
    for extra in (["--converge_mode", "eval"],                 # no threshold
                  ["--converge_mode", "eval", "--converge_eval_threshold", "0.6",
                   "--converge_eval_consecutive", "0"],
                  ["--utd_ratio", "-1"]):
        try:
            rc.main(["--campaign", str(Path(d) / "c"), "--plan", *extra])
        except SystemExit as exc:
            assert "[config]" in str(exc)
        else:
            raise AssertionError(f"accepted {extra}")
    assert not (Path(d) / "c").exists()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
