"""Tests for the per-cell W&B tags of examples/run_campaign.py: each learner
gets AQI_WANDB_TAGS = campaign, arm, base mode (named arms), tier, operator,
seed and the campaign's extra tags -- through the environment, so no training
command changes. Run from the repo root with
    python -m pytest tests/test_campaign_wandb_tags.py -q
    python tests/test_campaign_wandb_tags.py
"""
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for _p in ("serl_launcher", "examples"):
    sys.path.insert(0, str(REPO / _p))

import run_campaign as rc                     # noqa: E402

CAMPAIGNS = REPO / "examples" / "campaigns"


def _parse(config, extra=()):
    argv = ["--campaign", "unused", "--config", str(CAMPAIGNS / config), *extra]
    parser = rc.build_arg_parser()
    args = parser.parse_args(argv)
    cfg = rc.load_config(str(CAMPAIGNS / config))
    args.ops_per_tier_map = rc.apply_config(args, cfg, rc._explicit_dests(parser, argv)) or None
    return args


def _tags(args, campaign, cell):
    env = rc._wandb_tag_env(args, Path(campaign), cell)
    return env["AQI_WANDB_TAGS"].split(",") if env else None


def test_no_tags_when_wandb_is_off():
    args = _parse("rq4_ensemble_convergence.yaml", ["--wandb", "off"])
    assert rc._wandb_tag_env(args, Path("/x/rq4_ensemble_convergence"),
                             rc.Cell("logged", "nov01", "novice", 1)) == {}


def test_plain_mode_cell():
    args = _parse("rq4_ensemble_convergence.yaml", ["--wandb", "online"])
    tags = _tags(args, "/runs/rq4_ensemble_convergence", rc.Cell("logged", "nov01", "novice", 1))
    assert tags == ["rq4_ensemble_convergence", "logged", "novice", "nov01", "seed1"]


def test_named_arm_adds_its_base_mode():
    args = _parse("rq4_attribution_sweep.yaml", ["--wandb", "offline"])
    tags = _tags(args, "/runs/rq4_attribution_sweep", rc.Cell("ens-m1", "int02", "intermediate", 0))
    assert tags == ["rq4_attribution_sweep", "ens-m1", "logged", "intermediate", "int02", "seed0"]


def test_extra_tags_from_cli_deduplicated():
    args = _parse("rq4_ensemble_convergence.yaml", ["--wandb", "online", "--wandb_tags", " B2, RQ2 ,logged,,"])
    tags = _tags(args, "/runs/rq4_ensemble_convergence", rc.Cell("logged", "exp00", "expert", 0))
    assert tags == ["rq4_ensemble_convergence", "logged", "expert", "exp00", "seed0", "B2", "RQ2"]


def test_extra_tags_from_yaml():
    args = _parse("rq4_ensemble_convergence.yaml", ["--wandb", "online"])
    rc.apply_config(args, {"logging": {"wandb_tags": ["D3", "RQ4"]}}, set())
    assert args.wandb_tags == "D3,RQ4"
    tags = _tags(args, "/runs/x", rc.Cell("hilserl", "nov00", "novice", 1))
    assert tags[-2:] == ["D3", "RQ4"]


def test_learner_gets_the_tags_actor_does_not():
    """train_cell passes AQI_WANDB_TAGS in the learner's env (the only process
    that opens W&B) and leaves both command lines untouched."""
    args = _parse("rq4_ensemble_convergence.yaml", ["--wandb", "online"])
    camp = Path(tempfile.mkdtemp()) / "rq4_ensemble_convergence"
    cell = rc.Cell("logged", "nov01", "novice", 1)
    demo = rc._demo_pkl(camp, cell.operator, cell.seed)
    demo.parent.mkdir(parents=True, exist_ok=True)
    demo.write_bytes(b"")
    launched = []

    class _P:
        def poll(self):
            return 1

    def fake_launch(pyargs, log_path, extra_env=None, background=False):
        launched.append((list(pyargs), dict(extra_env or {})))
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        return _P(), open(log_path, "w")

    saved = rc._launch, rc._wait_for_banner, rc._terminate
    rc._launch, rc._wait_for_banner, rc._terminate = fake_launch, (lambda *a, **k: False), (lambda *a, **k: None)
    try:
        rc.train_cell(args, camp, cell)          # learner "never READY" -> returns early
    finally:
        rc._launch, rc._wait_for_banner, rc._terminate = saved
    assert len(launched) == 1                    # only the learner was started
    cmd, env = launched[0]
    assert env["AQI_WANDB_TAGS"] == "rq4_ensemble_convergence,logged,novice,nov01,seed1"
    expected = rc._learner_cmd(args, camp / "checkpoints" / cell.id, demo, cell.method, cell.seed,
                               f"{cell.id}_learner")
    assert cmd == expected                       # the command line is unchanged
    assert not any("wandb_tags" in str(a) or "AQI_WANDB_TAGS" in str(a) for a in cmd)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
