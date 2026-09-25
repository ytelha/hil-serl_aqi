"""Tests for the e2hil mode's plumbing: AQIConfig semantics, campaign YAML ->
learner/actor flags, provenance rows, and the intervention-rate analysis.

Run from the repo root with either
    python -m pytest tests/test_campaign_e2hil.py -q
    python tests/test_campaign_e2hil.py
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for _p in ("serl_launcher", "examples"):
    sys.path.insert(0, str(REPO / _p))

import provenance as prov                     # noqa: E402
import run_campaign as rc                     # noqa: E402
from serl_launcher.aqi.config import AQIConfig  # noqa: E402

CONFIG = REPO / "examples" / "campaigns" / "e2hil_comparison.yaml"


def _parse(argv):
    parser = rc.build_arg_parser()
    args = parser.parse_args(argv)
    cfg = rc.load_config(str(CONFIG))
    args.ops_per_tier_map = rc.apply_config(
        args, cfg, rc._explicit_dests(parser, argv)) or None
    return args


def test_e2hil_mode_semantics():
    e2, base = AQIConfig(mode="e2hil"), AQIConfig(mode="hilserl")
    assert e2.baseline_arch and base.baseline_arch
    assert e2.e2hil_on and not base.e2hil_on
    assert not e2.logging_on and not e2.active_on and not e2.quality_on
    assert (e2.e2hil_K, e2.e2hil_q_low, e2.e2hil_q_high, e2.e2hil_q_agg) == (
        8, 0.05, 0.90, "min")
    assert AQIConfig(mode="logged").logging_on
    for bad in (dict(e2hil_q_low=0.9, e2hil_q_high=0.5),
                dict(e2hil_q_agg="max"), dict(e2hil_K=0)):
        try:
            AQIConfig(mode="e2hil", **bad)
        except AssertionError:
            continue
        raise AssertionError(f"accepted invalid {bad}")


def test_yaml_to_commands():
    args = _parse(["--campaign", "unused"])
    assert args.methods == "hilserl,e2hil"
    assert (args.e2hil_K, args.e2hil_q_low, args.e2hil_q_high,
            args.e2hil_q_agg) == (8, 0.05, 0.90, "min")
    ckpt = Path("ckpt")
    for method in ("hilserl", "e2hil"):
        lcmd = rc._learner_cmd(args, ckpt, Path("d.pkl"), method, 0, "r")
        acmd = rc._actor_cmd(args, ckpt, method, "expert", "exp00",
                             Path("t.json"), 0, "r")
        for cmd in (lcmd, acmd):
            s = " ".join(cmd)
            assert f"--aqi_mode {method}" in s
            assert "--e2hil_K 8 --e2hil_q_low 0.05 --e2hil_q_high 0.9 " \
                   "--e2hil_q_agg min" in s
        # both arms plain: no warm start, built-in LR
        assert "--nodemo_pretrain" in lcmd and "--learning_rate" not in lcmd
    # CLI still wins over the YAML
    args = _parse(["--campaign", "unused", "--e2hil_K", "16",
                   "--e2hil_q_agg", "mean"])
    assert args.e2hil_K == 16 and args.e2hil_q_agg == "mean"


def test_provenance_rows():
    def rows(mode):
        return {r["choice"]: r for r in prov.condition_rows(AQIConfig(mode=mode))}
    e2, base = rows("e2hil"), rows("hilserl")
    assert e2["aqi_mode"]["note"].startswith("EXTERNAL baseline E2HiL")
    assert e2["e2hil_K"]["status"] == "active"
    assert base["e2hil_K"]["status"] == "inert"
    assert e2["critic_ensemble_size"]["status"] == "baseline"
    assert e2["training logs"]["value"] == "off"


def _write_cell(campaign, cell_id, episodes):
    d = campaign / "checkpoints" / cell_id
    d.mkdir(parents=True)
    with open(d / "episodes.jsonl", "w") as f:
        for n_int, length in episodes:
            f.write(json.dumps({"intervention_steps": n_int, "length": length,
                                "success": True, "step": 0,
                                "intervention_count": int(n_int > 0)}) + "\n")


def test_intervention_rate_analysis():
    with tempfile.TemporaryDirectory() as tmp:
        camp = Path(tmp)
        # hilserl: 30% and 20% operator steps; e2hil: 10% and 20%
        _write_cell(camp, "hilserl_exp00_s0", [(30, 100), (30, 100)])
        _write_cell(camp, "hilserl_exp00_s1", [(20, 100)])
        _write_cell(camp, "e2hil_exp00_s0", [(10, 100), (10, 100)])
        _write_cell(camp, "e2hil_exp00_s1", [(20, 100)])
        ivn_by_method = {}
        for method in ("hilserl", "e2hil"):
            ivn = {}
            for s in (0, 1):
                v = rc._read_cell_interventions(
                    camp, rc.Cell(method, "exp00", "expert", s))
                ivn[s] = v
            ivn_by_method[method] = {"exp00": {"tier": "expert", "ivn": ivn}}
        assert rc._read_cell_interventions(
            camp, rc.Cell("e2hil", "nov00", "novice", 0)) is None
        out = rc.compute_interventions(ivn_by_method, ["hilserl", "e2hil"],
                                       ["expert"], "hilserl")
        assert abs(out["methods"]["hilserl"]["rate_mean"] - 0.25) < 1e-12
        assert abs(out["methods"]["e2hil"]["rate_mean"] - 0.15) < 1e-12
        pr = out["paired"]["e2hil"]
        assert (pr["n_pairs"], pr["n_method_lower"], pr["n_tied"]) == (2, 1, 1)
        assert abs(pr["mean_delta"] + 0.10) < 1e-12

        report = {"baseline": "hilserl", "methods": {}}
        meta = dict(campaign_dir="c", exp_name="x", peg_shape="cuboid",
                    grid_sha256="0" * 64, n_grid_trials=9, n_operators=1,
                    ops_per_tier=1, n_seeds=2, steps=10)
        md = rc.render_report_md(report, meta, ["expert"], ivn=out)
        assert "## Human effort: intervention rate" in md
        assert "| `e2hil` | 2 |  15.0% |" in md


def test_sign_test_unchanged():
    # the refactored helper must reproduce the previous inline computation
    assert rc._sign_test_p(0, 0) != rc._sign_test_p(0, 0)      # nan
    assert abs(rc._sign_test_p(1, 11) - 0.00634765625) < 1e-12
    assert rc._sign_test_p(3, 3) == 1.0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
