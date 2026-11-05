"""Tests for the inconsistent synthetic operator and the discard filters
(operator_model.Inconsistency, operator_gate, aqi/runtime._discard,
examples/inconsistent_operator_audit.py): the hidden condition's share and
persistence, what the inattentive condition changes, that the label reaches the
segments, the score / oracle / random filters, the campaign wiring, and the
audit's AUROCs on synthetic rows. Off, the operator is the original one.

Without gymnasium (this PC) the stand-in of test_takeover_channels.py is used.

Run from the repo root with
    python -m pytest tests/test_inconsistent_operator.py -q
    python tests/test_inconsistent_operator.py
"""
import contextlib
import io
import runpy
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parent.parent
for _p in ("serl_launcher", "examples"):
    sys.path.insert(0, str(REPO / _p))
_tc = runpy.run_path(str(REPO / "tests" / "test_takeover_channels.py"), run_name="takeover_channels_helpers")

from serl_launcher.aqi.config import AQIConfig, DISCARD_FILTERS, QUALITY_FILTERS  # noqa: E402
from serl_launcher.aqi.operator_gate import MonitoringSignal, SyntheticOperatorWrapper  # noqa: E402
from serl_launcher.aqi.operator_model import (BAD_AIM_OFF, BAD_P_EARLY, EXEC_ANCHORS,  # noqa: E402
                                              Inconsistency, PhaseOracle, SyntheticOperator,
                                              inattentive_theta, make_operator_theta)
from serl_launcher.aqi.quality import InterventionSegment  # noqa: E402
from serl_launcher.aqi.runtime import AQIRuntime  # noqa: E402
from serl_launcher.aqi.toy_insertion import ToyInsertionOracle  # noqa: E402


_OBS = {"state": np.zeros(8)}           # at rest above the hole: no contact, no jam


def _op(inc=None, tier="expert", seed=0):
    return SyntheticOperator(ToyInsertionOracle(), make_operator_theta(tier), rng=np.random.default_rng(seed),
                             inconsistency=inc)


def test_off_is_the_original_operator():
    op = _op()
    assert op.inconsistency is None and op.condition is None
    with contextlib.suppress(ValueError):
        op.force_condition("inattentive")
        raise AssertionError("force_condition without the inconsistent operator")


def test_condition_share_and_persistence():
    op = _op(Inconsistency(bad_share=0.4, persistence=0.7), seed=3)
    seq = []
    for _ in range(20000):
        op.begin_takeover(_OBS)
        seq.append(op.condition == "inattentive")
    seq = np.asarray(seq)
    assert abs(seq.mean() - 0.4) < 0.03
    stay = np.mean(seq[1:] == seq[:-1])
    assert abs(stay - (0.7 + 0.3 * (0.4 ** 2 + 0.6 ** 2))) < 0.02     # 0.856


def test_condition_sequence_depends_on_seed_and_operator():
    def seq(tier, op_id, seed, n=200):
        th = make_operator_theta(tier, operator_id=op_id)
        op = SyntheticOperator(ToyInsertionOracle(), th, rng=np.random.default_rng(seed + 17),
                               inconsistency=Inconsistency())
        out = []
        for _ in range(n):
            op.begin_takeover(_OBS)
            out.append(op.condition)
        return out
    a = seq("expert", "exp00", 0)
    assert a == seq("expert", "exp00", 0)                         # the arms of one operator and seed are paired
    assert a != seq("expert", "exp01", 0)                         # other operators differ (they used to share it)
    assert a != seq("novice", "nov00", 0) and a != seq("expert", "exp00", 1)


def test_inattentive_is_sloppy_and_wrong_but_monitors_the_same():
    th = make_operator_theta("expert")
    bad = inattentive_theta(th)
    for k in ("sigma0", "sigma_tr", "v_max", "lag", "q_rec", "p_pause"):
        assert getattr(bad, k) == EXEC_ANCHORS["novice"][k], k
    assert bad.p_early == BAD_P_EARLY
    for k in ("delta_in", "m_in", "l_react", "p_miss", "rho_fa", "delta_out", "d_max", "operator_id", "tier"):
        assert getattr(bad, k) == getattr(th, k), k
    op = _op(Inconsistency(), seed=1)
    op.force_condition("inattentive")
    assert op.th is op._th_bad and op.ou.sigma == bad.sigma_tr * op._noise_mult
    assert abs(np.linalg.norm(op.bias) - BAD_AIM_OFF * op.oracle.commit_threshold()) < 1e-9
    op.begin_takeover(_OBS)                                       # forced: stays inattentive
    assert op.condition == "inattentive" and op.th is op._th_bad
    op.force_condition("attentive")
    assert op.th is op._th_own and np.linalg.norm(op.bias) < BAD_AIM_OFF * op.oracle.commit_threshold()
    op.force_condition(None)
    op.condition = "inattentive"
    op.reset()                                                      # a new episode: own theta, condition kept
    assert op.th is op._th_own and op.condition == "inattentive"


def test_needs_the_geometric_oracle():
    class Pol:
        def action_dim(self):
            return 2

        def reset(self):
            pass
    try:
        SyntheticOperator(Pol(), make_operator_theta("expert"), inconsistency=Inconsistency())
    except ValueError:
        return
    raise AssertionError("accepted a non-geometric oracle")


def _run_toy(cond, episodes=40, seed=0):
    orc = ToyInsertionOracle()
    op = SyntheticOperator(orc, make_operator_theta("expert"), rng=np.random.default_rng(seed),
                           inconsistency=Inconsistency())
    env = SyntheticOperatorWrapper(_tc["_SucceedKey"](seed=seed), op, MonitoringSignal(orc),
                                   rng=np.random.default_rng(seed + 1), intervention_mode="always_latch")
    op.force_condition(cond)
    wins, conds = 0, set()
    for _ in range(episodes):
        env.reset()
        for _t in range(400):
            _o, _r, term, trunc, info = env.step(np.array([25.0, -25.0]))     # a poor policy: it needs help
            conds.add(info["synthetic_operator"]["condition"])
            if term or trunc:
                wins += int(info.get("succeed", False))
                break
    return wins / episodes, conds


def test_inattentive_corrections_succeed_less_and_info_names_the_condition():
    good, cg = _run_toy("attentive")
    bad, cb = _run_toy("inattentive")
    assert cg == {"attentive"} and cb == {"inattentive"}
    assert good > bad + 0.2, (good, bad)


def test_hiril_wrapper_builds_the_inconsistent_operator():
    """train_hiril.py builds its operator through synthetic_operator_env.wrap_synthetic_operator: with
    inconsistent=True the operator gets the same Inconsistency (share and persistence as passed), and the
    takeover's condition reaches info; off, there is none."""
    from synthetic_operator_env import wrap_synthetic_operator
    for on in (False, True):
        base = _tc["_SucceedKey"](seed=0)
        base.phase_oracle = ToyInsertionOracle()          # the wrapper's geometric oracle
        env, theta = wrap_synthetic_operator(base, synthetic_operator="expert", synthetic_seed=0,
                                             training_oracle="geometric", intervention_mode="always_latch",
                                             inconsistent=on, bad_share=0.2, bad_persistence=0.7,
                                             log=lambda *_a, **_k: None)
        assert theta.tier == "expert"
        if not on:
            assert env.op.inconsistency is None
            continue
        assert env.op.inconsistency.bad_share == 0.2 and env.op.inconsistency.persistence == 0.7
        conds = set()
        env.reset()
        for _t in range(400):
            _o, _r, term, trunc, info = env.step(np.array([25.0, -25.0]))     # a poor policy: it needs help
            conds.add((info.get("synthetic_operator") or {}).get("condition"))
            if term or trunc:
                env.reset()
        assert conds & {"attentive", "inattentive"}, conds
    src = (REPO / "examples" / "train_hiril.py").read_text(encoding="utf-8")
    for needle in ('"synthetic_inconsistent"', '"synthetic_bad_share"', '"synthetic_bad_persistence"',
                   "inconsistent=FLAGS.synthetic_inconsistent", 'condition=s.get("condition")'):
        assert needle in src, needle


def test_off_wrapper_info_has_no_condition():
    orc = ToyInsertionOracle()
    op = SyntheticOperator(orc, make_operator_theta("expert"), rng=np.random.default_rng(0))
    env = SyntheticOperatorWrapper(_tc["_SucceedKey"](seed=0), op, MonitoringSignal(orc), rng=np.random.default_rng(1))
    env.reset()
    _o, _r, _t, _tr, info = env.step(np.zeros(2))
    assert "condition" not in info["synthetic_operator"]


def test_the_label_reaches_the_segment():
    stub = SimpleNamespace(enabled=True, _ep_last_mask=1.0, _ep_reward={}, _ep_step=0, _ep_state_at={},
                           _pending_step_record=None, _open_seg=None, operator_id="exp00", _segments=[],
                           _global_step=0, _extract_ft=lambda state: (None, None),
                           cfg=SimpleNamespace(log_actions=False))
    stub._close_open_segment = types.MethodType(AQIRuntime._close_open_segment, stub)
    obs = {"state": np.zeros((1, 8))}
    tr = {"rewards": 0.0, "masks": 1.0, "actions": np.zeros(2)}
    for t, (cond, intervening) in enumerate([("attentive", True), ("attentive", True), ("attentive", False),
                                              ("inattentive", True), ("inattentive", False)]):
        AQIRuntime.note_operator_condition(stub, cond)
        AQIRuntime.record_transition(stub, obs, tr, np.zeros(2), intervening=intervening, requested=False,
                                     hard_violation=False)
    assert [s.condition for s in stub._segments] == ["attentive", "inattentive"]


def _seg(cond=None, a=0.0):
    s = InterventionSegment(operator_id="x", start_step=0, requested=False, condition=cond)
    s.a_hat = np.array([a, a]); s.weights = np.ones(2)
    return s


def _cfg(filters, frac=0.4):
    return AQIConfig(mode="quality", quality_filters=tuple(filters), quality_drop_fraction=frac,
                     quality_drop_window=200, quality_drop_min_segments=20)


def test_discard_filters():
    # score filter: after 20 segments, drop the ones below the 0.4 quantile of the recent scores
    st = SimpleNamespace(cfg=_cfg(["score_filter"]))
    rng = np.random.default_rng(0)
    dropped = []
    for i in range(400):
        s = _seg(a=float(rng.normal()))
        AQIRuntime._discard(st, s, None, warmup=False)
        dropped.append(s.excluded)
        assert s.drop_score is not None and (not s.excluded or s.exclusion_reason == "score_filter")
        if s.excluded:
            assert not s.weights.any()
    assert not any(dropped[:20]) and abs(np.mean(dropped[20:]) - 0.4) < 0.06
    # a score with two values (outcome): ties at the threshold keep the share
    st2 = SimpleNamespace(cfg=_cfg(["score_filter"]))
    hits, conds = [], []
    for i in range(2000):
        good = bool(rng.random() < 0.5)
        s = _seg(a=0.0)
        AQIRuntime._discard(st2, s, 1.0 if good else -1.0, warmup=False)
        hits.append(s.excluded)
        conds.append(good)
    hits, conds = np.asarray(hits[20:]), np.asarray(conds[20:])
    assert abs(hits.mean() - 0.4) < 0.04 and (hits & conds).mean() < 0.01, (hits.mean(), (hits & conds).sum())
    # segment-level score wins over the counterfactual mean
    s = _seg(a=5.0)
    AQIRuntime._discard(st, s, -9.0, warmup=False)
    assert s.excluded and s.drop_score == -9.0
    # warm-up: scored, never dropped
    s = _seg(a=-50.0)
    AQIRuntime._discard(st, s, None, warmup=True)
    assert not s.excluded
    # oracle: exactly the inattentive ones
    st = SimpleNamespace(cfg=_cfg(["oracle_filter"]))
    for cond, want in (("attentive", False), ("inattentive", True), (None, False)):
        s = _seg(cond)
        AQIRuntime._discard(st, s, None, warmup=False)
        assert s.excluded == want and (not want or s.exclusion_reason == "oracle_filter")
    # random: the same share, independent of the label
    st = SimpleNamespace(cfg=_cfg(["random_filter"], 0.4))
    hits = []
    for i in range(2000):
        s = _seg("inattentive" if i % 2 else "attentive")
        AQIRuntime._discard(st, s, None, warmup=False)
        hits.append(s.excluded)
    assert abs(np.mean(hits) - 0.4) < 0.04
    # no discard filter: nothing happens, nothing is scored
    st = SimpleNamespace(cfg=_cfg([]))
    s = _seg("inattentive", -3.0)
    AQIRuntime._discard(st, s, None, warmup=False)
    assert not s.excluded and s.drop_score is None


def test_config_defaults_unchanged():
    assert AQIConfig().quality_filters == QUALITY_FILTERS
    assert not set(DISCARD_FILTERS) & set(QUALITY_FILTERS)
    for bad in (dict(quality_filters=("nope",)), dict(quality_drop_fraction=1.0)):
        try:
            AQIConfig(**bad)
        except AssertionError:
            continue
        raise AssertionError(f"accepted {bad}")


def test_campaign_wiring():
    import run_campaign as rc
    cfgdir = REPO / "examples" / "campaigns" / "inconsistent_operator"

    def parse(name, extra=()):
        argv = ["--campaign", "unused", "--config", str(cfgdir / f"{name}.yaml"), *extra]
        parser = rc.build_arg_parser()
        args = parser.parse_args(argv)
        args.ops_per_tier_map = rc.apply_config(args, rc.load_config(argv[3]), rc._explicit_dests(parser, argv)) or None
        return args

    def cmds(args, arm):
        ck = Path("runs/x/checkpoints/c")
        return (" ".join(rc._learner_cmd(args, ck, Path("d.pkl"), arm, 0, "r")),
                " ".join(rc._actor_cmd(args, ck, arm, "expert", "exp00", Path("t.json"), 0, "r")))

    args = parse("Q2_pilot_orig")
    rc.validate_arms(args, args.methods.split(","))
    l_, a = cmds(args, "drop-hind")
    assert "--synthetic_inconsistent --synthetic_bad_share 0.4 --synthetic_bad_persistence 0.7" in a
    assert "--quality_filters score_filter" in l_ and "--quality_drop_fraction 0.4" in l_ and "--per_eta 0.0" in l_
    l_, a = cmds(args, "keepall")
    assert "--quality_filters none" in l_ and "--quality_drop_fraction" not in l_
    assert "synthetic_inconsistent" not in l_                      # the operator lives in the actor only
    l_, a = cmds(parse("Q3_consistent_orig"), "drop-hind")
    assert "--synthetic_inconsistent" not in a
    assert "synthetic_inconsistent" in rc._platform_switches(args)
    defined = set(__import__("re").findall(r'flags\.DEFINE_\w+\(\s*"(\w+)"',
                                           (REPO / "examples" / "train_aqi_rlpd.py").read_text(encoding="utf-8")))
    assert {"synthetic_inconsistent", "synthetic_bad_share", "synthetic_bad_persistence",
            "quality_drop_fraction"} <= defined


def _rows(n=600, seed=0, informative="a_hind"):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        att = bool(rng.random() < 0.6)
        r = {k: float(rng.normal()) for k in ("q_int_mean", "a_hat_mean", "a_hat_at_trigger", "d_value", "d_progress",
                                              "f_peak", "jerk_mean", "sigma_mean", "B", "B_no_t", "a_hind",
                                              "d_progress_post", "v_onset", "sigma_beta")}
        r[informative] += 1.5 if att else -1.5
        r.update(run="r0", arm="logged", tier=("expert", "intermediate", "novice")[i % 3], cluster=f"r0#{i // 3}",
                 t_int=float(rng.integers(5, 40)), start_step=float(rng.integers(0, 250)), success=bool(rng.random() < 0.5),
                 condition="attentive" if att else "inattentive")
        r["end_step"] = r["start_step"] + r["t_int"]
        rows.append(r)
    return rows


def test_audit_finds_the_informative_score():
    audit = runpy.run_path(str(REPO / "examples" / "inconsistent_operator_audit.py"), run_name="audit")
    res = audit["analyse"](_rows(), n_boot=100)
    sc = res["scores"]
    assert sc["a_hind (hindsight)"]["auroc"] > 0.9 and sc["a_hind (hindsight)"]["verdict"] == "valid"
    assert abs(sc["a_hat_mean"]["auroc"] - 0.5) < 0.08
    assert res["best_online_score"]["quality_score_mode"] == "hindsight"
    assert not res["label_confounded"]
    assert abs(res["inattentive_share"]["expert"] - 0.4) < 0.1
    text = audit["render"](res, "fake")
    assert "`quality_score_mode: hindsight`" in text
    assert res["discard"] == {}                                     # no discard arm in Q1
    # Q2: the oracle drops exactly the inattentive ones; a random arm drops at the base rate
    base, rows = _rows(seed=1), []
    rng = np.random.default_rng(2)
    for r in base:
        r2 = dict(r, arm="drop-oracle", drop_warmup=False,
                  exclusion_reason="oracle_filter" if r["condition"] == "inattentive" else "")
        r3 = dict(r, arm="drop-random", drop_warmup=False,
                  exclusion_reason="random_filter" if rng.random() < 0.4 else "")
        r4 = dict(r, arm="drop-hind", drop_warmup=True, exclusion_reason="")   # warm-up: not counted
        rows += [r, r2, r3, r4]
    d = audit["analyse"](rows, n_boot=20)["discard"]
    assert set(d) == {"drop-oracle", "drop-random"}
    assert d["drop-oracle"]["precision"] == d["drop-oracle"]["recall"] == 1.0
    assert d["drop-oracle"]["attentive_dropped"] == 0.0 and d["drop-oracle"]["kept_inattentive_share"] == 0.0
    assert abs(d["drop-random"]["precision"] - d["drop-random"]["inattentive_share"]) < 0.07
    res = audit["analyse"](rows, n_boot=20)
    text = audit["render"](res, "fake")
    assert "| drop-oracle |" in text and "| drop-oracle / novice |" in text
    # the within-run/phase/onset AUROC finds the informative score too, and is present per tier
    assert res["scores"]["a_hind (hindsight)"]["auroc_within"]["all"] > 0.9
    assert res["scores"]["a_hind (hindsight)"]["auroc_within"]["novice"] > 0.8
    assert res["discard"]["drop-oracle"]["by_tier"]["novice"]["precision"] == 1.0


def _suite():
    import synthetic_suite as ss
    ss.configure("inconsistent_operator")
    return ss


def test_suite_configs():
    ss = _suite()
    names = ss.ORDER
    assert len(names) == 12 and set(ss.GROUPS["q1"]) | set(ss.GROUPS["q2"]) | set(ss.GROUPS["q3"]) \
        | set(ss.GROUPS["q2confirm"]) | set(ss.GROUPS["twoarm"]) | set(ss.GROUPS["twoarmconfirm"]) == set(names)
    # a platform group runs the pilots; the 108-cell confirmation never starts without a decision
    assert ss.GROUPS["orig"] == ["Q1_validity_orig", "Q2_pilot_orig", "Q3_consistent_orig"]
    assert not any("confirm" in n for n in ss.GROUPS["new"])
    for n in names:
        cfg = ss.load_yaml(ss.config_path(n))
        sw = ss.switches(cfg)
        assert sw["inconsistent"] == (not n.startswith("Q3_")), n
        assert sw["rotation"] == sw["channel"] == n.endswith("_new"), n
        arms = cfg.get("arms") or {}
        quality = {k: v for k, v in arms.items() if v.get("mode") == "quality"}
        if n.startswith("Q1_"):
            assert cfg["methods"] == ["logged"] and not quality
        elif n.startswith("two_arm_"):
            # the two uses of the score alone, defined exactly as in Q2 of the same platform
            assert cfg["methods"] == ["drop-hind", "reweight-hind"] and set(arms) == set(cfg["methods"])
            q2 = ss.load_yaml(ss.config_path("Q2_pilot_" + n.rsplit("_", 1)[-1]))
            # (only the score may differ: it is set from each platform's Q1 audit)
            same = lambda d: {k: v for k, v in d.items() if k != "quality_score_mode"}   # noqa: E731
            assert all(same(arms[a]) == same(q2["arms"][a]) for a in arms), n
            skip = ("methods", "arms", "operators", "replicates", "budget", "logging", "analysis")
            assert {k: v for k, v in cfg.items() if k not in skip} == {k: v for k, v in q2.items() if k not in skip}, n
            assert cfg["analysis"]["baseline"] == "drop-hind"
            confirm = "confirm" in n
            assert cfg["operators"]["expert"] == (3 if confirm else 1)
            assert cfg["replicates"]["seeds"] == (2 if confirm else 1)
            assert cfg["budget"]["steps_cap"] == (40000 if confirm else 30000)
            assert {k: v for k, v in cfg["budget"].items() if k != "steps_cap" and k != "minutes_per_cell"} \
                == {k: v for k, v in q2["budget"].items() if k != "steps_cap" and k != "minutes_per_cell"}, n
        else:
            # the quality arms differ only in what they drop and how they replay
            common = {k for k in set().union(*map(set, quality.values()))
                      if len({str(v.get(k)) for v in quality.values()}) == 1}
            assert {"mode", "quality_score_mode", "quality_warmup_steps"} <= common, (n, common)
            assert {"hilserl", "keepall", "drop-hind", "drop-random", "reweight-hind"} <= set(cfg["methods"])
            assert ("drop-oracle" in cfg["methods"]) == (not n.startswith("Q3_")), n
            assert quality["keepall"]["quality_filters"] == [] and quality["keepall"]["per_eta"] == 0.0
            assert quality["drop-hind"]["quality_drop_fraction"] == quality["drop-random"]["quality_drop_fraction"] \
                == cfg["aqi"]["synthetic_bad_share"]


def _fake_arm_cell(camp, arm, sr, seed=0):
    import json
    cid = f"{arm}_exp00_s{seed}"
    d = camp / "checkpoints" / cid
    d.mkdir(parents=True)
    (d / ".done").write_text(json.dumps({"cell": cid, "method": arm, "operator": "exp00", "tier": "expert",
                                         "seed": seed, "converged": False, "converged_step": None}))
    (d / "episodes.jsonl").write_text(json.dumps({"step": 300, "length": 300, "success": False,
                                                  "intervention_count": 1, "intervention_steps": 50}) + "\n")
    (camp / "eval_runs" / cid).mkdir(parents=True)
    (camp / "eval_runs" / cid / "cell.json").write_text(json.dumps({"outcomes": {"sr": sr}}))


def test_suite_report_pairs_each_arm_once():
    import tempfile
    ss = _suite()
    root = Path(tempfile.mkdtemp(prefix="inc_report_"))
    for arm, sr in (("hilserl", 0.5), ("keepall", 0.6), ("drop-hind", 0.7)):
        _fake_arm_cell(root / "Q2_pilot_orig", arm, sr)
    for arm, sr in (("drop-hind", 0.7), ("reweight-hind", 0.8)):
        _fake_arm_cell(root / "two_arm_pilot_orig", arm, sr)
    with contextlib.redirect_stdout(io.StringIO()):
        assert ss.cmd_report(type("A", (), dict(root=str(root), names=[]))()) == 0
    text = (root / "report.md").read_text(encoding="utf-8")
    assert "| two_arm_pilot_orig | reweight-hind | drop-hind | 1 | 0.10 | 1 / 0 / 0 |" in text
    # a second seed: the per-operator list shows the mean over the seeds, not the last seed
    _fake_arm_cell(root / "two_arm_pilot_orig", "drop-hind", 0.5, seed=1)
    with contextlib.redirect_stdout(io.StringIO()):
        assert ss.cmd_report(type("A", (), dict(root=str(root), names=[]))()) == 0
    text = (root / "report.md").read_text(encoding="utf-8")
    assert "two_arm_pilot_orig / drop-hind: exp00 0.60" in text
    assert "last ckpt SR" in text
    for arm, ref in (("drop-hind", "keepall"), ("hilserl", "keepall"), ("drop-hind", "hilserl")):
        assert f"| Q2_pilot_orig | {arm} | {ref} | 1 |" in text, (arm, ref)
    for arm, ref in (("keepall", "hilserl"), ("keepall", "drop-hind"), ("hilserl", "drop-hind")):
        assert f"| Q2_pilot_orig | {arm} | {ref} |" not in text, (arm, ref)       # the same pair reversed


def test_suite_refuses_the_original_platform_and_keeps_limits_local():
    import json
    import tempfile
    ss = _suite()
    root = Path(tempfile.mkdtemp(prefix="inc_orig_"))
    args = dict(root=str(root), wandb=None, extra=[], dry=True, keep_going=False, default_limits=True,
                without_active=False, score_as_is=True)
    run = lambda name, **kw: ss.cmd_run(type("A", (), dict(args, names=[name], **kw))())   # noqa: E731
    with contextlib.redirect_stdout(io.StringIO()):
        assert run("Q1_validity_orig", allow_orig=False) == 2
        assert run("Q1_validity_orig", allow_orig=True) == 0
        assert run("Q1_validity_new", allow_orig=False) == 0
    # the limits: this suite falls back on the synthetic suite's stage 0; that suite does not fall back at all
    repo, ss.REPO = ss.REPO, root / "repo"
    try:
        p = ss.REPO / "runs" / "synthetic_suite" / "stage0" / "limits.json"
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps({"limits": {"cuboid": {"tilt_rad": 0.2}}}))
        assert ss.read_limits(root, "cuboid") == {"tilt_rad": 0.2}
        ss.configure("synthetic_suite")
        assert ss.read_limits(root, "cuboid") is None
    finally:
        ss.REPO = repo
        ss.configure("inconsistent_operator")


def test_suite_score_guard(tmp=None):
    import json
    import tempfile
    ss = _suite()
    root = Path(tmp or tempfile.mkdtemp(prefix="inc_suite_"))
    assert ss.score_mismatch(root, "Q1_validity_orig") is None
    assert "no Q1 audit" in ss.score_mismatch(root, "Q2_pilot_orig")
    rep = root / "Q1_validity_orig" / "report"
    rep.mkdir(parents=True)
    for best, ok in (("hindsight", True), ("behavioral", False)):
        (rep / "inconsistent_audit.json").write_text(json.dumps({"best_online_score": {"quality_score_mode": best}}))
        assert (ss.score_mismatch(root, "Q2_pilot_orig") is None) == ok, best
        assert (ss.score_mismatch(root, "Q3_consistent_orig") is None) == ok, best
        assert (ss.score_mismatch(root, "two_arm_pilot_orig") is None) == ok, best
    assert "no Q1 audit" in ss.score_mismatch(root, "Q2_pilot_new")          # the platform's own Q1


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
