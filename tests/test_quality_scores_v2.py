"""Tests for the quality scores v2 (docs/quality_scores_v2.md):
quality_score_mode hindsight / progress / outcome / uniform, the post-release
window, and the operator-consistency BC weight.

The runtime tests replace the jitted critic calls with a deterministic fake
(V(s_t) = 0.01 t), so they need numpy only. Run from the repo root with
    python -m pytest tests/test_quality_scores_v2.py -q
    python tests/test_quality_scores_v2.py
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
for _p in ("serl_launcher", "examples"):
    sys.path.insert(0, str(REPO / _p))

from serl_launcher.aqi import runtime as rt                        # noqa: E402
from serl_launcher.aqi.config import AQIConfig, QUALITY_SCORE_MODES  # noqa: E402
from serl_launcher.aqi.consistency import OperatorConsistency      # noqa: E402
from serl_launcher.aqi.quality import (                            # noqa: E402
    InterventionSegment, QualityScorer, hindsight_advantage,
    post_release_window)

G = 0.97


# ---------------------------------------------------------------- config
def test_score_modes_validated():
    for m in QUALITY_SCORE_MODES:
        AQIConfig(mode="quality", quality_score_mode=m)
    for bad in [dict(quality_score_mode="hindsite"),
                dict(quality_post_release_steps=-1),
                dict(quality_hindsight_discount=0.0),
                dict(quality_consistency_k=1),
                dict(quality_consistency_sigma_ref=0.0)]:
        try:
            AQIConfig(mode="quality", **bad)
        except AssertionError:
            continue
        raise AssertionError(f"accepted {bad}")


# ------------------------------------------------------ post-release window
def test_window_inside_episode_bootstraps():
    assert post_release_window(5, 5, 30, 10, terminal=True) == (20, True)
    assert post_release_window(5, 5, 30, 0, terminal=False) == (10, True)


def test_window_clipped_at_terminal_and_truncation():
    # terminated: all rewards to the end, nothing after the terminal state
    assert post_release_window(25, 4, 30, 10, terminal=True) == (30, False)
    # truncated: bootstrap from the last state that has a value
    assert post_release_window(25, 4, 30, 10, terminal=False) == (29, True)
    # a one-step takeover on the last step of a truncated episode: A = 0
    assert post_release_window(29, 1, 30, 10, terminal=False) == (29, True)


def test_hindsight_advantage_matches_hand_computation():
    T = 30
    values = 0.01 * np.arange(T)
    rewards = np.zeros(T)
    rewards[-1] = 1.0
    a, t_e = hindsight_advantage(values, rewards, 5, 5, 10, G, terminal=True)
    assert t_e == 20
    assert np.isclose(a, G ** 15 * values[20] - values[5])
    a, t_e = hindsight_advantage(values, rewards, 25, 4, 10, G, terminal=True)
    assert t_e == 30
    assert np.isclose(a, G ** 4 * 1.0 - values[25])
    a, t_e = hindsight_advantage(values, rewards, 25, 4, 10, G, terminal=False)
    assert t_e == 29          # truncated: r_29 is outside, V(s_29) bootstraps
    assert np.isclose(a, G ** 4 * values[29] - values[25])
    a, _ = hindsight_advantage(values, rewards, 29, 1, 10, G, terminal=False)
    assert a == 0.0
    # a missing value estimate propagates as NaN (scored neutral later)
    v = values.copy()
    v[20] = np.nan
    a, _ = hindsight_advantage(v, rewards, 5, 5, 10, G, terminal=True)
    assert np.isnan(a)


# ------------------------------------------------------------- the scorer
def _seg(n, excluded=False):
    s = InterventionSegment(operator_id="op", start_step=0, requested=False)
    s.transitions = [{}] * n
    s.weights = np.zeros(n) if excluded else np.full(n, 0.3)
    s.excluded = excluded
    return s


def test_segment_score_mapping():
    cfg = AQIConfig(mode="quality")
    sc = QualityScorer(cfg)
    s = sc.apply_segment_score(_seg(4), 0.7)
    w = np.clip(1 / (1 + np.exp(-cfg.quality_gamma * 0.7)),
                cfg.quality_w_min, cfg.quality_w_max)
    assert np.allclose(s.q_signed, 0.7) and np.allclose(s.q_int, 0.7)
    assert np.allclose(s.weights, w)
    s = sc.apply_segment_score(_seg(4), -0.7)
    assert np.allclose(s.q_int, 0.0) and np.all(s.weights < 0.5)
    s = sc.apply_segment_score(_seg(4, excluded=True), 5.0)
    assert np.allclose(s.weights, 0.0)             # exclusion survives
    s = sc.apply_segment_score(_seg(3), 0.0)        # "uniform"
    assert np.allclose(s.weights, 0.5) and np.allclose(s.q_int, 0.0)


def test_signed_scales_keep_the_sign():
    sc = QualityScorer(AQIConfig(mode="quality"))
    xs = [0.2, -0.1, 0.05, -0.4, 0.3]
    qs = [sc.hindsight_q(x) for x in xs]
    assert all(np.sign(q) == np.sign(x) for q, x in zip(qs, xs))
    assert sc.hindsight_q(float("nan")) == 0.0
    assert sc._hind_scale.count == len(xs)          # NaN did not update
    assert sc.outcome_q(True) == 1.0 and sc.outcome_q(False) == -1.0
    assert sc.progress_q(-0.01) < 0


# ------------------------------------------------------ operator consistency
def _segments(rng, n_seg, noise, n=20, dim=4):
    """n_seg takeovers through the same states; a_H = f(s) + noise."""
    base = np.linspace(0, 1, n)[:, None] * np.ones((1, dim))
    for _ in range(n_seg):
        s = base + 0.01 * rng.normal(size=base.shape)
        a = np.tanh(s[:, :2]) + noise * rng.normal(size=(n, 2))
        yield s, a


def test_consistency_not_ready_until_enough_segments():
    oc = OperatorConsistency(state_dim=4, k=5, min_ref=10)
    rng = np.random.default_rng(0)
    for s, a in _segments(rng, 4, 0.1):
        oc.add_segment(s, a)
    assert len(oc) == 80 and oc.n_segments == 4 and not oc.ready
    sig, c = oc.weights(np.zeros((3, 4)))
    assert np.all(np.isnan(sig)) and np.allclose(c, 1.0)


def test_consistency_separates_consistent_from_noisy_operators():
    rng = np.random.default_rng(1)
    sig = {}
    for noise in (0.02, 0.5):
        oc = OperatorConsistency(state_dim=4, k=5, min_ref=50)
        for s, a in _segments(rng, 12, noise):
            oc.add_segment(s, a)
        q, _ = next(_segments(rng, 1, noise))
        sig[noise] = float(np.mean(oc.sigma_beta(q)))
    assert sig[0.02] < 0.1 < sig[0.5], sig


def test_consistency_uses_one_neighbour_per_segment():
    """ONE long smooth-but-arbitrary segment must not look consistent: the
    across-segment spread is what counts, not the within-trajectory one."""
    rng = np.random.default_rng(2)
    oc = OperatorConsistency(state_dim=2, k=3, min_ref=3)
    s = np.zeros((50, 2))
    oc.add_segment(s, np.zeros((50, 1)))            # 50 identical actions
    assert not oc.ready                             # one segment < k
    for v in (1.0, -1.0):
        oc.add_segment(np.zeros((5, 2)), np.full((5, 1), v))
    assert oc.ready
    sb = oc.sigma_beta(np.zeros((1, 2)))[0]
    # neighbours are {0, +1, -1}: sample std 1.0 (a plain 3-NN would pick
    # three zeros from the long segment and report 0)
    assert np.isclose(sb, 1.0), sb


def test_consistency_weight_formula():
    oc = OperatorConsistency(state_dim=2, k=2, min_ref=2, sigma_ref=0.5)
    oc.add_segment(np.zeros((2, 2)), np.full((2, 1), 0.0))
    oc.add_segment(np.zeros((2, 2)), np.full((2, 1), 1.0))
    sig, c = oc.weights(np.zeros((1, 2)))
    assert np.isclose(sig[0], np.sqrt(0.5))
    assert np.isclose(c[0], 1 / (1 + (np.sqrt(0.5) / 0.5) ** 2))


# ------------------------------------------------ runtime, end to end (fake)
SLICES = {"tcp_pose": (0, 7), "tcp_force": (7, 10), "tcp_torque": (10, 13)}
T = 30
SEGS = [(5, 5), (25, 4)]                     # (onset, length)


def _state(t):
    s = np.zeros(13)
    s[2] = -0.001 * t                        # z goes down: depth = 0.001 t
    s[7] = 1.0                               # |F| = 1 N (no veto)
    return s


def _obs(t):
    return {"state": _state(t).astype(np.float32)}


def _patch():
    rt.make_proprio_slicer = lambda env, keys=None: dict(SLICES)
    # V(s_t) = mu_Q(s_t, a_robot) = 0.01 t, read back from the state's z
    rt.q_stats_single = lambda agent, obs, a: (
        float(-np.asarray(obs["state"])[2] * 10.0), 0.1)
    rt.stack_obs = lambda obs_list: obs_list

    def cf(agent, obs_batch, a_h, a_r):
        n = len(a_h)
        return (np.full(n, 0.05), np.full(n, 0.1),
                np.full(n, 0.3), np.full(n, 0.25))
    rt.counterfactual_advantage_batch_safe = cf


def _run_episode(runtime, rng, terminal=True, human_noise=0.0,
                 timeout_done=True):
    """terminal = the episode succeeds on its last step. timeout_done: a
    failed episode also ends with done=True / mask 0, as in the sim and
    franka_env (time limit = done); False models a `truncated` time limit."""
    runtime.begin_episode()
    in_seg = {t for t0, n in SEGS for t in range(t0, t0 + n)}
    for t in range(T):
        obs, nxt = _obs(t), _obs(t + 1)
        a_r = np.zeros(6)
        runtime.observe_step(None, obs, a_r, intervening=False)
        last = t == T - 1
        done = bool(last and (terminal or timeout_done))
        a = (0.5 + human_noise * rng.normal(size=6)) if t in in_seg else a_r
        tr = dict(observations=obs, actions=a, next_observations=nxt,
                  rewards=1.0 if (last and terminal) else 0.0,
                  masks=0.0 if done else 1.0,
                  dones=done)
        runtime.record_transition(obs, tr, a_r, intervening=t in in_seg,
                                  requested=False, hard_violation=False)
    return runtime.end_episode(None, success=terminal, ep_return=1.0)


def _runtime(tmp, **kw):
    _patch()
    cfg = AQIConfig(log_dir=tmp, quality_filters=("safety_veto",), **kw)
    # meta.json config as train_aqi_rlpd.py writes it
    return rt.AQIRuntime(cfg, env=None, operator_id="op", run_name="r",
                         max_episode_length=T,
                         config_dict_for_log={**vars(cfg), "arm": cfg.mode})


def _read(tmp, name):
    with open(Path(tmp) / "r" / name) as f:
        return [json.loads(line) for line in f]


def test_logged_mode_logs_post_release_features_and_rewards():
    with tempfile.TemporaryDirectory() as tmp:
        r = _runtime(tmp, mode="logged")
        _run_episode(r, np.random.default_rng(0))
        r.close()
        segs = _read(tmp, "segments.jsonl")
        f0, f1 = segs[0]["features"], segs[1]["features"]
        v = 0.01 * np.arange(T)
        assert np.isclose(f0["a_hind"], G ** 15 * v[20] - v[5])
        assert f0["hind_window"] == 15 and f0["progress_window"] == 15
        assert np.isclose(f0["d_progress_post"], 0.001 * (20 - 5))
        assert np.isclose(f0["v_onset"], v[5])
        # 2nd takeover runs into the terminal success: no bootstrap, and
        # the progress window ends at the FINAL next-state (t = 30)
        assert np.isclose(f1["a_hind"], G ** 4 - v[25])
        assert f1["hind_window"] == 5
        assert np.isclose(f1["d_progress_post"], 0.001 * (30 - 25))
        assert "sigma_beta_mean" not in segs[0]      # consistency off
        steps = _read(tmp, "steps.jsonl")
        assert len(steps) == T
        assert steps[-1]["reward"] == 1.0 and steps[-1]["mask"] == 0.0
        assert steps[0]["reward"] == 0.0 and steps[0]["mask"] == 1.0


def _scored(tmp, **kw):
    r = _runtime(tmp, mode="quality", **kw)
    out = _run_episode(r, np.random.default_rng(0))
    r.close()
    return out, _read(tmp, "segments.jsonl")


def test_quality_modes_drive_q_and_w():
    for mode, sign0, sign1 in (("hindsight", +1, +1), ("progress", +1, +1),
                               ("outcome", +1, +1)):
        with tempfile.TemporaryDirectory() as tmp:
            out, segs = _scored(tmp, quality_score_mode=mode)
            assert len(out) == 9                       # 5 + 4 transitions
            q0, q1 = segs[0]["q_int"][0], segs[1]["q_int"][0]
            assert (q0 > 0) == (sign0 > 0) and (q1 > 0) == (sign1 > 0), mode
            assert len(set(segs[0]["weights"])) == 1   # one score / segment
    with tempfile.TemporaryDirectory() as tmp:
        out, segs = _scored(tmp, quality_score_mode="uniform")
        assert all(t["q_int"] == 0.0 for t in out)
        assert all(np.isclose(t["quality_weight"], 0.5) for t in out)
    with tempfile.TemporaryDirectory() as tmp:   # counterfactual untouched
        out, segs = _scored(tmp, quality_score_mode="counterfactual")
        assert segs[0]["a_hat_mean"] == 0.05
        g = 1 / (1 + (0.1 / 0.1) ** 2)             # sigma_ref = mean sigma
        assert np.isclose(segs[0]["q_int"][0], 0.05 * g)


def test_hindsight_q_is_negative_for_a_harmful_takeover():
    """A takeover after which the robot sits in a LOWER-value state."""
    with tempfile.TemporaryDirectory() as tmp:
        r = _runtime(tmp, mode="quality", quality_score_mode="hindsight",
                     quality_post_release_steps=0)
        # V(s_t) = 1 - 0.01 t: every step deeper into the episode is worse
        rt.q_stats_single = lambda agent, obs, a: (
            float(1.0 + np.asarray(obs["state"])[2] * 10.0), 0.1)
        _run_episode(r, np.random.default_rng(0), terminal=False)
        r.close()
        segs = _read(tmp, "segments.jsonl")
        assert segs[0]["features"]["a_hind"] < 0
        assert segs[0]["q_int"][0] == 0.0 and segs[0]["weights"][0] < 0.5


def test_behavioral_mode_is_unchanged():
    """Refactor guard: the behavioral branch equals the pre-v2 code."""
    with tempfile.TemporaryDirectory() as tmp:
        _, segs = _scored(tmp, quality_score_mode="behavioral")
        cfg = AQIConfig(mode="quality", quality_score_mode="behavioral")
        ref = QualityScorer(cfg)
        for s in segs:
            f = s["features"]
            q = ref.behavioral_q(s["length"], f["d_value"], f["f_peak"])
            w = float(np.clip(1.0 / (1.0 + np.exp(-np.clip(
                cfg.quality_gamma * q, -30, 30))),
                cfg.quality_w_min, cfg.quality_w_max))
            assert s["q_int"] == [max(q, 0.0)] * s["length"]
            assert s["weights"] == [w] * s["length"]


def test_consistency_weight_scales_bc_weight_only():
    with tempfile.TemporaryDirectory() as tmp:
        r = _runtime(tmp, mode="quality", quality_score_mode="uniform",
                     quality_consistency_weight=True,
                     quality_consistency_k=2, quality_consistency_min_ref=5)
        rng = np.random.default_rng(3)
        _run_episode(r, rng, human_noise=0.3)      # refs only after scoring
        out = _run_episode(r, rng, human_noise=0.3)
        r.close()
        segs = _read(tmp, "segments.jsonl")
        assert segs[0]["consistency_ready"] is False    # 1st episode: no refs
        assert segs[0]["consistency_mean"] == 1.0
        late = segs[2:]
        assert all(s["consistency_ready"] for s in late)
        assert all(0.0 < s["consistency_mean"] < 1.0 for s in late)
        assert all(t["q_int"] == 0.0 for t in out)      # priority untouched
        assert all(t["quality_weight"] < 0.5 for t in out)


def test_consistency_is_off_outside_quality_modes():
    with tempfile.TemporaryDirectory() as tmp:
        r = _runtime(tmp, mode="logged", quality_consistency_weight=True)
        assert r.consistency is None
        r.close()


# --------------------------------------- offline audit (PART C) vs online
def _audit_rows(tmp, **kw):
    import quality_signal_audit as qa
    return qa.load_segments(tmp, max_episode_steps=T, **kw)


def _set_meta(tmp, **cfg):
    p = Path(tmp) / "r" / "meta.json"
    meta = json.loads(p.read_text())
    meta["config"].update(cfg)
    p.write_text(json.dumps(meta))


def test_audit_recomputes_post_release_scores_like_the_runtime():
    with tempfile.TemporaryDirectory() as tmp:
        r = _runtime(tmp, mode="logged")
        _run_episode(r, np.random.default_rng(0))
        r.close()
        meta = json.loads((Path(tmp) / "r" / "meta.json").read_text())
        assert meta["config"]["proprio_slices"]["tcp_pose"] == [0, 7]
        online = _audit_rows(tmp)
        assert [x["post_source"] for x in online] == ["online"] * 2
        # a different logged K -> the audit must recompute from steps.jsonl
        _set_meta(tmp, quality_post_release_steps=99)
        offline = _audit_rows(tmp)
        assert [x["post_source"] for x in offline] == ["offline"] * 2
        for on, off in zip(online, offline):
            assert np.isclose(on["a_hind"], off["a_hind"])
            assert np.isclose(on["v_onset"], off["v_onset"])
        # progress: same window inside the episode; at the terminal end the
        # offline window stops at the last LOGGED state (t = 29, not 30)
        assert np.isclose(online[0]["d_progress_post"],
                          offline[0]["d_progress_post"])
        assert np.isclose(offline[1]["d_progress_post"], 0.001 * (29 - 25))
        assert [x["window_hits_end"] for x in offline] == [False, True]
        # pre-v2 logs have no reward/mask: sparse reward is inferred
        steps = Path(tmp) / "r" / "steps.jsonl"
        lines = [json.loads(s) for s in steps.read_text().splitlines()]
        for s in lines:
            s.pop("reward"), s.pop("mask")
        steps.write_text("".join(json.dumps(s) + "\n" for s in lines))
        old = _audit_rows(tmp)
        for on, o in zip(online, old):
            assert np.isclose(on["a_hind"], o["a_hind"])


def test_failed_episode_end_is_terminal_online_and_offline():
    """Time limit = done (mask 0) in this codebase: a window reaching a
    FAILED episode's end gets no bootstrap, online and in the pre-v2
    offline inference alike."""
    with tempfile.TemporaryDirectory() as tmp:
        r = _runtime(tmp, mode="logged")
        _run_episode(r, np.random.default_rng(0), terminal=False)
        r.close()
        seg1 = _read(tmp, "segments.jsonl")[1]["features"]
        v = 0.01 * np.arange(T)
        assert np.isclose(seg1["a_hind"], 0.0 - v[25])       # no bootstrap
        online = _audit_rows(tmp)
        steps = Path(tmp) / "r" / "steps.jsonl"
        lines = [json.loads(s) for s in steps.read_text().splitlines()]
        assert lines[-1]["mask"] == 0.0 and lines[-1]["reward"] == 0.0
        for s in lines:
            s.pop("reward"), s.pop("mask")
        steps.write_text("".join(json.dumps(s) + "\n" for s in lines))
        _set_meta(tmp, quality_post_release_steps=99)
        old = _audit_rows(tmp)
        for on, o in zip(online, old):
            assert np.isclose(on["a_hind"], o["a_hind"])
    with tempfile.TemporaryDirectory() as tmp:  # a `truncated` env: bootstrap
        r = _runtime(tmp, mode="logged")
        _run_episode(r, np.random.default_rng(0), terminal=False,
                     timeout_done=False)
        r.close()
        seg1 = _read(tmp, "segments.jsonl")[1]["features"]
        assert np.isclose(seg1["a_hind"], G ** 4 * v[29] - v[25])


def test_audit_sigma_beta_replays_the_online_estimate():
    with tempfile.TemporaryDirectory() as tmp:
        r = _runtime(tmp, mode="quality", quality_score_mode="uniform",
                     quality_consistency_weight=True,
                     quality_consistency_k=2, quality_consistency_min_ref=5)
        rng = np.random.default_rng(4)
        for _ in range(3):
            _run_episode(r, rng, human_noise=0.3)
        r.close()
        segs = _read(tmp, "segments.jsonl")
        rows = _audit_rows(tmp, consistency_k=2, consistency_min_ref=5,
                           consistency_queries=100)
        assert len(rows) == len(segs) == 6
        for s, row in zip(segs, rows):
            if s["sigma_beta_mean"] is None:
                assert row["sigma_beta"] is None
            else:
                assert np.isclose(s["sigma_beta_mean"], row["sigma_beta"])
        assert rows[0]["sigma_beta"] is None and rows[-1]["sigma_beta"] > 0


def test_audit_report_has_part_c():
    import quality_signal_audit as qa
    with tempfile.TemporaryDirectory() as tmp:
        r = _runtime(tmp, mode="logged")
        rng = np.random.default_rng(5)
        for i in range(6):
            _run_episode(r, rng, terminal=bool(i % 2))
        r.close()
        out = qa.make_audit(log_root=tmp, out=str(Path(tmp) / "audit"),
                            max_episode_steps=T, n_boot=50, plots=False)
        res = json.loads((Path(out) / "audit.json").read_text())
        g = res["arms"]["logged"]
        for name in ("a_hind (hindsight)", "d_progress_post",
                     "V(onset) (difficulty only)"):
            e = g["auroc"][name]
            assert e["n"] == 12 and "auroc_difficulty" in e, name
            assert "auroc_window_clean" in e, name
        assert "a_hind (hindsight)" in g["corr_with_v_onset"]
        text = (Path(out) / "summary.md").read_text()
        assert "difficulty-stratified" in text and "C2." in text
        assert "Post-release scores (C1)" in text


# ----------------------------------------------------------- campaign YAMLs
def _parse(config):
    import run_campaign as rc
    argv = ["--campaign", "unused", "--config",
            str(REPO / "examples" / "campaigns" / config)]
    parser = rc.build_arg_parser()
    args = parser.parse_args(argv)
    cfg = rc.load_config(argv[-1])
    args.ops_per_tier_map = rc.apply_config(
        args, cfg, rc._explicit_dests(parser, argv)) or None
    return rc, args


def _actor(rc, args, arm):
    return rc._actor_cmd(args, Path("ckpt"), arm, "expert", "exp00",
                         Path("t.json"), 0, "r")


def test_campaign_arms_resolve():
    rc, args = _parse("rq5_quality_score_ext.yaml")
    rc.validate_arms(args, args.methods.split(","))
    for arm, mode in (("quality-hind", "hindsight"),
                      ("quality-prog", "progress"),
                      ("quality-outc", "outcome")):
        cmd = _actor(rc, args, arm)
        assert cmd[cmd.index("--quality_score_mode") + 1] == mode
    cmd = _actor(rc, args, "quality-outc")
    assert "outcome_filter" in cmd[cmd.index("--quality_filters") + 1]
    rc, args = _parse("rq5_quality_imitation.yaml")
    rc.validate_arms(args, args.methods.split(","))
    cons = _actor(rc, args, "quality-beh-bc-cons")
    plain = _actor(rc, args, "quality-beh-bc")
    assert "--quality_consistency_weight" in cons
    assert "--quality_consistency_weight" not in plain
    for arm in ("quality-unif-bc", "quality-beh-bc", "quality-beh-bc-cons"):
        cmd = _actor(rc, args, arm)
        assert cmd[cmd.index("--bc_loss_weight") + 1] == "0.1", arm
    cmd = _actor(rc, args, "logged")
    assert cmd[cmd.index("--bc_loss_weight") + 1] == "0.0"


def test_campaign_rejects_unknown_score_mode():
    rc, args = _parse("rq5_quality_score_ext.yaml")
    args.arms = dict(args.arms)
    args.arms["quality-hind"] = {"mode": "quality",
                                 "quality_score_mode": "hindsite"}
    try:
        rc.validate_arms(args, ["quality-hind"])
    except SystemExit as exc:
        assert "hindsite" in str(exc)
        return
    raise AssertionError("accepted an unknown quality_score_mode")


def test_provenance_marks_consistency_inert_without_bc():
    import provenance as prov
    rows = {r["choice"]: r for r in prov.condition_rows(AQIConfig(
        mode="quality", quality_consistency_weight=True, bc_loss_weight=0.0))}
    assert rows["quality_consistency_weight"]["status"] == "inert"
    rows = {r["choice"]: r for r in prov.condition_rows(AQIConfig(
        mode="quality", quality_consistency_weight=True, bc_loss_weight=0.1))}
    assert rows["quality_consistency_weight"]["status"] == "active"
    rows = {r["choice"]: r for r in prov.condition_rows(AQIConfig(
        mode="logged"))}
    assert rows["quality_post_release_steps"]["status"] == "active"
    assert "LOGGED" in rows["quality_post_release_steps"]["note"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
