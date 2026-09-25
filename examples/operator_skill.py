"""Operator skill assessment CLI (methodology doc Part B).

Two subcommands (via --mode):

  assess             one feature vector + tier per operator from real
                     or synthetic interaction data. Inputs: --log_root
                     with one AQI-log run directory per operator (dir
                     name = operator id), and/or --demo_glob matching
                     collect_synthetic_demos outputs. Runs Method A
                     (composite, tertile or --cuts), Method B (k-means,
                     only if >= 8 operators; small-cohort warning under
                     15 per doc S10.3), Method D (Beta-Bernoulli per
                     operator), and Method C when --labels_json gives
                     ground-truth tiers. Writes skill_report.md + .json
                     into --out_dir.

  validate-synthetic the doc S11 closed loop on the toy study: jittered
                     tier cohort -> full pipeline -> accuracy / macro-F1
                     / QWK / Spearman rho(S, s) / episodes-needed curve
                     / test-retest ICC, written to --out_dir. This is
                     the gate that must pass before the pipeline is
                     trusted on humans (and its thresholds are the
                     re-anchoring caveat's reminder).

Examples:
    python operator_skill.py --mode assess \
        --log_root ./campaigns/c1/aqi_logs --out_dir ./skill_c1
    python operator_skill.py --mode validate-synthetic --out_dir ./val
"""

import glob
import json
import os
import pickle as pkl

import numpy as np
from absl import app, flags

from serl_launcher.aqi import skill as SK
from serl_launcher.aqi.operator_model import (jitter_theta,
                                              make_operator_theta,
                                              theta_from_scale)
from serl_launcher.aqi.toy_insertion import generate_population

FLAGS = flags.FLAGS
flags.DEFINE_string("mode", "assess", "assess | validate-synthetic")
flags.DEFINE_string("log_root", "", "dir of per-operator AQI-log runs")
flags.DEFINE_string("demo_glob", "", "glob of demo pkls (operator id ="
                    " file stem); *_groundtruth.json sidecars are used"
                    " for success labels when present")
flags.DEFINE_string("labels_json", "", "optional {operator: tier} truth"
                    " for Method C (tiers: novice/intermediate/expert)")
flags.DEFINE_string("out_dir", "./skill_report", "output directory")
flags.DEFINE_list("cuts", None, "optional composite cuts 'lo,hi' from"
                  " anchor operators (skewed cohorts, doc S10.2)")
flags.DEFINE_float("budget_s", 15.0, "episode budget for censoring")
flags.DEFINE_string("pos_slice", "0:3", "state slice for positions in"
                    " AQI logs, e.g. 0:3")
# validate-synthetic knobs
flags.DEFINE_integer("val_ops_per_tier", 10, "")
flags.DEFINE_integer("val_episodes", 12, "")
flags.DEFINE_integer("val_seed", 3, "")

TIERS = ["novice", "intermediate", "expert"]


def _demo_episode_rows(pkl_path, budget_s):
    """Rebuild per-episode trajectories from a demo pkl (episode ends
    where dones==1) + sidecar ground truth when present."""
    with open(pkl_path, "rb") as f:
        transitions = pkl.load(f)
    side = os.path.splitext(pkl_path)[0] + "_groundtruth.json"
    rows, P = [], []
    for tr in transitions:
        obs = tr["observations"]
        state = obs["state"] if isinstance(obs, dict) else obs
        P.append(np.asarray(state, dtype=np.float64)[:3]
                 if len(np.shape(state)) == 1 else np.asarray(state)[0][:3])
        if tr.get("dones", 0):
            arr = np.array(P)
            rows.append(SK.episode_features(
                dict(positions=arr[:, :2] if arr.shape[1] >= 2 else arr,
                     success=bool(tr["infos"].get("succeed", False)),
                     attempts=1),
                budget_s=budget_s))
            P = []
    meta = {}
    if os.path.exists(side):
        with open(side) as f:
            meta = json.load(f)
    return rows, meta


def cmd_assess():
    os.makedirs(FLAGS.out_dir, exist_ok=True)
    lo, hi = (int(x) for x in FLAGS.pos_slice.split(":"))
    vecs, extra = {}, {}
    if FLAGS.log_root:
        for run in sorted(glob.glob(os.path.join(FLAGS.log_root, "*"))):
            if not os.path.isdir(run):
                continue
            oid = os.path.basename(run)
            r = SK.features_from_aqi_logs(run, pos_slice=(lo, hi),
                                          budget_s=FLAGS.budget_s)
            if not r["episode_rows"]:
                continue
            vecs[oid] = SK.aggregate_operator(
                r["episode_rows"], FLAGS.budget_s, r["interventions"])
            extra[oid] = dict(source="aqi_logs", meta=r["meta"])
    for path in sorted(glob.glob(FLAGS.demo_glob)) if FLAGS.demo_glob \
            else []:
        oid = os.path.splitext(os.path.basename(path))[0]
        rows, meta = _demo_episode_rows(path, FLAGS.budget_s)
        if not rows:
            continue
        vecs[oid] = SK.aggregate_operator(rows, FLAGS.budget_s)
        extra[oid] = dict(source="demos", ground_truth=meta.get("theta"))
    if not vecs:
        raise SystemExit("no operator data found (check --log_root /"
                         " --demo_glob)")

    feats = [f for f in SK.CORE7
             if any(np.isfinite(v.get(f, np.nan)) for v in vecs.values())]
    Z = SK.normalize_features(vecs, feats)
    S = {o: SK.composite_score(Z[o], feats) for o in Z}
    cuts = tuple(float(x) for x in FLAGS.cuts) if FLAGS.cuts else None
    comp = SK.composite_tiers(S, cuts=cuts)

    out = dict(features_used=feats, operators={}, composite=comp)
    for o in sorted(vecs):
        bt = SK.BetaBernoulliTracker(
            1 + vecs[o]["k_success"],
            1 + vecs[o]["n_episodes"] - vecs[o]["k_success"])
        out["operators"][o] = dict(
            features={k: v for k, v in vecs[o].items()
                      if isinstance(v, (int, float)) and np.isfinite(v)},
            z={f: Z[o][f] for f in feats}, composite=S[o],
            tier_composite=comp["tiers"][o],
            beta_mean=bt.mean, beta_ci90=bt.ci(0.90),
            tier_beta=bt.tier(), **extra.get(o, {}))
    if len(vecs) >= 8:
        cl = SK.cluster_operators(Z, feats,
                                  sr={o: vecs[o]["sr"] for o in vecs},
                                  rng=np.random.default_rng(0))
        out["clustering"] = dict(tiers=cl["tiers"],
                                 stability=cl["stability"],
                                 small_cohort_warning=cl[
                                     "small_cohort_warning"])
    if FLAGS.labels_json:
        with open(FLAGS.labels_json) as f:
            labels = {o: TIERS.index(t) for o, t in json.load(f).items()
                      if o in Z}
        if len(labels) >= 6:
            out["loso"] = SK.loso_classify(
                {o: Z[o] for o in labels}, labels, feats)

    with open(os.path.join(FLAGS.out_dir, "skill_report.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    _write_md(out, vecs, feats)
    print(f"skill report -> {FLAGS.out_dir}/skill_report.md")


def _write_md(out, vecs, feats):
    lines = ["# Operator skill report", "",
             f"Features used: {', '.join(feats)}",
             f"Composite cuts: {np.round(out['composite']['cuts'], 3)}",
             "", "| operator | n | SR | KM med T | composite S | tier "
             "(A) | Beta mean [90% CI] | tier (D) |",
             "|---|---|---|---|---|---|---|---|"]
    for o, d in sorted(out["operators"].items()):
        v = vecs[o]
        ci = d["beta_ci90"]
        lines.append(
            f"| {o} | {v['n_episodes']} | {v['sr']:.2f} | "
            f"{v['t_km_median']:.1f} | {d['composite']:+.2f} | "
            f"{d['tier_composite']} | {d['beta_mean']:.2f} "
            f"[{ci[0]:.2f},{ci[1]:.2f}] | {d['tier_beta']} |")
    if "clustering" in out:
        lines += ["", f"k-means tiers (stability "
                  f"{out['clustering']['stability']:.2f}"
                  + (", SMALL-COHORT WARNING: <15 operators"
                     if out['clustering']['small_cohort_warning'] else "")
                  + "): " + json.dumps(out["clustering"]["tiers"])]
    if "loso" in out:
        l = out["loso"]
        lines += ["", f"LOSO ({l['model']}): acc={l['accuracy']:.2f} "
                  f"macro-F1={l['macro_f1']:.2f} QWK={l['qwk']:.2f}"]
    lines += ["", "Caveats: composite tertiles assume a roughly balanced"
              " cohort; intervention statistics compare only at matched"
              " training phase; synthetic validation licenses ordinal"
              " claims, thresholds re-anchor on human data (doc S11)."]
    with open(os.path.join(FLAGS.out_dir, "skill_report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def cmd_validate():
    os.makedirs(FLAGS.out_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    thetas = []
    for tier in TIERS:
        for j in range(FLAGS.val_ops_per_tier):
            th = jitter_theta(make_operator_theta(tier), rng)
            th.operator_id, th.tier = f"{tier[:3]}{j}", tier
            thetas.append(th)
    pop = generate_population(thetas, FLAGS.val_episodes,
                              seed0=FLAGS.val_seed)
    truth = {t.operator_id: TIERS.index(t.tier) for t in thetas}
    rows = {o: [SK.episode_features(t, budget_s=15.0, goal=(0.0, -20.0))
                for t in tr] for o, tr in pop.items()}
    vecs = {o: SK.aggregate_operator(r, 15.0) for o, r in rows.items()}
    Z = SK.normalize_features(vecs, SK.CORE7)
    S = {o: SK.composite_score(Z[o]) for o in Z}
    comp = SK.composite_tiers(S)
    acc_a = float(np.mean([comp["tiers"][o] == TIERS[truth[o]]
                           for o in S]))
    cl = SK.cluster_operators(Z, sr={o: vecs[o]["sr"] for o in vecs},
                              rng=np.random.default_rng(1))
    acc_b = float(np.mean([cl["tiers"][o] == TIERS[truth[o]] for o in Z]))
    loso = SK.loso_classify(Z, truth)
    sweep = []
    for i, s in enumerate(np.linspace(0, 1, 9)):
        th = theta_from_scale(float(s))
        th.operator_id = f"s{i}"
        sweep.append(th)
    pop_s = generate_population(sweep, FLAGS.val_episodes, seed0=41)
    vec_s = {o: SK.aggregate_operator(
        [SK.episode_features(t, budget_s=15.0, goal=(0.0, -20.0))
         for t in tr], 15.0) for o, tr in pop_s.items()}
    Zs = SK.normalize_features(vec_s, SK.CORE7)
    Ss = {o: SK.composite_score(Zs[o]) for o in Zs}
    rho = SK.spearman_rho([Ss[f"s{i}"] for i in range(9)],
                          list(range(9)))
    half = {o: rows[o][0::2] for o in rows}
    other = {o: rows[o][1::2] for o in rows}

    def scores(rw):
        v = {o: SK.aggregate_operator(r, 15.0) for o, r in rw.items()}
        z = SK.normalize_features(v, SK.CORE7)
        return {o: SK.composite_score(z[o]) for o in z}
    icc = SK.test_retest_icc(scores(half), scores(other))
    rep = dict(n_operators=len(thetas), episodes=FLAGS.val_episodes,
               method_a_accuracy=acc_a, composite_cuts=comp["cuts"],
               method_b_accuracy=acc_b, kmeans_stability=cl["stability"],
               loso=loso, spearman_rho_vs_s=rho, test_retest_icc=icc)
    with open(os.path.join(FLAGS.out_dir, "validation.json"), "w") as f:
        json.dump(rep, f, indent=1, default=float)
    print(json.dumps(rep, indent=1, default=float))
    ok = (acc_b >= 0.9 and loso["qwk"] >= 0.8 and rho >= 0.9
          and icc >= 0.75)
    print(f"\nVALIDATION {'PASSED' if ok else 'BELOW TARGETS'} "
          f"(doc S11 targets: clustering ~0.97, QWK high, rho>=0.9)")


def main(_):
    if FLAGS.mode == "assess":
        cmd_assess()
    elif FLAGS.mode == "validate-synthetic":
        cmd_validate()
    else:
        raise SystemExit(f"unknown --mode {FLAGS.mode}")


if __name__ == "__main__":
    app.run(main)
