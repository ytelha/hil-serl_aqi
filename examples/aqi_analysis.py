#!/usr/bin/env python3
"""Offline analysis for AQI-HIL-SERL runs (proposal S6.5/S6.6, RQ1-RQ3).

Reads the JSONL logs written by AQILogger (one directory per run under
aqi_logs/) and produces the evaluation artifacts of the proposal:

    RQ1 operator variance:
        per-operator / per-mode success, force and workload tables;
        variance, CV and ICC(1,1) of run-level success across operators;
        robustness gain vs the baseline mode:
            RG_abs = Var_op(baseline) - Var_op(treatment)      (S6.6)
            RG_rel = RG_abs / Var_op(baseline)                 (primary endpoint)

    RQ2 what predicts a useful intervention:
        logistic fit of post-intervention success on the Option-A features
        (S5.5), AUROC, and the correlation of the counterfactual q_int with
        outcome -- the direct test that Option B beats the hand-weighted
        correlates.

    RQ3 trigger quality:
        AUROC and reliability (ECE) of P_fail and of the acquisition score
        A(s_t) against "failure/hard-violation within H steps" labels
        rebuilt from the step logs.

Outputs (to --out): summary.md, results.json, and figures as both .png and
.svg (learning curves, per-operator boxplots, q_int distributions,
calibration curves).

Only numpy is required; matplotlib is needed for the figures and is
imported lazily. No sklearn/scipy dependency: AUROC is the Mann-Whitney
rank statistic, the logistic fit is Newton's method on the standardized
features, ICC(1,1) is the one-way random-effects estimator.

Usage:
    python aqi_analysis.py --log_root aqi_logs --baseline_mode logged \
        --out aqi_analysis_out
    python aqi_analysis.py --run_dirs aqi_logs/runA aqi_logs/runB --out out
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

FAIL_HORIZON_DEFAULT = 20

# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
def _read_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def load_run(run_dir):
    meta_path = os.path.join(run_dir, "meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    return dict(
        dir=run_dir,
        name=os.path.basename(run_dir.rstrip("/")),
        # group by campaign ARM (e.g. 'quality-beh'), which is the mode for
        # plain arms and for runs logged before arms existed
        mode=((meta.get("config") or {}).get("arm")
              or meta.get("mode", "unknown")),
        base_mode=meta.get("mode", "unknown"),
        operator=meta.get("operator_id", "unknown"),
        config=meta.get("config", {}),
        episodes=_read_jsonl(os.path.join(run_dir, "episodes.jsonl")),
        segments=_read_jsonl(os.path.join(run_dir, "segments.jsonl")),
        steps_path=os.path.join(run_dir, "steps.jsonl"),
    )


def discover_runs(log_root=None, run_dirs=None):
    dirs = list(run_dirs or [])
    if log_root:
        dirs += [d for d in glob.glob(os.path.join(log_root, "*"))
                 if os.path.isdir(d)]
    runs = [load_run(d) for d in sorted(set(dirs))]
    return [r for r in runs if r["episodes"]]


# ---------------------------------------------------------------------------
# Statistics helpers (numpy only)
# ---------------------------------------------------------------------------
def auroc(scores, labels):
    """Mann-Whitney AUROC; nan if one class is empty."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64) > 0.5
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = scores.argsort(kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # midranks for ties
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = 0.5 * (i + 1 + j + 1)
        i = j + 1
    u = ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def ece(probs, labels, n_bins=10):
    """Expected calibration error with equal-width bins."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(probs)
    err = 0.0
    curve = []
    for b in range(n_bins):
        m = (probs >= edges[b]) & (probs < edges[b + 1] if b < n_bins - 1
                                   else probs <= edges[b + 1])
        if m.sum() == 0:
            continue
        conf, acc = probs[m].mean(), labels[m].mean()
        err += (m.sum() / total) * abs(conf - acc)
        curve.append((float(conf), float(acc), int(m.sum())))
    return float(err), curve


def logistic_fit(X, y, l2=1e-3, iters=100):
    """Newton's method logistic regression on standardized X.
    Returns dict with coef, intercept, standard errors, feature means/stds."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Z = (X - mu) / sd
    Z = np.concatenate([np.ones((len(Z), 1)), Z], axis=1)
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Z @ w, -30, 30)))
        g = Z.T @ (p - y) + l2 * w
        W = p * (1 - p)
        H = (Z * W[:, None]).T @ Z + l2 * np.eye(Z.shape[1])
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    p = 1.0 / (1.0 + np.exp(-np.clip(Z @ w, -30, 30)))
    W = p * (1 - p)
    H = (Z * W[:, None]).T @ Z + l2 * np.eye(Z.shape[1])
    try:
        se = np.sqrt(np.diag(np.linalg.inv(H)))
    except np.linalg.LinAlgError:
        se = np.full_like(w, np.nan)
    return dict(intercept=float(w[0]), coef=w[1:].tolist(),
                se=se[1:].tolist(), auroc=auroc(Z @ w, y),
                feature_mean=mu.tolist(), feature_std=sd.tolist())


def icc_oneway(groups):
    """ICC(1,1): one-way random effects over operator groups of run-level
    scores. groups: dict operator -> list of run scores. Uses the mean
    group size for unbalanced designs (Searle approximation). Returns
    (icc, var_between, var_within, k0)."""
    groups = {g: np.asarray(v, dtype=np.float64)
              for g, v in groups.items() if len(v) > 0}
    n_groups = len(groups)
    sizes = np.array([len(v) for v in groups.values()], dtype=np.float64)
    n_total = sizes.sum()
    if n_groups < 2 or n_total <= n_groups:
        return float("nan"), float("nan"), float("nan"), float("nan")
    grand = np.concatenate(list(groups.values())).mean()
    ss_between = sum(len(v) * (v.mean() - grand) ** 2 for v in groups.values())
    ss_within = sum(((v - v.mean()) ** 2).sum() for v in groups.values())
    ms_between = ss_between / (n_groups - 1)
    ms_within = ss_within / (n_total - n_groups)
    k0 = (n_total - (sizes ** 2).sum() / n_total) / (n_groups - 1)
    icc = (ms_between - ms_within) / (ms_between + (k0 - 1) * ms_within + 1e-12)
    return float(icc), float(ms_between), float(ms_within), float(k0)


# ---------------------------------------------------------------------------
# RQ1: per-run summaries and operator variance
# ---------------------------------------------------------------------------
def run_summary(run):
    eps = run["episodes"]
    succ = np.array([float(e.get("success", 0)) for e in eps])
    return dict(
        name=run["name"], mode=run["mode"], operator=run["operator"],
        n_episodes=len(eps),
        success_rate=float(succ.mean()),
        final_success_rate=float(succ[-20:].mean()) if len(succ) >= 5 else float(succ.mean()),
        mean_f_peak=float(np.mean([e.get("f_peak", 0.0) for e in eps])),
        max_f_peak=float(np.max([e.get("f_peak", 0.0) for e in eps])),
        hard_violation_rate=float(np.mean(
            [e.get("n_hard_violations", 0) > 0 for e in eps])),
        interventions_per_ep=float(np.mean(
            [e.get("n_segments", 0) for e in eps])),
        intervention_step_frac=float(
            sum(e.get("n_intervention_steps", 0) for e in eps)
            / max(sum(e.get("length", 1) for e in eps), 1)),
        requests_per_ep=float(np.mean([e.get("n_requests", 0) for e in eps])),
        excluded_segments=int(sum(e.get("n_excluded_segments", 0) for e in eps)),
        episodes_to_first_70=_episodes_to_rate(succ, 0.7),
    )


def _episodes_to_rate(succ, target, window=20):
    if len(succ) < window:
        return None
    roll = np.convolve(succ, np.ones(window) / window, mode="valid")
    hits = np.where(roll >= target)[0]
    return int(hits[0] + window) if len(hits) else None


def rq1_operator_variance(summaries, baseline_mode, metric="final_success_rate"):
    by_mode = defaultdict(lambda: defaultdict(list))
    for s in summaries:
        by_mode[s["mode"]][s["operator"]].append(s[metric])
    out = {}
    for mode, ops in by_mode.items():
        op_means = {op: float(np.mean(v)) for op, v in ops.items()}
        vals = np.array(list(op_means.values()))
        icc, msb, msw, k0 = icc_oneway(ops)
        out[mode] = dict(
            metric=metric,
            operator_means=op_means,
            n_operators=len(vals),
            var_across_operators=float(vals.var(ddof=1)) if len(vals) > 1 else float("nan"),
            cv_across_operators=float(vals.std(ddof=1) / (vals.mean() + 1e-12)) if len(vals) > 1 else float("nan"),
            icc1=icc, ms_between=msb, ms_within=msw, k0=k0,
        )
    base = out.get(baseline_mode)
    for mode, d in out.items():
        if base and mode != baseline_mode and np.isfinite(
                base["var_across_operators"]):
            rg_abs = base["var_across_operators"] - d["var_across_operators"]
            d["robustness_gain_abs"] = float(rg_abs)
            d["robustness_gain_rel"] = float(
                rg_abs / (base["var_across_operators"] + 1e-12))
    return out


# ---------------------------------------------------------------------------
# RQ2: intervention-quality features vs outcome
# ---------------------------------------------------------------------------
RQ2_FEATURES = ["d_progress", "d_value", "f_peak", "jerk_mean", "t_int"]


def rq2_feature_analysis(runs):
    rows, labels, q_means, a_means = [], [], [], []
    per_operator_q = defaultdict(list)
    for run in runs:
        for seg in run["segments"]:
            f = seg.get("features", {})
            if not f:
                continue
            rows.append([float(f.get(k, 0.0)) for k in RQ2_FEATURES])
            labels.append(float(bool(f.get("success", False))))
            q_means.append(float(seg.get("q_int_mean", 0.0)))
            a_means.append(float(seg.get("a_hat_mean", 0.0)))
            per_operator_q[run["operator"]].append(
                float(seg.get("q_int_mean", 0.0)))
    if len(rows) < 10 or len(set(labels)) < 2:
        return dict(note="not enough labeled segments for RQ2 fit",
                    n_segments=len(rows)), per_operator_q
    X, y = np.asarray(rows), np.asarray(labels)
    fit = logistic_fit(X, y)
    res = dict(
        n_segments=len(rows),
        base_rate=float(y.mean()),
        features=RQ2_FEATURES,
        logistic=fit,
        auroc_features=fit["auroc"],
        auroc_q_int=auroc(q_means, y),
        auroc_a_hat=auroc(a_means, y),
        corr_q_int_success=float(np.corrcoef(q_means, y)[0, 1]),
    )
    return res, per_operator_q


# ---------------------------------------------------------------------------
# RQ3: trigger calibration from the step logs
# ---------------------------------------------------------------------------
def rq3_trigger_calibration(runs, horizon=FAIL_HORIZON_DEFAULT,
                            max_steps_per_run=200000):
    a_scores, p_fails, sigmas, labels = [], [], [], []
    for run in runs:
        ep_success = {e["episode"]: bool(e.get("success", False))
                      for e in run["episodes"]}
        by_ep = defaultdict(list)
        n = 0
        if not os.path.exists(run["steps_path"]):
            continue
        with open(run["steps_path"]) as f:
            for line in f:
                if n >= max_steps_per_run:
                    break
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                by_ep[r.get("episode", 0)].append(r)
                n += 1
        for ep, recs in by_ep.items():
            recs.sort(key=lambda r: r.get("step", 0))
            T = len(recs)
            lab = np.zeros(T)
            events = [i for i, r in enumerate(recs)
                      if r.get("hard_violation", False)]
            if ep in ep_success and not ep_success[ep]:
                events.append(T - 1)
            for e in events:
                lab[max(0, e - horizon):e + 1] = 1.0
            def _f(rec, key, default):
                v = rec.get(key)
                return float(v) if v is not None else default
            for i, r in enumerate(recs):
                # composable runs omit inactive sections' keys (e.g. no
                # p_fail under trigger_terms=uncertainty) -> use neutral
                # defaults so cross-config analyses still run.
                a_scores.append(_f(r, "a_score", 0.0))
                p_fails.append(_f(r, "p_fail", 0.5))
                sigmas.append(_f(r, "sigma_q", 0.0))
                labels.append(lab[i])
    if not labels:
        return dict(note="no step logs found")
    labels = np.asarray(labels)
    e, curve = ece(np.asarray(p_fails), labels)
    return dict(
        n_steps=len(labels),
        label_rate=float(labels.mean()),
        auroc_a_score=auroc(a_scores, labels),
        auroc_p_fail=auroc(p_fails, labels),
        auroc_sigma_q=auroc(sigmas, labels),
        ece_p_fail=e,
        reliability_curve=curve,
    )


# ---------------------------------------------------------------------------
# Figures (saved as both .png and .svg)
# ---------------------------------------------------------------------------
def _savefig(fig, out_dir, name):
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"),
                    bbox_inches="tight", dpi=150)


def make_figures(runs, summaries, rq1, rq2_qdist, rq3, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping figures")
        return

    # learning curves (rolling success by episode)
    fig, ax = plt.subplots(figsize=(7, 4))
    for run in runs:
        succ = np.array([float(e.get("success", 0)) for e in run["episodes"]])
        if len(succ) < 3:
            continue
        w = min(20, max(3, len(succ) // 5))
        roll = np.convolve(succ, np.ones(w) / w, mode="valid")
        ax.plot(np.arange(len(roll)) + w, roll,
                label=f"{run['mode']}/{run['operator']}")
    ax.set_xlabel("episode")
    ax.set_ylabel(f"success rate (rolling)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=7)
    ax.set_title("Learning curves per run")
    _savefig(fig, out_dir, "learning_curves")
    plt.close(fig)

    # per-operator final success by mode
    modes = sorted({s["mode"] for s in summaries})
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, mode in enumerate(modes):
        vals = [s["final_success_rate"] for s in summaries
                if s["mode"] == mode]
        ax.scatter([i] * len(vals), vals, alpha=0.8)
        if len(vals) > 1:
            ax.boxplot([vals], positions=[i], widths=0.5, showfliers=False)
    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels(modes)
    ax.set_ylabel("final success rate (per run)")
    ax.set_title("Operator/run spread per mode (RQ1)")
    _savefig(fig, out_dir, "operator_variance")
    plt.close(fig)

    # q_int distributions per operator
    if rq2_qdist:
        fig, ax = plt.subplots(figsize=(6, 4))
        for op, q in sorted(rq2_qdist.items()):
            if q:
                ax.hist(q, bins=30, alpha=0.5, label=op, density=True)
        ax.set_xlabel("segment mean q_int")
        ax.set_ylabel("density")
        ax.set_title("Counterfactual intervention quality by operator (S5.5)")
        ax.legend(fontsize=8)
        _savefig(fig, out_dir, "q_int_by_operator")
        plt.close(fig)

    # reliability curve for P_fail
    if isinstance(rq3, dict) and rq3.get("reliability_curve"):
        conf = [c for c, _, _ in rq3["reliability_curve"]]
        acc = [a for _, a, _ in rq3["reliability_curve"]]
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.plot(conf, acc, "o-")
        ax.set_xlabel("predicted P_fail")
        ax.set_ylabel("empirical failure rate")
        ax.set_title(f"P_fail calibration (ECE={rq3.get('ece_p_fail', 0):.3f})")
        _savefig(fig, out_dir, "p_fail_calibration")
        plt.close(fig)


# ---------------------------------------------------------------------------
def _md_table(rows, cols):
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(
            f"{r.get(c):.3f}" if isinstance(r.get(c), float)
            else str(r.get(c)) for c in cols) + " |")
    return "\n".join(lines)


def write_summary_md(out_dir, summaries, rq1, rq2, rq3, baseline_mode):
    cols = ["name", "mode", "operator", "n_episodes", "success_rate",
            "final_success_rate", "mean_f_peak", "hard_violation_rate",
            "interventions_per_ep", "intervention_step_frac",
            "requests_per_ep"]
    lines = ["# AQI-HIL-SERL analysis", "",
             "## Per-run summary", "", _md_table(summaries, cols), "",
             f"## RQ1 - operator variance (baseline mode: {baseline_mode})",
             ""]
    for mode, d in rq1.items():
        lines.append(f"### mode = {mode}")
        lines.append(f"- operator means: {d['operator_means']}")
        lines.append(f"- var across operators: {d['var_across_operators']:.5f}"
                     f", CV: {d['cv_across_operators']:.3f}"
                     f", ICC(1,1): {d['icc1']:.3f}")
        if "robustness_gain_rel" in d:
            lines.append(f"- **robustness gain vs {baseline_mode}:** "
                         f"RG_abs={d['robustness_gain_abs']:.5f}, "
                         f"RG_rel={d['robustness_gain_rel']:.3f}")
        lines.append("")
    lines += ["## RQ2 - intervention quality", "",
              "```json", json.dumps(rq2, indent=2), "```", "",
              "## RQ3 - trigger calibration", "",
              "```json",
              json.dumps({k: v for k, v in rq3.items()
                          if k != "reliability_curve"}, indent=2),
              "```", ""]
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log_root", default=None,
                    help="Directory containing one subdirectory per run "
                         "(the AQIConfig.log_dir, default 'aqi_logs').")
    ap.add_argument("--run_dirs", nargs="*", default=None,
                    help="Explicit run directories (alternative to --log_root).")
    ap.add_argument("--out", default="aqi_analysis_out")
    ap.add_argument("--baseline_mode", default="logged",
                    help="Mode used as Var_op baseline for the robustness "
                         "gain (proposal S6.6). 'logged' is the "
                         "same-architecture diagnostic control; use "
                         "'hilserl' for the plain baseline.")
    ap.add_argument("--metric", default="final_success_rate",
                    choices=["final_success_rate", "success_rate",
                             "mean_f_peak", "intervention_step_frac"])
    ap.add_argument("--fail_horizon", type=int, default=FAIL_HORIZON_DEFAULT)
    args = ap.parse_args()

    runs = discover_runs(args.log_root, args.run_dirs)
    if not runs:
        raise SystemExit("No runs with episodes.jsonl found. Point "
                         "--log_root at the AQIConfig.log_dir.")
    os.makedirs(args.out, exist_ok=True)

    summaries = [run_summary(r) for r in runs]
    rq1 = rq1_operator_variance(summaries, args.baseline_mode, args.metric)
    rq2, qdist = rq2_feature_analysis(runs)
    rq3 = rq3_trigger_calibration(runs, horizon=args.fail_horizon)

    results = dict(runs=summaries, rq1=rq1, rq2=rq2, rq3=rq3,
                   baseline_mode=args.baseline_mode, metric=args.metric)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    write_summary_md(args.out, summaries, rq1, rq2, rq3, args.baseline_mode)
    make_figures(runs, summaries, rq1, qdist, rq3, args.out)

    print(f"Analyzed {len(runs)} runs -> {args.out}/summary.md, "
          f"results.json, figures (*.png / *.svg)")


if __name__ == "__main__":
    main()
