#!/usr/bin/env python3
"""Intervention-quality signal audit: which per-takeover signals predict the
episode outcome, and is that prediction genuine quality or LEAKAGE from
takeover timing?

Why this script exists
----------------------
RQ4 (docs/thesis/rq4_ensemble_convergence_and_qa_log.md sections 11-12)
scored every intervention segment offline by AUROC(score, episode success)
and concluded that the counterfactual Q-gap is anti-predictive (0.175) while
the critic-free behavioural composite B = z(d_value) - z(t_int) - z(f_peak)
predicts success (0.72 offline, 0.78 live). That analysis was done ad hoc;
this script makes it reproducible (PART A) and adds the checks needed before
trusting it (PART B), because the synthetic operator can make takeover length
t_int track the episode outcome mechanically:

  * every operator has a maximum takeover time d_max (expert ~3.8 s,
    intermediate ~5.5 s, novice ~8 s, jittered per operator; 10 Hz control),
    after which it hands control back (SyntheticInterventionGate,
    "handback_dmax");
  * its trouble signal z_t contains a time-pressure term 0.5 * t / budget
    (operator_gate.MonitoringSignal), and it only hands back once z_t stays
    below delta_out (0.2-0.35). Late in an episode z_t cannot fall that low,
    so late takeovers run to the d_max cap or to the episode end;
  * late episodes are mostly failing episodes (timeout = failure).

So "short takeover => success" can partly mean "early takeover => success"
and "t_int ~ d_max of the operator's tier". PART B separates the two.

What it computes (per arm; runs grouped by meta.json config.arm, else mode)
------------------------------------------------------------------------
PART A -- signal AUROC vs episode success (reproduces RQ4 11.1 / 12.1):
  q_int (as logged online; counterfactual OR behavioural, per the run),
  a_hat_mean, a_hat_at_trigger, d_value, d_progress, -t_int, -f_peak,
  -jerk_mean, sigma_mean, B = z(d_value)-z(t_int)-z(f_peak),
  B without t_int, plus TIME-ONLY yardsticks (-start_step, -end_step) that
  know nothing but WHEN the takeover happened. Overall, per tier, and with a
  95% bootstrap CI that resamples whole episodes (segments of one episode
  share the label). Per-tier means (e.g. the t_int skill gradient).
PART B -- leakage diagnostics:
  B1 how each takeover ended: "cap" (length within --cap_tolerance of the
     operator's d_max cap), "handback" (released before the cap), or
     "episode_end" (still in control when the episode ended); share per tier
     and by outcome. Needs d_max from meta.json (synthetic operators).
  B2 t_int vs the cap: per tier mean t_int, mean cap, ratio, share at cap;
     per-operator correlation of mean t_int with cap.
  B3 the same AUROCs on the CLEAN subset (handback only: neither capped nor
     cut by the episode end), TIME-STRATIFIED (pairs compared only within
     quantile bins of start_step / max episode length), and against a
     post-release label (episode succeeded within --horizon steps after the
     operator released control; released segments only).
  B4 logistic fits success ~ -t_int, ~ time, ~ -t_int + time: does -t_int
     add anything once time-in-episode is known?
The summary ends with rule-based observations: numbers first, cautious
wording; they flag what to look at, they do not decide the thesis claim.

Inputs / outputs
----------------
  python quality_signal_audit.py --campaign runs/quality_behavioral_diag
      -> <campaign>/report/quality_audit/{summary.md,audit.json,segments.csv,
         *.png}
  python quality_signal_audit.py --log_root <dir with run folders> --out <dir>
Reads only aqi_logs/<run>/{meta.json,segments.jsonl,episodes.jsonl} (JSONL
written by serl_launcher/aqi/logger.py). numpy only; matplotlib optional.
Offline B uses whole-run z-scores; the online score used running z-scores,
so the logged q_int of a behavioural run is the exact online value and B
here is its offline twin.
"""

import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np

_TIER_BY_PREFIX = {"exp": "expert", "int": "intermediate", "nov": "novice"}
TIERS = ("expert", "intermediate", "novice")

# (name, getter(row) -> float or None, note). Higher score = predicted success.
SCORES = [
    ("q_int (logged)", lambda r: r["q_int_mean"],
     "the online score the run used (counterfactual or behavioral)"),
    ("a_hat_mean", lambda r: r["a_hat_mean"], "counterfactual Q-gap, mean"),
    ("a_hat_at_trigger", lambda r: r["a_hat_at_trigger"],
     "counterfactual Q-gap at takeover start"),
    ("d_value", lambda r: r["d_value"], "mu_Q(end) - mu_Q(start)"),
    ("d_progress", lambda r: r["d_progress"], "insertion depth gained"),
    ("-t_int", lambda r: _neg(r["t_int"]), "shorter takeover"),
    ("-f_peak", lambda r: _neg(r["f_peak"]), "lower peak force"),
    ("-jerk_mean", lambda r: _neg(r["jerk_mean"]), "smoother operator"),
    ("sigma_mean", lambda r: r["sigma_mean"],
     "critic disagreement (RQ4: temporal confound)"),
    ("B = z(dV)-z(t)-z(F)", lambda r: r["B"], "behavioral composite"),
    ("B without t_int", lambda r: r["B_no_t"], "z(dV) - z(F)"),
    ("-start_step (time only)", lambda r: _neg(r["start_step"]),
     "TIME-ONLY yardstick: earlier takeover"),
    ("-end_step (time only)", lambda r: _neg(r["end_step"]),
     "TIME-ONLY yardstick: earlier release"),
]
KEY_SCORES = ("q_int (logged)", "-t_int", "B = z(dV)-z(t)-z(F)",
              "B without t_int", "d_value", "-start_step (time only)")


def _neg(x):
    return None if x is None else -x


# ===========================================================================
# statistics (numpy only)
# ===========================================================================
def _ranks(x):
    """Average ranks (1-based), ties share their mean rank."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    xs = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def _mann_whitney(scores, labels):
    """(U, n_pos * n_neg): U counts positive > negative pairs (ties 1/2)."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.0, 0
    r = _ranks(s)
    return float(r[y].sum() - n_pos * (n_pos + 1) / 2.0), n_pos * n_neg


def auroc(scores, labels):
    """P(score of a random success > score of a random failure); ties 1/2.
    NaN if either class is empty. 0.5 = no information, < 0.5 = inverted."""
    u, pairs = _mann_whitney(scores, labels)
    return u / pairs if pairs else float("nan")


def stratified_auroc(scores, labels, strata):
    """AUROC counting only pairs INSIDE the same stratum (e.g. same
    time-in-episode bin), so a score cannot win by merely encoding the
    stratum. Equals auroc() when there is a single stratum."""
    scores, labels, strata = map(np.asarray, (scores, labels, strata))
    u_tot, pairs_tot = 0.0, 0
    for s in np.unique(strata):
        m = strata == s
        u, pairs = _mann_whitney(scores[m], labels[m])
        u_tot, pairs_tot = u_tot + u, pairs_tot + pairs
    return u_tot / pairs_tot if pairs_tot else float("nan")


def cluster_bootstrap_auroc(scores, labels, clusters, n_boot=500, seed=0):
    """95% CI of the AUROC resampling whole clusters (episodes) with
    replacement; segments of one episode share its label, so they are not
    independent observations."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    groups = defaultdict(list)
    for i, c in enumerate(clusters):
        groups[c].append(i)
    idx = [np.asarray(v) for v in groups.values()]
    if len(idx) < 5:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = np.concatenate([idx[k] for k in
                               rng.integers(0, len(idx), len(idx))])
        a = auroc(scores[pick], labels[pick])
        if a == a:
            vals.append(a)
    if len(vals) < n_boot // 2:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def logistic_fit(X, y, l2=1e-3, iters=500, lr=0.5):
    """Standardised-feature logistic regression by gradient descent. Returns
    (coefficients incl. intercept first, in-sample AUROC)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Z = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Z @ w, -30, 30)))
        grad = Z.T @ (p - y) / len(y) + l2 * np.r_[0.0, w[1:]]
        w -= lr * grad
    return w, auroc(Z @ w, y.astype(bool))


def _pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


# ===========================================================================
# loading
# ===========================================================================
def _read_jsonl(path):
    out = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        continue
    return out


def _f(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def load_segments(log_root, control_hz=10.0, cap_tolerance=2,
                  max_episode_steps=300, horizon=50):
    """One row per intervention segment with raw features + derived timing /
    termination fields. Runs without segments are skipped."""
    rows = []
    for meta_path in sorted(glob.glob(os.path.join(log_root, "*",
                                                   "meta.json"))):
        run_dir = os.path.dirname(meta_path)
        try:
            meta = json.load(open(meta_path))
        except (ValueError, OSError):
            continue
        segs = _read_jsonl(os.path.join(run_dir, "segments.jsonl"))
        if not segs:
            continue
        eps = _read_jsonl(os.path.join(run_dir, "episodes.jsonl"))
        ep_len = {int(e["episode"]): int(e.get("length", 0)) for e in eps
                  if "episode" in e}
        cfg = meta.get("config") or {}
        op = cfg.get("synthetic_operator") or {}
        operator = meta.get("operator_id") or op.get("operator_id") or "?"
        tier = op.get("tier") or _TIER_BY_PREFIX.get(str(operator)[:3],
                                                     "unknown")
        d_max = _f(op.get("d_max"))
        cap = (int(math.ceil(d_max * control_hz - 1e-9))
               if d_max is not None else None)
        t_max = max([max_episode_steps] + list(ep_len.values()))
        run = os.path.basename(run_dir.rstrip("/\\"))
        arm = cfg.get("arm") or meta.get("mode", "unknown")
        for s in segs:
            feats = s.get("features") or {}
            ep = int(s.get("episode", -1))
            start = int(s.get("start_step", 0))
            length = int(s.get("length", feats.get("t_int", 0)) or 0)
            end = start + length
            L = ep_len.get(ep)
            success = bool(s.get("episode_success", False))
            if L is not None and end >= L:
                ended = "episode_end"
            elif cap is None:
                ended = "released"          # human / unknown cap
            elif length >= cap - cap_tolerance:
                ended = "cap"
            else:
                ended = "handback"
            post = None
            if ended != "episode_end" and L is not None:
                post = bool(success and 0 < L - end <= horizon)
            rows.append(dict(
                run=run, arm=arm, mode=meta.get("mode", "unknown"),
                score_mode=cfg.get("quality_score_mode"),
                operator=operator, tier=tier, episode=ep,
                cluster=f"{run}#{ep}", start_step=start, end_step=end,
                t_int=float(length), episode_length=L, d_max=d_max,
                cap_steps=cap, ended=ended, success=success,
                post_release_success=post,
                start_frac=start / float(t_max),
                requested=bool(s.get("requested", False)),
                excluded=bool(s.get("excluded", False)),
                q_int_mean=_f(s.get("q_int_mean")),
                a_hat_mean=_f(s.get("a_hat_mean")),
                a_hat_at_trigger=_f(s.get("a_hat_at_trigger")),
                sigma_mean=_f(s.get("sigma_mean")),
                d_value=_f(feats.get("d_value")),
                d_progress=_f(feats.get("d_progress")),
                f_peak=_f(feats.get("f_peak")),
                jerk_mean=_f(feats.get("jerk_mean"))))
    _add_composites(rows)
    return rows


def _add_composites(rows):
    """Offline behavioral composite, z-scored within each run (the online
    score is standardised per run too)."""
    by_run = defaultdict(list)
    for r in rows:
        by_run[r["run"]].append(r)

    def z(vals):
        a = np.asarray([np.nan if v is None else v for v in vals], float)
        if np.isnan(a).all():           # feature never logged in this run
            return a
        sd = np.nanstd(a)
        return (a - np.nanmean(a)) / sd if sd > 0 else np.zeros_like(a)

    for rs in by_run.values():
        zd = z([r["d_value"] for r in rs])
        zt = z([r["t_int"] for r in rs])
        zf = z([r["f_peak"] for r in rs])
        for i, r in enumerate(rs):
            b = zd[i] - zt[i] - zf[i]
            b0 = zd[i] - zf[i]
            r["B"] = float(b) if np.isfinite(b) else None
            r["B_no_t"] = float(b0) if np.isfinite(b0) else None


# ===========================================================================
# analysis
# ===========================================================================
def _score_vectors(rows, getter, label_key="success"):
    """(scores, labels, idx) over rows where both score and label exist."""
    s, y, idx = [], [], []
    for i, r in enumerate(rows):
        v, lab = getter(r), r[label_key]
        if v is None or lab is None:
            continue
        s.append(v)
        y.append(bool(lab))
        idx.append(i)
    return np.asarray(s, float), np.asarray(y, bool), idx


def _time_strata(rows, n_bins):
    """Quantile bins of start_frac (time into the episode)."""
    t = np.asarray([r["start_frac"] for r in rows], float)
    if len(t) == 0:
        return np.zeros(0, int)
    edges = np.unique(np.quantile(t, np.linspace(0, 1, n_bins + 1)))
    return np.clip(np.searchsorted(edges, t, side="right") - 1, 0,
                   max(len(edges) - 2, 0))


def analyse_group(rows, n_bins=5, n_boot=500):
    """PART A + B for one arm's segments."""
    out = {"n_segments": len(rows),
           "n_success": int(sum(r["success"] for r in rows)),
           "n_runs": len({r["run"] for r in rows}),
           "score_modes": sorted({str(r["score_mode"]) for r in rows})}
    strata_all = _time_strata(rows, n_bins)
    clean = [r for r in rows if r["ended"] == "handback"]
    released = [r for r in rows if r["post_release_success"] is not None]

    # ---- A: AUROC table + B3 variants --------------------------------------
    table = {}
    for name, get, note in SCORES:
        s, y, idx = _score_vectors(rows, get)
        entry = {"note": note, "n": int(len(s)), "auroc": auroc(s, y)}
        entry["ci95"] = (cluster_bootstrap_auroc(
            s, y, [rows[i]["cluster"] for i in idx], n_boot=n_boot)
            if name in KEY_SCORES and len(s) else (float("nan"),) * 2)
        entry["auroc_stratified"] = (stratified_auroc(
            s, y, strata_all[idx]) if len(s) else float("nan"))
        cs, cy, _ = _score_vectors(clean, get)
        entry["auroc_clean"], entry["n_clean"] = auroc(cs, cy), int(len(cs))
        ps, py, _ = _score_vectors(released, get, "post_release_success")
        entry["auroc_post_release"] = auroc(ps, py)
        entry["by_tier"] = {}
        for t in TIERS:
            ts, ty, _ = _score_vectors([r for r in rows if r["tier"] == t],
                                       get)
            entry["by_tier"][t] = auroc(ts, ty)
        table[name] = entry
    out["auroc"] = table

    # ---- per-tier means (skill gradient of each raw signal) ----------------
    means = {}
    for key in ("t_int", "q_int_mean", "d_value", "f_peak", "sigma_mean",
                "start_frac"):
        means[key] = {}
        for t in TIERS:
            v = [r[key] for r in rows if r["tier"] == t and r[key] is not None]
            means[key][t] = float(np.mean(v)) if v else float("nan")
    out["tier_means"] = means

    # ---- B1/B2: how takeovers end, t_int vs cap -------------------------------
    term = {}
    for t in TIERS + ("all",):
        rs = [r for r in rows if t == "all" or r["tier"] == t]
        if not rs:
            continue
        caps = [r["cap_steps"] for r in rs if r["cap_steps"] is not None]
        row = {"n": len(rs),
               "mean_t_int": float(np.mean([r["t_int"] for r in rs])),
               "mean_cap": float(np.mean(caps)) if caps else float("nan")}
        row["t_int_over_cap"] = (row["mean_t_int"] / row["mean_cap"]
                                 if caps else float("nan"))
        for kind in ("cap", "handback", "episode_end", "released"):
            k = [r for r in rs if r["ended"] == kind]
            row[f"frac_{kind}"] = len(k) / len(rs)
            row[f"success_rate_{kind}"] = (
                float(np.mean([r["success"] for r in k])) if k
                else float("nan"))
        term[t] = row
    out["termination"] = term
    per_op = defaultdict(list)
    for r in rows:
        if r["cap_steps"] is not None:
            per_op[r["operator"]].append(r)
    ops = sorted(per_op)
    out["per_operator"] = {
        op: {"tier": per_op[op][0]["tier"],
             "cap_steps": per_op[op][0]["cap_steps"],
             "mean_t_int": float(np.mean([r["t_int"] for r in per_op[op]])),
             "n": len(per_op[op])} for op in ops}
    out["corr_t_int_vs_cap_across_operators"] = _pearson(
        [out["per_operator"][o]["cap_steps"] for o in ops],
        [out["per_operator"][o]["mean_t_int"] for o in ops])

    # ---- B4: does -t_int add anything once time is known? --------------------
    # time = start of the takeover only: end_step = start_step + t_int would
    # make the three features collinear and the t_int coefficient meaningless
    y = np.asarray([r["success"] for r in rows], float)
    fits = {}
    if 0 < y.sum() < len(y):
        t_int = np.asarray([[-r["t_int"]] for r in rows])
        time = np.asarray([[-r["start_frac"]] for r in rows])
        for name, X in (("-t_int", t_int), ("time", time),
                        ("-t_int + time", np.hstack([t_int, time]))):
            w, a = logistic_fit(X, y)
            fits[name] = {"coef": [float(c) for c in w[1:]], "auroc": a}
    out["logistic"] = fits
    out["observations"] = _observations(out)
    return out


def _observations(g):
    """Rule-based, cautious reading of the numbers (flags, not verdicts)."""
    obs = []
    a = g["auroc"]

    def get(name, key="auroc"):
        return a.get(name, {}).get(key, float("nan"))

    t_all, t_clean, t_str = get("-t_int"), get("-t_int", "auroc_clean"), \
        get("-t_int", "auroc_stratified")
    time_only = max(get("-start_step (time only)"),
                    get("-end_step (time only)"))
    if t_all == t_all and time_only == time_only:
        obs.append(
            f"-t_int reaches AUROC {t_all:.3f} on all segments; a score that "
            f"knows ONLY when the takeover happened reaches {time_only:.3f}."
            + (" Timing alone explains most of -t_int's predictive power."
               if time_only >= t_all - 0.03 else ""))
    if t_all == t_all and t_clean == t_clean:
        drop = t_all - t_clean
        obs.append(
            f"-t_int on the clean subset (released by handback, "
            f"n={a['-t_int']['n_clean']}): {t_clean:.3f} "
            f"({'-' if drop >= 0 else '+'}{abs(drop):.3f} vs all)."
            + (" The signal largely disappears when capped and "
               "episode-end-truncated takeovers are removed."
               if t_clean < 0.6 <= t_all else ""))
    if t_str == t_str:
        obs.append(f"-t_int time-stratified AUROC (pairs compared within "
                   f"the same time-in-episode bin): {t_str:.3f}.")
    for name in ("B = z(dV)-z(t)-z(F)", "B without t_int", "d_value"):
        if get(name) == get(name):
            obs.append(f"{name}: all {get(name):.3f}, clean "
                       f"{get(name, 'auroc_clean'):.3f}, stratified "
                       f"{get(name, 'auroc_stratified'):.3f}, post-release "
                       f"{get(name, 'auroc_post_release'):.3f}.")
    term = g["termination"].get("all", {})
    if term:
        obs.append(
            f"How takeovers ended: {term['frac_cap']:.0%} at the d_max cap, "
            f"{term['frac_episode_end']:.0%} cut by the episode end, "
            f"{term['frac_handback']:.0%} released by handback"
            + (f" ({term['frac_released']:.0%} released, cap unknown)"
               if term["frac_released"] else "") + ".")
    tiers = [t for t in TIERS if t in g["termination"]]
    if tiers:
        obs.append("Mean t_int / d_max cap per tier: " + ", ".join(
            f"{t} {g['termination'][t]['mean_t_int']:.1f}/"
            f"{g['termination'][t]['mean_cap']:.0f}"
            f" ({g['termination'][t]['t_int_over_cap']:.0%})"
            for t in tiers) + ".")
    r = g.get("corr_t_int_vs_cap_across_operators", float("nan"))
    if r == r:
        obs.append(f"Across operators, corr(mean t_int, d_max cap) = {r:.2f}"
                   + (" -- takeover length largely tracks the operator's "
                      "cap parameter." if r >= 0.8 else "."))
    fits = g.get("logistic", {})
    if "-t_int" in fits and "-t_int + time" in fits:
        obs.append(
            f"Logistic AUROC: -t_int alone {fits['-t_int']['auroc']:.3f}, "
            f"time alone {fits['time']['auroc']:.3f}, both "
            f"{fits['-t_int + time']['auroc']:.3f}; standardised -t_int "
            f"coefficient {fits['-t_int']['coef'][0]:+.2f} alone vs "
            f"{fits['-t_int + time']['coef'][0]:+.2f} with time.")
    return obs


# ===========================================================================
# report
# ===========================================================================
def _fmt(x, nd=3):
    return "n/a" if x is None or x != x else f"{x:.{nd}f}"


def render_md(result, args_ns):
    L = ["# Intervention-quality signal audit\n",
         f"- log root: `{result['log_root']}`",
         f"- segments: {result['n_segments']} in {result['n_runs']} runs; "
         f"cap tolerance {args_ns['cap_tolerance']} steps, "
         f"{args_ns['control_hz']:g} Hz, post-release horizon "
         f"{args_ns['horizon']} steps, {args_ns['n_bins']} time bins",
         "- AUROC = P(score of a segment from a SUCCESSFUL episode > score "
         "of one from a FAILED episode); 0.5 = no information, < 0.5 = "
         "inverted. Method: docs/quality_signal_audit.md.\n"]
    for arm, g in result["arms"].items():
        L.append(f"## Arm `{arm}` ({g['n_segments']} segments, "
                 f"{g['n_success']} from successful episodes, "
                 f"{g['n_runs']} runs; score mode "
                 f"{', '.join(g['score_modes'])})\n")
        L.append("### Observations\n")
        L += [f"- {o}" for o in g["observations"]]
        L.append("\n### A. AUROC vs episode success (+ B3 leakage variants)\n")
        L.append("| score | n | all | 95% CI (episode bootstrap) | clean "
                 "(handback only) | time-stratified | post-release label | "
                 + " | ".join(TIERS) + " |")
        L.append("|---|---|---|---|---|---|---|" + "---|" * len(TIERS))
        for name, e in g["auroc"].items():
            ci = e["ci95"]
            ci_s = (f"[{_fmt(ci[0])}, {_fmt(ci[1])}]"
                    if ci[0] == ci[0] else "")
            L.append(
                f"| {name} | {e['n']} | {_fmt(e['auroc'])} | {ci_s} | "
                f"{_fmt(e['auroc_clean'])} (n={e['n_clean']}) | "
                f"{_fmt(e['auroc_stratified'])} | "
                f"{_fmt(e['auroc_post_release'])} | "
                + " | ".join(_fmt(e["by_tier"][t]) for t in TIERS) + " |")
        L.append("\n### Per-tier means\n")
        L.append("| signal | " + " | ".join(TIERS) + " |")
        L.append("|---|" + "---|" * len(TIERS))
        for key, m in g["tier_means"].items():
            L.append(f"| {key} | " + " | ".join(_fmt(m[t]) for t in TIERS)
                     + " |")
        L.append("\n### B1/B2. How takeovers ended\n")
        L.append("| tier | n | mean t_int | mean cap | t_int/cap | at cap | "
                 "cut by episode end | handback | SR at cap | SR episode-end "
                 "| SR handback |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for t, r in g["termination"].items():
            L.append(
                f"| {t} | {r['n']} | {_fmt(r['mean_t_int'], 1)} | "
                f"{_fmt(r['mean_cap'], 0)} | {_fmt(r['t_int_over_cap'], 2)} | "
                f"{r['frac_cap']:.0%} | {r['frac_episode_end']:.0%} | "
                f"{r['frac_handback']:.0%} | "
                f"{_fmt(r['success_rate_cap'], 2)} | "
                f"{_fmt(r['success_rate_episode_end'], 2)} | "
                f"{_fmt(r['success_rate_handback'], 2)} |")
        if g["per_operator"]:
            L.append("\n| operator | tier | d_max cap (steps) | mean t_int | "
                     "segments |")
            L.append("|---|---|---|---|---|")
            for op, r in g["per_operator"].items():
                L.append(f"| `{op}` | {r['tier']} | {r['cap_steps']} | "
                         f"{r['mean_t_int']:.1f} | {r['n']} |")
        if g["logistic"]:
            L.append("\n### B4. Logistic fits (standardised features)\n")
            L.append("| model | coefficients | in-sample AUROC |")
            L.append("|---|---|---|")
            for name, f in g["logistic"].items():
                L.append(f"| success ~ {name} | "
                         + ", ".join(f"{c:+.2f}" for c in f["coef"])
                         + f" | {_fmt(f['auroc'])} |")
        L.append("")
    return "\n".join(L)


def _plots(result, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    written = []
    for arm, g in result["arms"].items():
        names = [n for n in g["auroc"]]
        cols = [("auroc", "all"), ("auroc_clean", "clean (handback)"),
                ("auroc_stratified", "time-stratified"),
                ("auroc_post_release", "post-release label")]
        x = np.arange(len(names))
        fig, ax = plt.subplots(figsize=(12, 4.5))
        for k, (key, lab) in enumerate(cols):
            vals = [g["auroc"][n][key] for n in names]
            ax.bar(x + (k - 1.5) * 0.2, [v if v == v else 0 for v in vals],
                   0.2, label=lab)
        ax.axhline(0.5, color="k", lw=0.8, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel("AUROC vs episode success")
        ax.set_title(f"{arm}: signal AUROC, all vs leakage-controlled")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = os.path.join(out_dir, f"{arm}_auroc.png")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written.append(path)
    return written


def make_audit(campaign=None, log_root=None, out=None, control_hz=10.0,
               cap_tolerance=2, max_episode_steps=300, horizon=50, n_bins=5,
               n_boot=500, plots=True):
    """Run the audit; returns the output directory or None if no segments.
    Importable (run_campaign.py analyze calls it) and CLI-callable."""
    if log_root is None:
        log_root = os.path.join(campaign, "aqi_logs")
    if out is None:
        out = (os.path.join(campaign, "report", "quality_audit") if campaign
               else os.path.join(log_root, "quality_audit"))
    rows = load_segments(log_root, control_hz, cap_tolerance,
                         max_episode_steps, horizon)
    if not rows:
        return None
    os.makedirs(out, exist_ok=True)
    by_arm = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)
    result = {"log_root": log_root, "n_segments": len(rows),
              "n_runs": len({r["run"] for r in rows}),
              "arms": {arm: analyse_group(rs, n_bins, n_boot)
                       for arm, rs in sorted(by_arm.items())}}
    params = dict(control_hz=control_hz, cap_tolerance=cap_tolerance,
                  horizon=horizon, n_bins=n_bins)
    with open(os.path.join(out, "audit.json"), "w") as f:
        json.dump({**result, "params": params}, f, indent=2)
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write(render_md(result, params))
    keys = [k for k in rows[0] if k != "cluster"]
    with open(os.path.join(out, "segments.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    if plots:
        _plots(result, out)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--campaign", help="campaign dir (reads <dir>/aqi_logs)")
    src.add_argument("--log_root", help="dir holding one folder per run")
    ap.add_argument("--out", default=None, help="output dir (default: "
                    "<campaign>/report/quality_audit)")
    ap.add_argument("--control_hz", type=float, default=10.0)
    ap.add_argument("--cap_tolerance", type=int, default=2,
                    help="a takeover within this many steps of the "
                         "operator's d_max cap counts as capped")
    ap.add_argument("--max_episode_steps", type=int, default=300,
                    help="episode budget used to normalise start time")
    ap.add_argument("--horizon", type=int, default=50,
                    help="post-release label: success within this many "
                         "steps after the operator released control")
    ap.add_argument("--n_bins", type=int, default=5,
                    help="time-in-episode quantile bins for the stratified "
                         "AUROC")
    ap.add_argument("--n_boot", type=int, default=500)
    ap.add_argument("--no_plots", action="store_true")
    a = ap.parse_args()
    out = make_audit(a.campaign, a.log_root, a.out, a.control_hz,
                     a.cap_tolerance, a.max_episode_steps, a.horizon,
                     a.n_bins, a.n_boot, plots=not a.no_plots)
    if out is None:
        print("no intervention segments found (segments.jsonl) -- nothing "
              "to audit.")
        return 1
    print(open(os.path.join(out, "summary.md")).read())
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
