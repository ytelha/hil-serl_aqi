#!/usr/bin/env python3
"""Counterfactual-score study: why A_hat = mu_Q(s,a_H) - mu_Q(s,a_robot)
fails, and whether a better-built counterfactual score works.

Needs a run with quality_cf_diag on (examples/campaigns/cf_score_study.yaml):
segments.jsonl carries every critic's Q at a_H / a_robot / the policy mode /
8 policy samples per takeover step, and steps.jsonl the simulator ground
truth (true lateral error, height, jam) after every step.

    python examples/cf_score_analysis.py --campaign runs/cf-study [--json out.json]

Read-only. numpy only. See docs/counterfactual_score_study.md.

Scores compared (per takeover step, then averaged per segment):
    A_cf    mean_k Q_k(a_H) - mean_k Q_k(a_robot)      the current score
    A_mode  mean_k Q_k(a_H) - mean_k Q_k(mu_pi)        vs the policy mode
    A_V     mean_k Q_k(a_H) - mean_k V_k               vs E_pi Q (8 samples)
    pct     share of policy samples with Q below Q(a_H) (scale-free rank)
    t_k     mean_k(Q_k(a_H)-V_k) / std_k(...)           across-critic agreement
            on the DIFFERENCE, not on Q
Ground-truth labels (simulator, timing-free):
    per step    d_align = lateral error before - after the step (mm)
    per segment G_align = lateral error at onset - at handback (mm)
                aligned_end = lateral error at handback < clearance
                jam_end     = jammed at handback
plus the classic labels: episode success (all and time-stratified).
"""

import argparse
import glob
import json
import os

import numpy as np


def _rank(x):
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    # average ranks over ties
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 5 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(_rank(x), _rank(y))[0, 1])


def auroc(score, label):
    s, l = np.asarray(score, float), np.asarray(label, bool)
    m = np.isfinite(s)
    s, l = s[m], l[m]
    n1, n0 = int(l.sum()), int((~l).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = _rank(s) + 1
    return float((r[l].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auroc_stratified(score, label, strata, n_bins=3):
    """Mean within-bin AUROC (pairs compared only inside a time bin)."""
    s, l, t = (np.asarray(v, float) for v in (score, label, strata))
    edges = np.quantile(t, np.linspace(0, 1, n_bins + 1))
    vals, w = [], []
    for b in range(n_bins):
        m = (t >= edges[b]) & (t <= edges[b + 1] if b == n_bins - 1
                               else t < edges[b + 1])
        a = auroc(s[m], l[m].astype(bool))
        if np.isfinite(a):
            n1 = l[m].sum()
            vals.append(a)
            w.append(n1 * (m.sum() - n1))
    return float(np.average(vals, weights=w)) if vals else float("nan")


def jl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def load(camp):
    runs = []
    for d in sorted(glob.glob(os.path.join(camp, "aqi_logs", "*_actor"))):
        segs = [s for s in jl(os.path.join(d, "segments.jsonl"))
                if isinstance(s.get("cf_diag"), dict) and "qh" in s["cf_diag"]]
        if not segs:
            continue
        steps = {(r["episode"], r["step"]): r
                 for r in jl(os.path.join(d, "steps.jsonl"))}
        eps = {e["episode"]: e for e in jl(os.path.join(d, "episodes.jsonl"))}
        runs.append((os.path.basename(d), segs, steps, eps))
    return runs


def analyse(camp, clearance=5.0, horizon=50):
    runs = load(camp)
    if not runs:
        raise SystemExit(f"no segments with cf_diag under {camp}/aqi_logs")
    P = {k: [] for k in ("A_cf", "A_mode", "A_V", "pct", "t_k", "sig_q",
                         "sig_diff", "base_err", "q_spread", "dist_mode",
                         "d_align", "d_height", "lat_pre", "logged",
                         "seg_id")}
    S = {k: [] for k in ("A_cf", "A_mode", "A_V", "pct", "t_k", "G_align",
                         "aligned_end", "jam_end", "ep_success", "post_succ",
                         "episode", "length", "lat_onset")}
    n_err = 0
    seg_id = 0
    for _name, segs, steps, eps in runs:
        for seg in segs:
            d = seg["cf_diag"]
            qh, qr, qmu, vk = (np.asarray(d[k], float) for k in
                               ("qh", "qr", "qmu", "vk"))
            qs, a_mu = np.asarray(d["qs"], float), np.asarray(d["a_mu"], float)
            e, t0, L = seg["episode"], seg["start_step"], seg["length"]
            if qh.shape[1] != L:
                n_err += 1
                continue
            a_cf = qh.mean(0) - qr.mean(0)
            a_mode = qh.mean(0) - qmu.mean(0)
            a_v = qh.mean(0) - vk.mean(0)
            pct = (qs < qh.mean(0)[None]).mean(0)
            diff = qh - vk
            t_k = diff.mean(0) / (diff.std(0) + 1e-9)
            for i in range(L):
                r = steps.get((e, t0 + i), {})
                prev = steps.get((e, t0 + i - 1), {})
                a_h = r.get("a_h")
                lat, lat_p = r.get("gt_lat_err_mm"), prev.get("gt_lat_err_mm")
                h, h_p = r.get("gt_height_mm"), prev.get("gt_height_mm")
                P["A_cf"].append(a_cf[i]); P["A_mode"].append(a_mode[i])
                P["A_V"].append(a_v[i]); P["pct"].append(pct[i])
                P["t_k"].append(t_k[i])
                P["sig_q"].append(qh[:, i].std())
                P["sig_diff"].append(diff[:, i].std())
                P["base_err"].append(qr[:, i].mean() - vk[:, i].mean())
                P["q_spread"].append(qs[:, i].std())
                P["dist_mode"].append(
                    float(np.linalg.norm(np.asarray(a_h) - a_mu[i]))
                    if a_h is not None else np.nan)
                P["d_align"].append(lat_p - lat if None not in (lat, lat_p)
                                    else np.nan)
                P["d_height"].append(h_p - h if None not in (h, h_p)
                                     else np.nan)
                P["lat_pre"].append(lat_p if lat_p is not None else np.nan)
                P["logged"].append(seg["a_hat"][i])
                P["seg_id"].append(seg_id)
            onset = steps.get((e, t0 - 1), {}).get("gt_lat_err_mm")
            last = steps.get((e, t0 + L - 1), {})
            ep = eps.get(e, {})
            ep_len = ep.get("length")
            post = (bool(seg["episode_success"]) and ep_len is not None
                    and ep_len - (t0 + L) <= horizon)
            S["A_cf"].append(a_cf.mean()); S["A_mode"].append(a_mode.mean())
            S["A_V"].append(a_v.mean()); S["pct"].append(pct.mean())
            S["t_k"].append(t_k.mean())
            S["G_align"].append(onset - last["gt_lat_err_mm"]
                                if onset is not None
                                and last.get("gt_lat_err_mm") is not None
                                else np.nan)
            S["aligned_end"].append(last.get("gt_lat_err_mm", np.inf)
                                    < clearance)
            S["jam_end"].append(bool(last.get("gt_jammed", False)))
            S["ep_success"].append(bool(seg["episode_success"]))
            S["post_succ"].append(post)
            S["episode"].append(e)
            S["length"].append(L)
            S["lat_onset"].append(onset if onset is not None else np.nan)
            seg_id += 1
    P = {k: np.asarray(v, float) for k, v in P.items()}
    S = {k: np.asarray(v, float) for k, v in S.items()}
    scores = ("A_cf", "A_mode", "A_V", "pct", "t_k")
    out = {"n_segments": int(len(S["A_cf"])), "n_steps": int(len(P["A_cf"])),
           "n_bad_segments": n_err,
           "logged_vs_recomputed_max_abs": float(np.nanmax(np.abs(
               P["A_cf"] - P["logged"]))) if len(P["A_cf"]) else None,
           "episodes": int(len(np.unique(S["episode"]))),
           "segments_in_successful_episodes": int(S["ep_success"].sum())}
    med = lambda x: float(np.nanmedian(np.abs(x)))
    out["magnitude"] = {
        "median|A_cf|": med(P["A_cf"]), "median|A_V|": med(P["A_V"]),
        "median sigma_Q(a_H) (the gate's uncertainty)": med(P["sig_q"]),
        "median across-critic std of Q(a_H)-V (the difference's)":
            med(P["sig_diff"]),
        "median |Q(a_robot)-E_pi Q| (one-sample baseline error)":
            med(P["base_err"]),
        "median spread of Q over policy samples": med(P["q_spread"]),
    }
    out["sign"] = {k: float(np.nanmean(P[k] < 0))
                   for k in ("A_cf", "A_mode", "A_V")}
    out["sign"]["pct<0.5"] = float(np.nanmean(P["pct"] < 0.5))
    out["argmax_bias"] = {
        "spearman(A_mode, -||a_H - mu_pi||)": spearman(P["A_mode"],
                                                       -P["dist_mode"]),
        "spearman(A_V, -||a_H - mu_pi||)": spearman(P["A_V"],
                                                    -P["dist_mode"])}
    mis = P["lat_pre"] > clearance
    out["per_step_vs_ground_truth"] = {
        k: {"spearman(score, d_align) all": spearman(P[k], P["d_align"]),
            "... misaligned steps only": spearman(P[k][mis], P["d_align"][mis]),
            "spearman(score, d_height)": spearman(P[k], P["d_height"]),
            "within-segment mean spearman(score, d_align)": float(np.nanmean(
                [spearman(P[k][P["seg_id"] == s], P["d_align"][P["seg_id"] == s])
                 for s in np.unique(P["seg_id"])
                 if (P["seg_id"] == s).sum() >= 8]))}
        for k in scores}
    out["per_segment_vs_labels"] = {
        k: {"spearman(score, G_align)": spearman(S[k], S["G_align"]),
            "AUROC(score, aligned_end)": auroc(S[k], S["aligned_end"]),
            "AUROC(score, not jam_end)": auroc(S[k], ~S["jam_end"].astype(bool)),
            "AUROC(score, episode success)": auroc(S[k], S["ep_success"]),
            "... time-stratified": auroc_stratified(S[k], S["ep_success"],
                                                     S["episode"]),
            "AUROC(score, post-release success)": auroc(S[k], S["post_succ"]),
            "spearman(score, takeover length)": spearman(S[k], S["length"]),
            "spearman(score, episode index)": spearman(S[k], S["episode"])}
        for k in scores}
    out["label_sanity"] = {
        "AUROC(G_align, episode success)": auroc(S["G_align"], S["ep_success"]),
        "AUROC(aligned_end, episode success)": auroc(S["aligned_end"],
                                                     S["ep_success"]),
        "share aligned_end": float(np.mean(S["aligned_end"])),
        "share jam_end": float(np.mean(S["jam_end"])),
        "median G_align (mm)": float(np.nanmedian(S["G_align"]))}
    thirds = np.array_split(np.argsort(S["episode"], kind="mergesort"), 3)
    out["drift"] = {f"third {i + 1}": {
        "mean A_cf": float(np.mean(S["A_cf"][ix])),
        "share A_cf<0": float(np.mean(S["A_cf"][ix] < 0)),
        "success share": float(np.mean(S["ep_success"][ix]))}
        for i, ix in enumerate(thirds)}
    return out


def _print(out):
    def fmt(v):
        return f"{v:+.3f}" if isinstance(v, float) and abs(v) < 100 else str(v)
    for k, v in out.items():
        if isinstance(v, dict):
            print(f"\n## {k}")
            for kk, vv in v.items():
                if isinstance(vv, dict):
                    print(f"  {kk}: " + "  ".join(f"{a}={fmt(b)}"
                                                  for a, b in vv.items()))
                else:
                    print(f"  {kk}: {vv:.3g}" if isinstance(vv, float)
                          else f"  {kk}: {vv}")
        else:
            print(f"{k}: {v}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--clearance_mm", type=float, default=5.0,
                    help="cuboid 5, cylinder 8, hex 3")
    ap.add_argument("--horizon", type=int, default=50,
                    help="post-release success: success within this many steps")
    ap.add_argument("--json", default=None, help="also write the numbers here")
    a = ap.parse_args()
    out = analyse(a.campaign, a.clearance_mm, a.horizon)
    _print(out)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
