#!/usr/bin/env python3
"""Active-gate signal audit (RQ3): does the ensemble-disagreement trigger fire
when the operator is GENUINELY in trouble?

Reads the per-step trigger timeline written by `active`/`aqi` runs with
`quality_cf_diag: true` (steps.jsonl: sigma_q, a_score, request, lcb_q, and the
simulator ground truth gt_lat_err_mm / gt_height_mm / gt_jammed + op_z_task).

For each score (sigma_Q, the acquisition score A(s_t), -LCB) it reports
AUROC / correlation against TIMING-FREE trouble labels -- unlike the quality
audit's episode-success label, these are physical (jam / misalignment) or the
operator's own trouble estimate op_z_task -- pooled, per tier and
time-stratified, plus the request precision / recall and a gate-DIRECTION check
(AUROC < 0.5 => the gate would be inverted).

Usage:
  ./.venv/bin/python examples/active_signal_audit.py --campaign runs/active_validity
"""
import argparse
import collections
import glob
import json
import math
import os

import numpy as np


def _auroc(score, label):
    """P(score of a TROUBLE step > score of a NO-trouble step). 0.5 = useless,
    <0.5 = inverted. Rank-based (handles ties, class imbalance)."""
    score = np.asarray(score, float)
    label = np.asarray(label, int)
    m = np.isfinite(score)
    score, label = score[m], label[m]
    pos, neg = score[label == 1], score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), len(pos), len(neg)
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty_like(order, float)
    ranks[order] = np.arange(1, len(allv) + 1)
    a = (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return float(a), len(pos), len(neg)


def _spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = math.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


def _tier(run_dir):
    # aqi_logs/<arm>_<op>_s<seed>_actor  ->  op like exp00/int01/nov02
    base = os.path.basename(run_dir.rstrip("/"))
    for t in ("exp", "int", "nov"):
        if f"_{t}" in base:
            return {"exp": "expert", "int": "intermediate", "nov": "novice"}[t]
    return "?"


def load(campaign):
    rows = []
    files = sorted(glob.glob(os.path.join(campaign, "aqi_logs", "*", "steps.jsonl")))
    for f in files:
        tier = _tier(os.path.dirname(f))
        for line in open(f):
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append((tier, d))
    return rows, files


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--campaign", required=True, help="campaign dir (reads <dir>/aqi_logs/*/steps.jsonl)")
    ap.add_argument("--clearance_mm", type=float, default=5.0,
                    help="misalignment threshold: gt_lat_err_mm above this = trouble (cuboid 5mm)")
    ap.add_argument("--near_mm", type=float, default=15.0,
                    help="only count misalignment as trouble when within this height of the hole")
    ap.add_argument("--contact_mm", type=float, default=8.0,
                    help="contact zone: |gt_height_mm| below this. The headline AUROC is also "
                         "reported restricted to this zone so a score cannot 'predict' trouble "
                         "trivially through approach height (jam only happens in contact).")
    ap.add_argument("--z_task_thresh", type=float, default=0.5,
                    help="op_z_task above this = the operator would have worried")
    ap.add_argument("--n_time_bins", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows, files = load(args.campaign)
    if not rows:
        print(f"no steps.jsonl under {args.campaign}/aqi_logs/ (needs an active/aqi run "
              f"with log_every_step + quality_cf_diag). Files found: {len(files)}")
        return 2

    tiers = [r[0] for r in rows]
    recs = [r[1] for r in rows]

    def col(k):
        return np.array([r.get(k, np.nan) if r.get(k) is not None else np.nan for r in recs], float)

    sigma = col("sigma_q")
    a_score = col("a_score")
    lcb = col("lcb_q")
    request = np.array([1 if r.get("request") else 0 for r in recs], int)
    intv = np.array([1 if r.get("intervening") else 0 for r in recs], int)
    step = col("step")
    jam = np.array([1 if r.get("gt_jammed") else 0 for r in recs], int)
    lat = col("gt_lat_err_mm")
    height = col("gt_height_mm")
    op_z = col("op_z_task")

    has_gt = np.isfinite(lat).any() or jam.any() or np.isfinite(op_z).any()

    # Contact zone: where insertion trouble is physically possible (jam only
    # occurs here). Scores correlated with approach height can 'predict' jam
    # trivially out of zone, so the headline AUROC is also reported restricted
    # to this mask to remove that confound.
    contact = np.isfinite(height) & (np.abs(height) < args.contact_mm)

    # ---- trouble labels (timing-free) --------------------------------------
    # jam (the simulator's own stuck flag) is the ONE clean physical label; the
    # operator's actual takeover (intervening) is the clean BEHAVIOURAL target
    # -- what a real operator chose to help with, which is what the gate should
    # anticipate. The loose misalignment and op_z_task labels are retained only
    # as flagged references: gt_lat_err_mm is the full XY distance (median tens
    # of mm, so lat>clearance is ~always true) and op_z_task saturates near 1.
    misalign = (np.isfinite(lat) & (lat > args.clearance_mm)
                & np.isfinite(height) & (np.abs(height) < args.near_mm)).astype(int)
    trouble_phys = ((jam == 1) | (misalign == 1)).astype(int)
    trouble_op = (np.isfinite(op_z) & (op_z > args.z_task_thresh)).astype(int)

    scores = {"sigma_Q": sigma, "A(s_t)": a_score, "-LCB_Q": -lcb}
    # clean labels first (jam, operator-intervened), then the flagged loose ones
    labels = {
        "jam (physical, clean)": jam,
        "operator-intervened (behavioural)": intv,
        "phys (jam|misalign, LOOSE)": trouble_phys,
        "op_z_task>thr (SATURATED)": trouble_op,
    }

    lines = []
    w = lines.append
    w("# Active-gate signal audit (RQ3)\n")
    w(f"- campaign: `{args.campaign}`  |  runs: {len(files)}  |  steps: {len(recs)}")
    w(f"- clean labels: **jam** (simulator stuck flag) and **operator-intervened** "
      f"(the operator's actual takeover). Loose/flagged: phys = jam OR "
      f"(lat>{args.clearance_mm}mm within {args.near_mm}mm of hole); op_z_task>{args.z_task_thresh}.")
    w(f"- ground-truth fields present: {has_gt}  |  request base rate: {request.mean():.3f}  |  "
      f"contact zone |h|<{args.contact_mm}mm: {contact.mean():.3f} of steps\n")
    w("> **Label caveats.** `gt_lat_err_mm` is the full XY distance (median tens of mm), so a "
      "`lat>clearance` label is ~always true and the LOOSE phys label is near-degenerate. "
      "`op_z_task` saturates near 1 for most steps, so AUROC against it is not discriminative. "
      "`jam` only occurs in the contact zone, so any height-correlated score (e.g. -LCB) can "
      "'predict' it trivially out of zone — read the **contact-zone** column, not the pooled one.\n")
    if not has_gt:
        w("> **No ground-truth fields** — the run had `quality_cf_diag` off or the operator "
          "did not emit ground truth. Only request stats below are meaningful.\n")

    # ---- AUROC table -------------------------------------------------------
    w("## AUROC(score, trouble)  [0.5 = useless; <0.5 = gate inverted]\n")
    w("Contact-zone column = same AUROC restricted to |h|<"
      f"{args.contact_mm}mm (removes the approach-height confound).\n")
    w("| score | label | AUROC pooled | AUROC contact-zone | n_trouble | n_ok |")
    w("|---|---|---|---|---|---|")
    for lname, lab in labels.items():
        for sname, sc in scores.items():
            a, npos, nneg = _auroc(sc, lab)
            ac, _, _ = _auroc(sc[contact], lab[contact])
            w(f"| {sname} | {lname} | {a:.3f} | {ac:.3f} | {npos} | {nneg} |")
    w("")

    # ---- per-tier + time-stratified for the headline (clean labels) --------
    for lab_name, lab in (("jam", jam), ("operator-intervened", intv)):
        for sname, sc in (("sigma_Q", sigma), ("-LCB_Q", -lcb)):
            w(f"## {sname} vs {lab_name} (contact zone) — per tier & time-stratified\n")
            w("| slice | AUROC | n_trouble | n_ok |")
            w("|---|---|---|---|")
            for t in ("expert", "intermediate", "novice"):
                m = contact & np.array([x == t for x in tiers])
                a, p, n = _auroc(sc[m], lab[m])
                w(f"| tier={t} | {a:.3f} | {p} | {n} |")
            fin = step[np.isfinite(step)]
            if fin.size:
                edges = np.quantile(fin, np.linspace(0, 1, args.n_time_bins + 1))
                for i in range(args.n_time_bins):
                    m = (contact & np.isfinite(step)
                         & (step >= edges[i]) & (step <= edges[i + 1]))
                    a, p, n = _auroc(sc[m], lab[m])
                    w(f"| step∈[{edges[i]:.0f},{edges[i+1]:.0f}] | {a:.3f} | {p} | {n} |")
            w("")

    # ---- correlation with the operator's own trouble estimate --------------
    w("## Agreement with op_z_task (does the ensemble know trouble when the operator does?)\n")
    w("| score | Spearman r with op_z_task |")
    w("|---|---|")
    for sname, sc in scores.items():
        w(f"| {sname} | {_spearman(sc, op_z):.3f} |")
    w("")

    # ---- request precision / recall vs trouble -----------------------------
    w("## Request precision / recall (did the gate ask when in trouble?)\n")
    w("| label | base rate | request rate | precision P(trouble\\|req) | recall P(req\\|trouble) | F1 |")
    w("|---|---|---|---|---|---|")
    for lname, lab in labels.items():
        req = request.astype(bool)
        tl = lab.astype(bool)
        tp = int((req & tl).sum()); fp = int((req & ~tl).sum()); fn = int((~req & tl).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        w(f"| {lname} | {lab.mean():.3f} | {request.mean():.3f} | {prec:.3f} | {rec:.3f} | {f1:.3f} |")
    w("")

    # ---- per-tier means (context) -----------------------------------------
    w("## Per-tier means\n")
    w("| tier | n_steps | mean sigma_Q | mean A(s_t) | jam rate | intervened rate | request rate |")
    w("|---|---|---|---|---|---|---|")
    for t in ("expert", "intermediate", "novice"):
        m = np.array([x == t for x in tiers])
        if not m.any():
            continue
        w(f"| {t} | {int(m.sum())} | {np.nanmean(sigma[m]):.4f} | {np.nanmean(a_score[m]):.4f} | "
          f"{jam[m].mean():.3f} | {intv[m].mean():.3f} | {request[m].mean():.3f} |")
    w("")

    out = args.out or os.path.join(args.campaign, "report", "active_audit")
    os.makedirs(out, exist_ok=True)
    txt = "\n".join(lines)
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write(txt + "\n")
    print(txt)
    print(f"\n-> {out}/summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
