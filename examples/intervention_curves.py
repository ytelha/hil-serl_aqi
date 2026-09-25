#!/usr/bin/env python3
"""Operator-intervention frequency curves over training, per cell.

For a `hilserl` (or any) campaign the actor dumps its replay buffers every
`buffer_period` steps:
  * `checkpoints/<cell>/buffer/transitions_N.pkl`      = ALL steps in the window
                                                         ending at env-step N;
  * `checkpoints/<cell>/demo_buffer/transitions_N.pkl` = the INTERVENTION steps
                                                         (operator takeovers) in
                                                         that same window.
So the per-window takeover rate is simply
    rate(N) = len(demo_buffer_N) / len(buffer_N).
Early in training the policy is weak and the synthetic operator takes over often
(high rate); as the policy converges the rate decays. This script reconstructs
that curve for every cell and saves one figure per campaign (tier panels: each
seed thin + the tier mean thick, with the convergence step marked).

Usage:
    python intervention_curves.py --campaign runs/rq1_3seed_nopretrain
Writes <campaign>/report/plots/intervention_curves.png (+ intervention_rates.csv).
Safe to import: campaign_plots.make_plots() calls make_intervention_curves().
"""
import argparse
import glob
import json
import os
import pickle
import re

import numpy as np

_TIER_KEYS = (("exp", "expert"), ("int", "intermediate"), ("nov", "novice"))
_TIER_COLOR = {"expert": "tab:green", "intermediate": "tab:orange",
               "novice": "tab:red"}
# one colour per method so each subplot row is visually distinct
_METHOD_COLORS = ["tab:blue", "tab:red", "tab:green", "tab:purple",
                 "tab:orange", "tab:brown"]


def _method_of(cid):
    """Method token of a cell id '<method>_<operator>_s<seed>'."""
    return cid.split("_", 1)[0]


def _pkl_len(path):
    try:
        with open(path, "rb") as f:
            return len(pickle.load(f))
    except Exception:
        return None


def _step_of(path):
    m = re.findall(r"(\d+)", os.path.basename(path))
    return int(m[0]) if m else -1


def cell_curve(cell_dir):
    """(steps, rates) for one cell from its buffer / demo_buffer pkls."""
    bufs = sorted(glob.glob(os.path.join(cell_dir, "buffer",
                                         "transitions_*.pkl")), key=_step_of)
    steps, rates = [], []
    for b in bufs:
        n_all = _pkl_len(b)
        if not n_all:
            continue
        d = b.replace(os.sep + "buffer" + os.sep,
                      os.sep + "demo_buffer" + os.sep)
        n_int = _pkl_len(d) if os.path.exists(d) else 0
        steps.append(_step_of(b))
        rates.append((n_int or 0) / n_all)
    return np.asarray(steps, float), np.asarray(rates, float)


def _tier_of(cell_id):
    for k, t in _TIER_KEYS:
        if k in cell_id:
            return t
    return "other"


def _smooth(y, w=3):
    if len(y) < w or w < 2:
        return y
    k = np.ones(w) / w
    return np.convolve(y, k, mode="same")


def _converged_step(cell_dir):
    cj = os.path.join(cell_dir, "CONVERGED.json")
    if os.path.exists(cj):
        try:
            return json.load(open(cj)).get("step")
        except Exception:
            return None
    return None


def _sig(t):
    return (np.asarray(t["actions"], np.float32).tobytes()
            + repr(float(np.ravel(t["rewards"])[0])).encode())


def _ep_takeover_from_buffers(cell_dir):
    """Exact per-window EPISODE takeover rate from the buffers: demo_buffer is a
    temporally ordered subsequence of buffer, so a two-pointer match recovers
    the per-step takeover flag; an episode is a takeover if any of its steps was
    an intervention. Returns (steps, ep_rate)."""
    bufs = sorted(glob.glob(os.path.join(cell_dir, "buffer",
                                         "transitions_*.pkl")), key=_step_of)
    steps, rates = [], []
    for b in bufs:
        try:
            tr = pickle.load(open(b, "rb"))
        except Exception:
            continue
        d = b.replace(os.sep + "buffer" + os.sep,
                      os.sep + "demo_buffer" + os.sep)
        dm = pickle.load(open(d, "rb")) if os.path.exists(d) else []
        dsig = [_sig(x) for x in dm]
        j, had, eps = 0, False, []
        for t in tr:
            if j < len(dsig) and _sig(t) == dsig[j]:
                had = True
                j += 1
            done = (bool(np.ravel(t.get("dones", 0))[0])
                    or bool(np.ravel(t.get("masks", 1))[0] == 0))
            if done:
                eps.append(had)
                had = False
        if had:
            eps.append(had)
        if eps:
            steps.append(_step_of(b))
            rates.append(float(np.mean(eps)))
    return np.asarray(steps, float), np.asarray(rates, float)


def _ep_takeover_from_jsonl(cell_dir, window=1000):
    """Exact per-window EPISODE takeover rate from the actor's episodes.jsonl
    (written by newer runs); None if absent/empty."""
    path = os.path.join(cell_dir, "episodes.jsonl")
    if not os.path.exists(path):
        return None
    bins = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        b = (int(r.get("step", 0)) // window) * window + window
        bins.setdefault(b, []).append(
            1.0 if int(r.get("intervention_count", 0)) > 0 else 0.0)
    if not bins:
        return None
    steps = sorted(bins)
    return (np.asarray(steps, float),
            np.asarray([float(np.mean(bins[s])) for s in steps], float))


def cell_episode_curve(cell_dir):
    """Episode-level takeover rate: prefer episodes.jsonl (exact), else
    reconstruct from the saved buffers."""
    j = _ep_takeover_from_jsonl(cell_dir)
    if j is not None and len(j[0]):
        return j
    return _ep_takeover_from_buffers(cell_dir)


def make_intervention_curves(campaign, out=None):
    """Reconstruct + plot per-cell takeover-rate curves. Returns the PNG path
    (or None if there are no buffers / matplotlib is missing)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[intervention_curves] matplotlib missing; skipping")
        return None

    cells = sorted(d for d in glob.glob(os.path.join(campaign, "checkpoints",
                                                     "*")) if os.path.isdir(d))
    # group by (method, tier): each METHOD gets its own row of subplots and the
    # two arms (hilserl vs quality) are never conflated in one panel.
    groups = {}
    csv_rows = [("cell", "method", "tier", "step", "episode_takeover_rate",
                 "step_rate")]
    for c in cells:
        cid = os.path.basename(c)
        method = _method_of(cid)
        tier = _tier_of(cid)
        if tier not in ("expert", "intermediate", "novice"):
            continue
        es, er = cell_episode_curve(c)   # per-EPISODE takeover rate (primary)
        ss, sr = cell_curve(c)           # per-STEP rate (kept in the csv)
        if len(es) == 0:
            continue
        groups.setdefault((method, tier), []).append(
            (cid, es, er, _converged_step(c)))
        srmap = dict(zip(ss.astype(int), sr))
        for st, rt in zip(es, er):
            csv_rows.append((cid, method, tier, int(st), round(float(rt), 4),
                             round(float(srmap.get(int(st), float("nan"))), 4)))

    if not groups:
        print(f"[intervention_curves] no cell buffers under {campaign}")
        return None

    methods = sorted({m for (m, _) in groups})
    tiers = [t for t in ("expert", "intermediate", "novice")
             if any((m, t) in groups for m in methods)]
    method_color = {m: _METHOD_COLORS[i % len(_METHOD_COLORS)]
                    for i, m in enumerate(methods)}

    out = out or os.path.join(campaign, "report", "plots",
                              "intervention_curves.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    nrows, ncols = len(methods), len(tiers)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows),
                             sharey=True, squeeze=False)
    for ri, m in enumerate(methods):
        col = method_color[m]
        for ci, t in enumerate(tiers):
            ax = axes[ri][ci]
            g = groups.get((m, t), [])
            if g:
                # common grid for the (method, tier) mean (union of seed steps)
                allsteps = np.unique(np.concatenate([s for _, s, _, _ in g]))
                stacked = []
                for cid, s, r, conv in g:
                    ax.plot(s, _smooth(r), color=col, alpha=0.35, lw=1.2)
                    if conv:
                        ax.axvline(conv, color=col, ls=":", alpha=0.35, lw=1)
                    stacked.append(np.interp(allsteps, s, r,
                                             left=np.nan, right=np.nan))
                mean = np.nanmean(np.vstack(stacked), axis=0)
                ax.plot(allsteps, _smooth(mean), color=col, lw=2.6,
                        label=f"{m} mean")
                ax.set_title(f"{m} · {t}  (n={len(g)} seeds)", fontsize=9)
            else:
                ax.set_title(f"{m} · {t}  (n=0)", fontsize=9)
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, color="0.6", fontsize=9)
            ax.axhline(0.1, color="gray", ls="--", lw=1, alpha=0.8)
            ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.3)
            if ri == nrows - 1:
                ax.set_xlabel("training env-step")
        axes[ri][0].set_ylabel(f"{m}\nper-episode takeover rate")
    fig.suptitle("Operator intervention frequency over training — "
                 f"{os.path.basename(campaign.rstrip('/'))}\n"
                 "rows = method, columns = tier  (thin = seed, thick = mean; "
                 "dashed = 10% target; dotted vline = convergence step)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=115)
    plt.close(fig)

    csv_path = os.path.join(os.path.dirname(out), "intervention_rates.csv")
    with open(csv_path, "w") as f:
        for row in csv_rows:
            f.write(",".join(str(x) for x in row) + "\n")
    print(f"[intervention_curves] wrote {out} and {csv_path}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True, help="campaign dir")
    ap.add_argument("--out", default=None, help="output PNG path")
    args = ap.parse_args()
    make_intervention_curves(args.campaign, args.out)


if __name__ == "__main__":
    main()
