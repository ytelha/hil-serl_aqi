#!/usr/bin/env python3
"""Figures for an operator-dependency campaign.

Reads ``<campaign>/report/dependency.json`` (written by run_campaign's analyze
stage) and renders PNGs into ``<campaign>/report/plots/``:

  * icc_by_method.png          the headline dependency number per method
  * sr_by_operator.png         per-operator mean SR within each method
                               (between-operator spread = the dependency; the
                               error bar on each point = within-operator spread)
  * sr_by_tier.png             mean SR per skill tier per method (skill gradient)
  * variance_decomposition.png sigma2_between vs sigma2_within, stacked
  * operator_spread.png        CV per method + the "operator matters" p-value
  * steps_to_convergence.png   training effort per method (+ converged fraction)

Import and call ``make_plots(campaign_dir)``, or run standalone:

    python campaign_plots.py --campaign runs/study1

matplotlib is imported lazily so a missing install degrades to "no plots"
without breaking the numeric report.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

TIER_COLOR = {"expert": "#2ca02c", "intermediate": "#ff7f0e",
              "novice": "#d62728"}
_METHOD_COLOR = "#4c72b0"
_BASE_COLOR = "#999999"


def _num(x):
    """Return a finite float or None."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _methods_ordered(report):
    ms = list(report.get("methods", {}).keys())
    base = report.get("baseline")
    if base in ms:
        ms = [base] + [m for m in ms if m != base]
    return ms


def _tiers(report):
    # infer the tier order from the campaign meta, else a sensible default
    t = (report.get("campaign", {}) or {}).get("tiers")
    return t or ["expert", "intermediate", "novice"]


# ===========================================================================
# individual figures
# ===========================================================================
def _fig_icc_by_method(plt, report, out):
    ms = _methods_ordered(report)
    conv = report.get("convergence", {})
    sr_icc = [_num(report["methods"][m].get("icc")) for m in ms]
    st_icc = [_num(conv.get(m, {}).get("icc")) for m in ms]
    x = range(len(ms))
    w = 0.38
    fig, ax = plt.subplots(figsize=(1.6 * len(ms) + 2, 4))
    ax.bar([i - w / 2 for i in x], [v or 0 for v in sr_icc], w,
           label="final SR", color=_METHOD_COLOR)
    ax.bar([i + w / 2 for i in x], [v or 0 for v in st_icc], w,
           label="steps-to-converge", color="#dd8452")
    ax.set_xticks(list(x))
    ax.set_xticklabels(ms)
    ax.set_ylabel("ICC (operator dependency)")
    ax.set_ylim(0, 1)
    ax.set_title("Operator dependency per method  (lower = better)")
    ax.legend()
    for i, v in enumerate(sr_icc):
        if v is not None:
            ax.text(i - w / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _fig_sr_by_operator(plt, report, out):
    import numpy as np
    ms = _methods_ordered(report)
    srbm = report.get("sr_by_method", {})
    fig, ax = plt.subplots(figsize=(1.7 * len(ms) + 2, 4.5))
    rng = np.random.default_rng(0)
    for i, m in enumerate(ms):
        ops = srbm.get(m, {})
        op_means = []
        for op, d in ops.items():
            vals = [_num(v) for v in d.get("sr", {}).values()]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            mean = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            op_means.append(mean)
            jx = i + rng.uniform(-0.16, 0.16)
            color = TIER_COLOR.get(d.get("tier"), "#333333")
            ax.errorbar(jx, mean, yerr=sd, fmt="o", ms=5, color=color,
                        ecolor=color, elinewidth=1, capsize=2, alpha=0.85)
        if op_means:
            ax.hlines(np.mean(op_means), i - 0.28, i + 0.28,
                      color="black", lw=2, zorder=5)
    ax.set_xticks(range(len(ms)))
    ax.set_xticklabels(ms)
    ax.set_ylabel("success rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Per-operator SR within each method\n"
                 "(scatter = between-operator spread; bar = within-operator "
                 "spread; black line = method mean)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=t)
               for t, c in TIER_COLOR.items()]
    ax.legend(handles=handles, title="tier", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _fig_sr_by_tier(plt, report, out):
    ms = _methods_ordered(report)
    tiers = _tiers(report)
    x = range(len(tiers))
    n = len(ms)
    w = 0.8 / max(n, 1)
    fig, ax = plt.subplots(figsize=(1.6 * len(tiers) + 2, 4))
    for j, m in enumerate(ms):
        tm = report["methods"][m].get("tier_mean_sr", {})
        vals = [_num(tm.get(t)) or 0 for t in tiers]
        ax.bar([i + (j - (n - 1) / 2) * w for i in x], vals, w, label=m)
    ax.set_xticks(list(x))
    ax.set_xticklabels(tiers)
    ax.set_ylabel("mean success rate")
    ax.set_ylim(0, 1)
    ax.set_title("Skill gradient: mean SR by tier\n"
                 "(baseline should slope down expert->novice; AQI flatter)")
    ax.legend(title="method", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _fig_variance_decomposition(plt, report, out):
    ms = _methods_ordered(report)
    betw = [_num(report["methods"][m].get("sigma2_between_operator")) or 0
            for m in ms]
    within = [_num(report["methods"][m].get("sigma2_within_operator")) or 0
              for m in ms]
    x = range(len(ms))
    fig, ax = plt.subplots(figsize=(1.5 * len(ms) + 2, 4))
    ax.bar(list(x), betw, 0.6, label="between-operator (sigma2_op)",
           color="#c44e52")
    ax.bar(list(x), within, 0.6, bottom=betw,
           label="within-operator (sigma2_seed)", color="#8172b3")
    ax.set_xticks(list(x))
    ax.set_xticklabels(ms)
    ax.set_ylabel("variance of SR")
    ax.set_title("Variance decomposition per method\n"
                 "(red = the operator effect we want small)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _fig_operator_spread(plt, report, out):
    ms = _methods_ordered(report)
    cv = [_num(report["methods"][m].get("cv_operator")) or 0 for m in ms]
    pv = [_num(report["methods"][m].get("operator_matters_p")) for m in ms]
    x = range(len(ms))
    fig, ax = plt.subplots(figsize=(1.5 * len(ms) + 2, 4))
    bars = ax.bar(list(x), cv, 0.6, color=_METHOD_COLOR)
    ax.set_xticks(list(x))
    ax.set_xticklabels(ms)
    ax.set_ylabel("CV of per-operator SR")
    ax.set_title("Operator spread (CV) + 'operator matters' p-value\n"
                 "(lower CV and higher p = less operator-dependent)")
    for i, b in enumerate(bars):
        if pv[i] is not None:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"p={pv[i]:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _fig_steps_to_convergence(plt, report, out):
    import numpy as np
    ms = _methods_ordered(report)
    stbm = report.get("steps_by_method", {})
    conv = report.get("convergence", {})
    data, labels, fracs = [], [], []
    for m in ms:
        vals = []
        for op, d in stbm.get(m, {}).items():
            for s in d.get("steps", {}).values():
                v = _num(s)
                if v is not None:
                    vals.append(v)
        data.append(vals or [0])
        labels.append(m)
        fracs.append(_num(conv.get(m, {}).get("converged_fraction")))
    fig, ax = plt.subplots(figsize=(1.6 * len(ms) + 2, 4))
    ax.boxplot(data, showmeans=True)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("env steps to convergence")
    ax.set_title("Training effort per method\n"
                 "(annotated: fraction of cells that converged)")
    for i, fr in enumerate(fracs):
        if fr is not None:
            ax.text(i + 1, ax.get_ylim()[1] * 0.98, f"{100 * fr:.0f}% conv.",
                    ha="center", va="top", fontsize=8, color="#444444")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


# ===========================================================================
# entry point
# ===========================================================================
_FIGURES = [
    ("icc_by_method.png", _fig_icc_by_method),
    ("sr_by_operator.png", _fig_sr_by_operator),
    ("sr_by_tier.png", _fig_sr_by_tier),
    ("variance_decomposition.png", _fig_variance_decomposition),
    ("operator_spread.png", _fig_operator_spread),
    ("steps_to_convergence.png", _fig_steps_to_convergence),
]


def make_plots(campaign_dir) -> list:
    """Render all figures for a campaign. Returns the list of written paths.
    Raises ImportError if matplotlib is missing (caller decides what to do)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    campaign = Path(campaign_dir)
    rep = campaign / "report" / "dependency.json"
    if not rep.exists():
        raise FileNotFoundError(f"no {rep}; run the analyze stage first.")
    report = json.load(open(rep))
    plots_dir = campaign / "report" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for name, fn in _FIGURES:
        out = plots_dir / name
        try:
            fn(plt, report, out)
            written.append(out)
        except Exception as exc:  # one bad figure must not sink the rest
            print(f"[plots] {name} failed: {exc}")

    # Operator-intervention frequency curves (reconstructed from the actor's
    # replay buffers; independent of dependency.json). Best-effort.
    try:
        from intervention_curves import make_intervention_curves
        p = make_intervention_curves(str(campaign))
        if p:
            written.append(Path(p))
    except Exception as exc:
        print(f"[plots] intervention_curves failed: {exc}")
    return written


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign", required=True,
                    help="campaign directory (must contain report/dependency.json).")
    args = ap.parse_args(argv)
    paths = make_plots(args.campaign)
    print(f"[plots] wrote {len(paths)} figure(s) to "
          f"{Path(args.campaign) / 'report' / 'plots'}")
    for p in paths:
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
