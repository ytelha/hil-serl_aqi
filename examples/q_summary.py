#!/usr/bin/env python3
"""Per-episode intervention-quality (q_int) summary and plots for a campaign.

Reads the AQI segment logs written by AQILogger (segments.jsonl + episodes.jsonl
+ meta.json, one directory per actor run under <campaign>/aqi_logs/) and, for
every quality / aqi cell, produces:

    report/q_analysis/<cell>_episode_q.csv   per-episode q_int statistics
    report/q_analysis/all_episode_q.csv      the same rows for every cell
    report/q_analysis/<cell>_q.png           per-cell: q_int per episode
                                             (mean +/- p10-p90 band, per-step
                                             scatter, success/fail markers) and
                                             the per-step q_int histogram
    report/q_analysis/q_by_tier.png          episode-mean q_int boxplot by tier
    report/q_analysis/q_by_method.png        ... by method (quality vs aqi)
    report/q_analysis/summary.{md,json}      per-cell + per-group aggregates

Only the counterfactual score q_int is summarised here (the graded signal that
drives the replay priority); the imitation weight w is included for reference.
Episodes with no operator takeover appear with empty q columns so the x-axis is
the full episode timeline.

numpy only; matplotlib is imported lazily (figures are skipped, CSV/JSON still
written, if it is unavailable).

Usage:
    python q_summary.py --campaign runs/rq2_quality_replayonly
    python q_summary.py --log_root examples/aqi_logs --out /tmp/q_out
"""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

QUALITY_MODES = ("quality", "aqi")
_TIER_BY_PREFIX = {"exp": "expert", "int": "intermediate", "nov": "novice"}


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
def _read_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _tier_of(operator_id):
    return _TIER_BY_PREFIX.get(str(operator_id)[:3], "unknown")


def _parse_run(run_dir, meta):
    """(method, operator, tier, seed, cell) from meta + the run directory name.
    run_name convention: <method>_<operator>_s<seed>_actor."""
    operator = meta.get("operator_id", "unknown")
    name = meta.get("run_name") or os.path.basename(run_dir.rstrip("/"))
    base = name[:-6] if name.endswith("_actor") else name
    seed = None
    mo = base
    if "_s" in base:
        mo, seed_s = base.rsplit("_s", 1)
        seed = int(seed_s) if seed_s.isdigit() else seed_s
    method = meta.get("mode", "unknown")
    arm = (meta.get("config") or {}).get("arm")
    if arm:                                  # campaign arm, recorded by the actor
        method = arm
    elif mo.endswith("_" + str(operator)):
        method = mo[: -(len(str(operator)) + 1)]
    elif "_" in mo:
        method = mo.rsplit("_", 1)[0]
    cell = f"{method}_{operator}_s{seed}"
    return method, operator, _tier_of(operator), seed, cell


def load_quality_runs(log_root):
    """Every quality/aqi actor run under log_root that has scored segments."""
    runs = []
    for meta_path in sorted(glob.glob(os.path.join(log_root, "*", "meta.json"))):
        run_dir = os.path.dirname(meta_path)
        try:
            meta = json.load(open(meta_path))
        except (ValueError, OSError):
            continue
        if meta.get("mode") not in QUALITY_MODES:
            continue
        segments = _read_jsonl(os.path.join(run_dir, "segments.jsonl"))
        if not segments:
            continue
        episodes = _read_jsonl(os.path.join(run_dir, "episodes.jsonl"))
        method, operator, tier, seed, cell = _parse_run(run_dir, meta)
        runs.append(dict(dir=run_dir, meta=meta, method=method,
                         operator=operator, tier=tier, seed=seed, cell=cell,
                         segments=segments, episodes=episodes))
    return runs


# ---------------------------------------------------------------------------
# Per-episode aggregation
# ---------------------------------------------------------------------------
_EPISODE_FIELDS = [
    "episode", "success", "n_segments", "n_intervention_steps",
    "n_excluded_segments", "q_mean", "q_median", "q_max", "q_p90",
    "q_pos_frac", "q_sum", "w_mean",
]


def _stats(q):
    q = np.asarray(q, dtype=np.float64)
    if q.size == 0:
        return dict(q_mean=np.nan, q_median=np.nan, q_max=np.nan, q_p90=np.nan,
                    q_pos_frac=np.nan, q_sum=np.nan)
    return dict(
        q_mean=float(q.mean()), q_median=float(np.median(q)),
        q_max=float(q.max()), q_p90=float(np.percentile(q, 90)),
        q_pos_frac=float(np.mean(q > 0.0)),
        q_sum=float(np.clip(q, 0.0, None).sum()),
    )


def episode_table(run):
    """List of per-episode dicts (spine = episodes.jsonl, q from segments).
    Also returns {episode: per-step q_int array} for the scatter/band."""
    by_ep_q = defaultdict(list)
    by_ep_w = defaultdict(list)
    by_ep_nseg = defaultdict(int)
    by_ep_nexcl = defaultdict(int)
    by_ep_nsteps = defaultdict(int)
    by_ep_success = {}
    for seg in run["segments"]:
        ep = seg.get("episode")
        by_ep_q[ep] += list(seg.get("q_int") or [])
        by_ep_w[ep] += list(seg.get("weights") or [])
        by_ep_nseg[ep] += 1
        by_ep_nexcl[ep] += int(bool(seg.get("excluded")))
        by_ep_nsteps[ep] += int(seg.get("length", len(seg.get("q_int") or [])))
        by_ep_success[ep] = bool(seg.get("episode_success",
                                         by_ep_success.get(ep, False)))

    spine = [e.get("episode") for e in run["episodes"]] or sorted(by_ep_q)
    success_spine = {e.get("episode"): bool(e.get("success"))
                     for e in run["episodes"]}

    rows, q_by_ep = [], {}
    for ep in spine:
        q = by_ep_q.get(ep, [])
        q_by_ep[ep] = np.asarray(q, dtype=np.float64)
        w = np.asarray(by_ep_w.get(ep, []), dtype=np.float64)
        row = dict(
            episode=ep,
            success=success_spine.get(ep, by_ep_success.get(ep)),
            n_segments=by_ep_nseg.get(ep, 0),
            n_intervention_steps=by_ep_nsteps.get(ep, 0),
            n_excluded_segments=by_ep_nexcl.get(ep, 0),
            w_mean=float(w.mean()) if w.size else np.nan,
            **_stats(q),
        )
        rows.append(row)
    return rows, q_by_ep


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------
def _write_cell_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_EPISODE_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in _EPISODE_FIELDS})


def _write_combined_csv(path, runs_rows):
    cols = ["method", "tier", "operator", "seed", "cell"] + _EPISODE_FIELDS
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for run, rows in runs_rows:
            for r in rows:
                w.writerow({"method": run["method"], "tier": run["tier"],
                            "operator": run["operator"], "seed": run["seed"],
                            "cell": run["cell"],
                            **{k: r.get(k) for k in _EPISODE_FIELDS}})


# ---------------------------------------------------------------------------
# Plots (lazy matplotlib)
# ---------------------------------------------------------------------------
def _cell_plot(plt, run, rows, q_by_ep, out_png):
    eps = [r["episode"] for r in rows]
    q_mean = np.array([r["q_mean"] for r in rows], dtype=np.float64)
    lo = np.array([np.percentile(q_by_ep[e], 10) if q_by_ep[e].size else np.nan
                   for e in eps])
    hi = np.array([np.percentile(q_by_ep[e], 90) if q_by_ep[e].size else np.nan
                   for e in eps])
    xs = np.arange(len(eps))

    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [3, 1]})

    ax0.axhline(0.0, color="0.7", lw=0.8, zorder=0)
    band = ~(np.isnan(lo) | np.isnan(hi))
    if band.any():
        ax0.fill_between(xs[band], lo[band], hi[band], color="#4c72b0",
                         alpha=0.20, label="p10-p90 (per step)")
    # faint per-step scatter
    for x, e in zip(xs, eps):
        q = q_by_ep[e]
        if q.size:
            ax0.scatter(np.full(q.shape, x), q, s=6, color="#4c72b0",
                        alpha=0.20, linewidths=0)
    ax0.plot(xs, q_mean, "-o", ms=3, color="#1f3b70", label="episode mean q_int")
    # success/fail markers on the mean line
    for x, r in zip(xs, rows):
        if r["n_segments"] and not np.isnan(r["q_mean"]):
            ok = bool(r["success"])
            ax0.scatter([x], [r["q_mean"]], s=28, zorder=5,
                        marker="^" if ok else "v",
                        color="#2ca02c" if ok else "#d62728")
    ax0.set_xlabel("episode (with intervention shown as points)")
    ax0.set_ylabel("q_int")
    ax0.set_title(f"{run['cell']}  [{run['method']}/{run['tier']}]  "
                  f"q_int per episode")
    step = max(1, len(eps) // 12)
    ax0.set_xticks(xs[::step])
    ax0.set_xticklabels([str(eps[i]) for i in range(0, len(eps), step)])
    ax0.legend(loc="upper right", fontsize=8)

    all_q = np.concatenate([q_by_ep[e] for e in eps if q_by_ep[e].size]) \
        if any(q_by_ep[e].size for e in eps) else np.zeros(0)
    if all_q.size:
        ax1.hist(all_q, bins=30, color="#4c72b0", alpha=0.85)
        ax1.axvline(0.0, color="0.4", lw=0.8)
        ax1.axvline(float(all_q.mean()), color="#d62728", lw=1.2,
                    label=f"mean {all_q.mean():.3f}")
        ax1.legend(fontsize=8)
    ax1.set_title("per-step q_int")
    ax1.set_xlabel("q_int")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def _group_box(plt, groups, title, ylabel, out_png):
    """groups: ordered list of (label, values)."""
    groups = [(k, np.asarray(v, dtype=np.float64)) for k, v in groups
              if len(v)]
    if not groups:
        return False
    fig, ax = plt.subplots(figsize=(1.6 * len(groups) + 2, 4.2))
    ax.axhline(0.0, color="0.8", lw=0.8, zorder=0)
    labels = [k for k, _ in groups]
    data = [v for _, v in groups]
    try:                                    # matplotlib >= 3.9 renamed the kwarg
        ax.boxplot(data, tick_labels=labels, showmeans=True)
    except TypeError:
        ax.boxplot(data, labels=labels, showmeans=True)
    # jittered points
    for i, (_, v) in enumerate(groups, start=1):
        x = np.random.default_rng(0).normal(i, 0.05, size=len(v))
        ax.scatter(x, v, s=8, alpha=0.35, color="#4c72b0", linewidths=0)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def make_q_report(campaign=None, log_root=None, out=None):
    """Build the per-episode q_int CSVs, plots and summary.

    campaign: a campaign dir; reads <campaign>/aqi_logs and writes
              <campaign>/report/q_analysis. log_root/out override either side.
    Returns the output directory (str) or None if there was nothing to do.
    """
    if log_root is None:
        if campaign is None:
            raise ValueError("make_q_report needs campaign or log_root")
        log_root = os.path.join(str(campaign), "aqi_logs")
    if out is None:
        out = (os.path.join(str(campaign), "report", "q_analysis")
               if campaign else "q_analysis_out")

    runs = load_quality_runs(log_root)
    if not runs:
        print(f"[q_summary] no quality/aqi segment logs under {log_root}")
        return None
    os.makedirs(out, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        plt = None

    runs_rows = []
    per_cell_summary = []
    by_tier_ep_q = defaultdict(list)
    by_method_ep_q = defaultdict(list)

    for run in sorted(runs, key=lambda r: r["cell"]):
        rows, q_by_ep = episode_table(run)
        runs_rows.append((run, rows))
        _write_cell_csv(os.path.join(out, f"{run['cell']}_episode_q.csv"), rows)
        if plt is not None:
            try:
                _cell_plot(plt, run, rows, q_by_ep,
                           os.path.join(out, f"{run['cell']}_q.png"))
            except Exception as exc:
                print(f"[q_summary] cell plot failed for {run['cell']}: {exc}")

        ep_means = [r["q_mean"] for r in rows
                    if r["n_segments"] and not np.isnan(r["q_mean"])]
        all_q = np.concatenate([q_by_ep[e] for e in q_by_ep
                                if q_by_ep[e].size]) \
            if any(q_by_ep[e].size for e in q_by_ep) else np.zeros(0)
        by_tier_ep_q[run["tier"]] += ep_means
        by_method_ep_q[run["method"]] += ep_means
        per_cell_summary.append(dict(
            cell=run["cell"], method=run["method"], tier=run["tier"],
            operator=run["operator"], seed=run["seed"],
            n_episodes=len(rows),
            n_episodes_with_intervention=int(sum(
                1 for r in rows if r["n_segments"])),
            q_mean_overall=float(np.mean(all_q)) if all_q.size else float("nan"),
            q_pos_frac_overall=(float(np.mean(all_q > 0))
                                if all_q.size else float("nan")),
            q_max_overall=float(all_q.max()) if all_q.size else float("nan"),
            episode_q_mean=(float(np.mean(ep_means))
                            if ep_means else float("nan")),
        ))

    _write_combined_csv(os.path.join(out, "all_episode_q.csv"), runs_rows)

    if plt is not None:
        try:
            tier_order = [t for t in ("expert", "intermediate", "novice",
                                      "unknown") if by_tier_ep_q.get(t)]
            _group_box(plt, [(t, by_tier_ep_q[t]) for t in tier_order],
                       "episode-mean q_int by operator tier",
                       "episode-mean q_int", os.path.join(out, "q_by_tier.png"))
            if len(by_method_ep_q) > 1:
                _group_box(plt, sorted(by_method_ep_q.items()),
                           "episode-mean q_int by method", "episode-mean q_int",
                           os.path.join(out, "q_by_method.png"))
        except Exception as exc:
            print(f"[q_summary] aggregate plot failed: {exc}")

    summary = dict(
        log_root=log_root, n_cells=len(runs),
        per_cell=per_cell_summary,
        by_tier={t: dict(n=len(v), mean=float(np.mean(v)) if v else float("nan"),
                         median=float(np.median(v)) if v else float("nan"))
                 for t, v in by_tier_ep_q.items()},
        by_method={m: dict(n=len(v),
                           mean=float(np.mean(v)) if v else float("nan"))
                   for m, v in by_method_ep_q.items()},
    )
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2)

    lines = ["# Intervention-quality (q_int) per-episode summary\n",
             f"cells: {len(runs)}  (source: {log_root})\n",
             "## Per cell\n",
             "| cell | method | tier | episodes | w/ intervention | "
             "mean q_int | q>0 frac | episode-mean q |",
             "|---|---|---|---|---|---|---|---|"]
    for c in per_cell_summary:
        lines.append(
            f"| `{c['cell']}` | {c['method']} | {c['tier']} | "
            f"{c['n_episodes']} | {c['n_episodes_with_intervention']} | "
            f"{c['q_mean_overall']:.4f} | {c['q_pos_frac_overall']:.2f} | "
            f"{c['episode_q_mean']:.4f} |")
    lines += ["\n## By tier (episode-mean q_int)\n",
              "| tier | n episodes | mean | median |", "|---|---|---|---|"]
    for t in ("expert", "intermediate", "novice", "unknown"):
        if t in summary["by_tier"]:
            s = summary["by_tier"][t]
            lines.append(f"| {t} | {s['n']} | {s['mean']:.4f} | "
                         f"{s['median']:.4f} |")
    open(os.path.join(out, "summary.md"), "w").write("\n".join(lines) + "\n")
    print(f"[q_summary] {len(runs)} cell(s) -> {out}")
    return out


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--campaign", default=None,
                   help="campaign dir: reads <campaign>/aqi_logs, writes "
                        "<campaign>/report/q_analysis.")
    p.add_argument("--log_root", default=None,
                   help="explicit AQI log root (overrides <campaign>/aqi_logs).")
    p.add_argument("--out", default=None,
                   help="output dir (default <campaign>/report/q_analysis).")
    return p


def main():
    args = _build_arg_parser().parse_args()
    if not args.campaign and not args.log_root:
        raise SystemExit("provide --campaign or --log_root")
    make_q_report(campaign=args.campaign, log_root=args.log_root, out=args.out)


if __name__ == "__main__":
    main()
