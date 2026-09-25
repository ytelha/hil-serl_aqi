#!/usr/bin/env python3
"""Build a thesis-ready metric bundle from one campaign.

This script gathers campaign-linked AQI logs, computes intervention and
mechanism metrics, and writes clean tables/plots into a single folder.

Outputs (under <campaign>/report/thesis_bundle by default):
  - bundle_summary.md
  - bundle_manifest.json
  - method_metrics.csv
  - method_vs_baseline.csv
  - episode_curve.csv
  - time_curve.csv
  - aqi_analysis/summary.md
  - aqi_analysis/results.json
  - aqi_analysis figures (png/svg when matplotlib is available)
  - plots/intervention_over_episode.png
  - plots/intervention_rate_over_time.png

Example:
  ./.venv/bin/python examples/build_thesis_bundle.py \
      --campaign examples/runs/cuboid_hilserl_vs_quality_35m_1op_2seed
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

import aqi_analysis as AQ


def _read_json(path: Path, default):
    if not path.exists():
        return default
    with open(path) as f:
        return json.load(f)


def _safe_mean(xs: Sequence[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b not in (0.0, 0) else float("nan")


def _rate_curve(intervention_steps: Sequence[int], lengths: Sequence[int],
                bin_steps: int = 1000) -> Dict[str, List[float]]:
    """Intervention rate per env-step bin, equivalent to B.2 style r(t)."""
    iv = np.asarray(intervention_steps, dtype=np.float64)
    ln = np.asarray(lengths, dtype=np.float64)
    total = ln.sum()
    n_bins = int(np.ceil(total / bin_steps)) if total > 0 else 0
    num = np.zeros(n_bins)
    den = np.zeros(n_bins)

    t = 0.0
    for i in range(len(ln)):
        if ln[i] <= 0:
            continue
        remaining = ln[i]
        iv_rate = iv[i] / ln[i]
        pos = t
        while remaining > 0:
            b = min(int(pos // bin_steps), n_bins - 1)
            take = min(remaining, (b + 1) * bin_steps - pos)
            den[b] += take
            num[b] += take * iv_rate
            pos += take
            remaining -= take
        t += ln[i]

    r = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)
    return {
        "bin_ends": [float((b + 1) * bin_steps) for b in range(n_bins)],
        "r": [float(x) for x in r],
    }


def _expected_cell_ids(campaign_dir: Path, manifest: dict) -> List[str]:
    methods = manifest.get("methods", [])
    operators = manifest.get("operators", [])
    seeds = manifest.get("seeds", [])
    if methods and operators and seeds:
        out = []
        for m in methods:
            for o in operators:
                for s in seeds:
                    out.append(f"{m}_{o}_s{s}")
        return sorted(set(out))

    # Fallback for older manifests: infer from checkpoint directory names.
    ckpt = campaign_dir / "checkpoints"
    if ckpt.exists():
        return sorted(p.name for p in ckpt.iterdir() if p.is_dir())
    return []


def _collect_run_dirs(campaign_dir: Path, aqi_log_root: Path,
                      cell_ids: Sequence[str]) -> Tuple[List[Path], Dict[str, str]]:
    run_dirs = []
    source = {}

    for cid in cell_ids:
        d = aqi_log_root / f"{cid}_actor"
        if d.exists():
            run_dirs.append(d)
            source[cid] = str(d)

    # Secondary fallback: infer from actor logs if some ids were not found.
    logs_dir = campaign_dir / "logs"
    if logs_dir.exists():
        for p in logs_dir.glob("*.actor.log"):
            cid = p.name.replace(".actor.log", "")
            if cid in source:
                continue
            d = aqi_log_root / f"{cid}_actor"
            if d.exists():
                run_dirs.append(d)
                source[cid] = str(d)

    # Deterministic order.
    run_dirs = sorted(set(run_dirs))
    return run_dirs, source


def _method_from_cell_id(cell_id: str, methods: Sequence[str]) -> str:
    for m in methods:
        if cell_id.startswith(f"{m}_"):
            return m
    return cell_id.split("_", 1)[0]


def _method_from_run_name(run_name: str, methods: Sequence[str]) -> str:
    for m in methods:
        prefix = f"{m}_"
        if run_name.startswith(prefix):
            return m
    # Last-resort fallback if naming is non-standard.
    return run_name.split("_", 1)[0]


def _write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _make_custom_plots(episode_rows: List[dict], time_rows: List[dict],
                       out_dir: Path) -> List[str]:
    written = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return written

    # Plot 1: intervention fraction over episode index.
    fig, ax = plt.subplots(figsize=(7, 4))
    methods = sorted({r["method"] for r in episode_rows})
    for m in methods:
        sub = [r for r in episode_rows if r["method"] == m]
        x = [int(r["episode_index"]) for r in sub]
        y = [float(r["mean_intervention_frac"]) for r in sub]
        ax.plot(x, y, label=m)
    ax.set_xlabel("episode index")
    ax.set_ylabel("mean intervention fraction")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Intervention fraction over episodes")
    ax.legend(fontsize=8)
    p1 = out_dir / "intervention_over_episode.png"
    fig.savefig(p1, bbox_inches="tight", dpi=150)
    plt.close(fig)
    written.append(str(p1))

    # Plot 2: intervention rate r(t) over env steps.
    fig, ax = plt.subplots(figsize=(7, 4))
    methods = sorted({r["method"] for r in time_rows})
    for m in methods:
        sub = [r for r in time_rows if r["method"] == m]
        x = [float(r["bin_end_steps"]) for r in sub]
        y = [float(r["mean_r_t"]) for r in sub]
        ax.plot(x, y, label=m)
    ax.set_xlabel("env steps")
    ax.set_ylabel("intervention rate r(t)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Intervention rate over time")
    ax.legend(fontsize=8)
    p2 = out_dir / "intervention_rate_over_time.png"
    fig.savefig(p2, bbox_inches="tight", dpi=150)
    plt.close(fig)
    written.append(str(p2))

    return written


def build_bundle(campaign: Path, out_dir: Path, aqi_log_root: Path,
                 baseline_mode: str, fail_horizon: int,
                 time_bin_steps: int) -> dict:
    manifest = _read_json(campaign / "campaign.json", default={})
    methods = manifest.get("methods", [])

    if not methods:
        raise SystemExit(f"No methods in {campaign / 'campaign.json'}")

    if baseline_mode not in methods:
        baseline_mode = methods[0]

    cell_ids = _expected_cell_ids(campaign, manifest)
    run_dirs, source_map = _collect_run_dirs(campaign, aqi_log_root, cell_ids)

    expected_by_method = defaultdict(int)
    linked_by_method = defaultdict(int)
    for cid in cell_ids:
        expected_by_method[_method_from_cell_id(cid, methods)] += 1
    for cid in source_map:
        linked_by_method[_method_from_cell_id(cid, methods)] += 1

    if not run_dirs:
        raise SystemExit(
            "No campaign-linked AQI logs were found. "
            "Expected paths like <aqi_log_root>/<cell_id>_actor."
        )

    runs = AQ.discover_runs(run_dirs=[str(p) for p in run_dirs])
    if not runs:
        raise SystemExit("No readable runs with episodes.jsonl were found.")

    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    aqi_out = out_dir / "aqi_analysis"
    aqi_out.mkdir(parents=True, exist_ok=True)

    # Reuse the core analysis for RQ1/RQ2/RQ3.
    summaries = [AQ.run_summary(r) for r in runs]
    rq1 = AQ.rq1_operator_variance(summaries, baseline_mode,
                                   metric="final_success_rate")
    rq2, qdist = AQ.rq2_feature_analysis(runs)
    rq3 = AQ.rq3_trigger_calibration(runs, horizon=fail_horizon)

    results = {
        "baseline_mode": baseline_mode,
        "run_dirs": [str(p) for p in run_dirs],
        "runs": summaries,
        "rq1": rq1,
        "rq2": rq2,
        "rq3": rq3,
    }
    with open(aqi_out / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    AQ.write_summary_md(str(aqi_out), summaries, rq1, rq2, rq3, baseline_mode)
    AQ.make_figures(runs, summaries, rq1, qdist, rq3, str(aqi_out))

    # Method-level intervention and safety table.
    by_method = defaultdict(lambda: dict(
        n_runs=0,
        n_episodes=0,
        n_success=0,
        total_steps=0,
        intervention_steps=0,
        segments=0,
        excluded_segments=0,
        requests=0,
        hard_violations=0,
        f_peak=[]))

    episode_agg = defaultdict(lambda: defaultdict(list))
    time_agg = defaultdict(lambda: defaultdict(list))

    for run in runs:
        run_name = run.get("name", "")
        m = _method_from_run_name(run_name, methods)
        eps = run.get("episodes", [])
        if not eps:
            continue

        stat = by_method[m]
        stat["n_runs"] += 1

        iv = []
        ln = []
        for i, e in enumerate(eps):
            succ = float(bool(e.get("success", False)))
            length = int(e.get("length", 0))
            n_int_steps = int(e.get("n_intervention_steps", 0))
            n_seg = int(e.get("n_segments", 0))
            n_exc = int(e.get("n_excluded_segments", 0))
            n_req = int(e.get("n_requests", 0))
            n_hv = int(e.get("n_hard_violations", 0))
            fpk = float(e.get("f_peak", 0.0))

            stat["n_episodes"] += 1
            stat["n_success"] += int(succ)
            stat["total_steps"] += max(length, 0)
            stat["intervention_steps"] += max(n_int_steps, 0)
            stat["segments"] += max(n_seg, 0)
            stat["excluded_segments"] += max(n_exc, 0)
            stat["requests"] += max(n_req, 0)
            stat["hard_violations"] += max(n_hv, 0)
            stat["f_peak"].append(fpk)

            frac = _safe_div(float(n_int_steps), float(length)) if length > 0 else float("nan")
            episode_agg[m][i].append({
                "intervention_frac": frac,
                "success": succ,
                "requests": float(n_req),
            })

            iv.append(n_int_steps)
            ln.append(length)

        curve = _rate_curve(iv, ln, bin_steps=time_bin_steps)
        for bi, rv in enumerate(curve["r"]):
            time_agg[m][bi].append({
                "bin_end_steps": curve["bin_ends"][bi],
                "r_t": float(rv),
            })

    method_rows = []
    for m in methods:
        s = by_method.get(m)
        if not s or s["n_episodes"] == 0:
            continue
        ne = float(s["n_episodes"])
        row = {
            "method": m,
            "n_runs": s["n_runs"],
            "n_episodes": s["n_episodes"],
            "success_rate": _safe_div(float(s["n_success"]), ne),
            "intervention_step_frac": _safe_div(float(s["intervention_steps"]),
                                                 float(max(s["total_steps"], 1))),
            "interventions_per_episode": _safe_div(float(s["segments"]), ne),
            "requests_per_episode": _safe_div(float(s["requests"]), ne),
            "excluded_segments_per_episode": _safe_div(float(s["excluded_segments"]), ne),
            "hard_violations_per_episode": _safe_div(float(s["hard_violations"]), ne),
            "mean_f_peak": _safe_mean(s["f_peak"]),
        }
        method_rows.append(row)

    _write_csv(
        out_dir / "method_metrics.csv",
        method_rows,
        [
            "method", "n_runs", "n_episodes", "success_rate",
            "intervention_step_frac", "interventions_per_episode",
            "requests_per_episode", "excluded_segments_per_episode",
            "hard_violations_per_episode", "mean_f_peak",
        ],
    )

    coverage_rows = []
    for m in methods:
        expected = int(expected_by_method.get(m, 0))
        linked = int(linked_by_method.get(m, 0))
        coverage_rows.append({
            "method": m,
            "expected_cells": expected,
            "cells_with_aqi_logs": linked,
            "coverage_frac": _safe_div(float(linked), float(expected)) if expected > 0 else float("nan"),
        })
    _write_csv(
        out_dir / "log_coverage_by_method.csv",
        coverage_rows,
        ["method", "expected_cells", "cells_with_aqi_logs", "coverage_frac"],
    )

    base_row = next((r for r in method_rows if r["method"] == baseline_mode), None)
    ratio_rows = []
    if base_row is not None:
        for r in method_rows:
            if r["method"] == baseline_mode:
                continue
            ratio_rows.append({
                "method": r["method"],
                "baseline": baseline_mode,
                "delta_success_rate": r["success_rate"] - base_row["success_rate"],
                "ratio_intervention_step_frac": _safe_div(
                    r["intervention_step_frac"], base_row["intervention_step_frac"]),
                "ratio_interventions_per_episode": _safe_div(
                    r["interventions_per_episode"], base_row["interventions_per_episode"]),
                "ratio_requests_per_episode": _safe_div(
                    r["requests_per_episode"], base_row["requests_per_episode"]),
                "delta_mean_f_peak": r["mean_f_peak"] - base_row["mean_f_peak"],
            })
    _write_csv(
        out_dir / "method_vs_baseline.csv",
        ratio_rows,
        [
            "method", "baseline", "delta_success_rate",
            "ratio_intervention_step_frac", "ratio_interventions_per_episode",
            "ratio_requests_per_episode", "delta_mean_f_peak",
        ],
    )

    episode_rows = []
    for m in methods:
        for ei in sorted(episode_agg[m].keys()):
            vals = episode_agg[m][ei]
            if not vals:
                continue
            episode_rows.append({
                "method": m,
                "episode_index": ei,
                "n_runs_contributing": len(vals),
                "mean_intervention_frac": _safe_mean(
                    [v["intervention_frac"] for v in vals if np.isfinite(v["intervention_frac"])]),
                "mean_success": _safe_mean([v["success"] for v in vals]),
                "mean_requests": _safe_mean([v["requests"] for v in vals]),
            })

    _write_csv(
        out_dir / "episode_curve.csv",
        episode_rows,
        [
            "method", "episode_index", "n_runs_contributing",
            "mean_intervention_frac", "mean_success", "mean_requests",
        ],
    )

    time_rows = []
    for m in methods:
        for bi in sorted(time_agg[m].keys()):
            vals = time_agg[m][bi]
            if not vals:
                continue
            time_rows.append({
                "method": m,
                "bin_index": bi,
                "n_runs_contributing": len(vals),
                "bin_end_steps": _safe_mean([v["bin_end_steps"] for v in vals]),
                "mean_r_t": _safe_mean([v["r_t"] for v in vals]),
            })

    _write_csv(
        out_dir / "time_curve.csv",
        time_rows,
        [
            "method", "bin_index", "n_runs_contributing",
            "bin_end_steps", "mean_r_t",
        ],
    )

    custom_plot_paths = _make_custom_plots(episode_rows, time_rows, plots_dir)

    dependency = _read_json(campaign / "report" / "dependency.json", default={})
    dep_methods = list((dependency.get("methods") or {}).keys())

    bundle_manifest = {
        "campaign": str(campaign),
        "baseline_mode": baseline_mode,
        "aqi_log_root": str(aqi_log_root),
        "cells_expected": len(cell_ids),
        "cells_with_logs": len(source_map),
        "log_coverage_by_method": {
            m: {
                "expected_cells": int(expected_by_method.get(m, 0)),
                "cells_with_aqi_logs": int(linked_by_method.get(m, 0)),
            }
            for m in methods
        },
        "baseline_available_in_aqi_logs": bool(base_row is not None),
        "run_dirs": [str(p) for p in run_dirs],
        "methods_in_manifest": methods,
        "methods_in_dependency_report": dep_methods,
        "custom_plots": custom_plot_paths,
    }
    with open(out_dir / "bundle_manifest.json", "w") as f:
        json.dump(bundle_manifest, f, indent=2)

    # Human-readable index for daily reporting.
    md = []
    md.append("# Thesis Bundle")
    md.append("")
    md.append(f"- campaign: {campaign}")
    md.append(f"- baseline mode: {baseline_mode}")
    md.append(f"- expected cells: {len(cell_ids)}")
    md.append(f"- cells with linked AQI logs: {len(source_map)}")
    md.append("")
    md.append("## Files")
    md.append("")
    md.append("- method table: method_metrics.csv")
    md.append("- baseline comparison ratios: method_vs_baseline.csv")
    md.append("- per-method log coverage: log_coverage_by_method.csv")
    md.append("- intervention by episode table: episode_curve.csv")
    md.append("- intervention over time table: time_curve.csv")
    md.append("- intervention plots: plots/intervention_over_episode.png, plots/intervention_rate_over_time.png")
    md.append("- RQ bundle: aqi_analysis/summary.md, aqi_analysis/results.json")
    md.append("")
    md.append("## Claims Mapping")
    md.append("")
    md.append("- Dependency claim: campaign report/dependency.md and report/plots")
    md.append("- Efficiency claim: method_metrics.csv + episode_curve.csv + time_curve.csv")
    md.append("- Safety claim: hard_violations_per_episode and mean_f_peak in method_metrics.csv")
    md.append("- Mechanism claim: aqi_analysis/summary.md (RQ2, RQ3)")
    md.append("- Stability claim: campaign report/dependency.md (steps-to-convergence section)")
    md.append("")
    md.append("## Notes")
    md.append("")
    md.append("- This bundle only uses AQI logs linked to campaign cell IDs.")
    md.append("- If a method has no AQI logs, intervention/mechanism tables for that method are unavailable.")
    if base_row is None:
        md.append(
            f"- Baseline mode '{baseline_mode}' has no AQI logs in this bundle, so method_vs_baseline.csv is empty."
        )

    with open(out_dir / "bundle_summary.md", "w") as f:
        f.write("\n".join(md) + "\n")

    return bundle_manifest


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign", required=True,
                    help="Campaign directory, e.g. examples/runs/my_run")
    ap.add_argument("--out", default=None,
                    help="Output directory (default: <campaign>/report/thesis_bundle)")
    ap.add_argument("--aqi_log_root", default="examples/aqi_logs",
                    help="Root directory containing AQI log run folders")
    ap.add_argument("--baseline_mode", default=None,
                    help="Baseline mode for ratio table (default: first campaign method)")
    ap.add_argument("--fail_horizon", type=int, default=20,
                    help="Horizon for failure labels in trigger calibration")
    ap.add_argument("--time_bin_steps", type=int, default=1000,
                    help="Bin size in env steps for intervention rate r(t)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    campaign = Path(args.campaign).resolve()
    if not campaign.exists():
        raise SystemExit(f"Campaign path not found: {campaign}")

    out_dir = (Path(args.out).resolve()
               if args.out else campaign / "report" / "thesis_bundle")
    aqi_log_root = Path(args.aqi_log_root).resolve()

    manifest = _read_json(campaign / "campaign.json", default={})
    methods = manifest.get("methods", [])
    baseline_mode = args.baseline_mode or (methods[0] if methods else "logged")

    bundle = build_bundle(
        campaign=campaign,
        out_dir=out_dir,
        aqi_log_root=aqi_log_root,
        baseline_mode=baseline_mode,
        fail_horizon=args.fail_horizon,
        time_bin_steps=args.time_bin_steps,
    )

    print("[bundle] wrote thesis bundle to", out_dir)
    print("[bundle] methods:", ", ".join(bundle.get("methods_in_manifest", [])))
    print("[bundle] linked logs:", bundle.get("cells_with_logs"),
          "/", bundle.get("cells_expected"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
