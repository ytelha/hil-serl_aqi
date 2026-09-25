#!/usr/bin/env python3
"""Backfill README.md + manifest.json onto artifacts created before the
provenance layer existed.

Scans a root (default: the repo's ``examples/``) for:

  * ``run_<peg>_<mode>/``  standalone checkpoint dirs   -> a run README from the
                           mode encoded in the folder name (AQIConfig(mode));
  * ``aqi_logs/<run>/``    training-log dirs            -> a run README from that
                           run's ``meta.json`` (mode, operator, full config);
  * ``demo_data/*.pkl``    recorded/synthetic demos     -> an index README that
                           lists every demo file (+ ground-truth θ if present).

Campaign directories already carry their own per-artifact docs (written by
``run_campaign.py``), so they are skipped. Use ``--dry_run`` to preview.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
for _p in ("serl_launcher", "examples"):
    _pp = str(REPO / _p)
    if _pp not in sys.path:
        sys.path.insert(0, _pp)

import provenance as prov  # noqa: E402

try:
    from serl_launcher.aqi import AQIConfig, MODES
except Exception:                       # pragma: no cover
    AQIConfig = None
    MODES = ("hilserl", "logged", "active", "quality", "aqi")


def _conditions_for_mode(mode, config_dict=None):
    """Rows from the exact logged config if available, else a canonical
    AQIConfig(mode); falls back to a bare mode row."""
    try:
        if config_dict:
            cfg = SimpleNamespace(**config_dict)
            cfg.mode = config_dict.get("mode", mode)
            return prov.condition_rows(cfg)
        if AQIConfig is not None:
            return prov.condition_rows(AQIConfig(mode=mode))
    except Exception:
        pass
    return [{"choice": "aqi_mode", "value": mode, "status": "mode",
             "note": prov.MODE_ONE_LINER.get(mode, "")}]


def _has_checkpoint(d: Path) -> bool:
    return d.is_dir() and any(p.name.startswith("checkpoint_") and p.is_dir()
                              for p in d.iterdir())


def backfill_run_dir(d: Path, dry: bool) -> bool:
    """Run README from the directory name — handles both naming conventions:
    legacy  `run_<peg>_<mode>/`  and  new  `<task>_<mode>_<op>_s<seed>/`."""
    name = d.name
    if name.startswith("run_"):
        parts = name[4:].split("_")
        mode = parts[-1] if parts and parts[-1] in MODES else None
        peg = "_".join(parts[:-1]) if len(parts) > 1 else None
        op_id = "unknown"
    else:
        parts = name.split("_")
        # new: <task>_<mode>_<operator>_s<seed>
        mode = parts[1] if len(parts) >= 2 and parts[1] in MODES else None
        peg = parts[0] if parts else None
        op_id = parts[2] if len(parts) >= 3 else "unknown"
    if mode is None:
        return False
    if dry:
        print(f"[backfill] would document run dir {d} (mode={mode})")
        return True
    return prov.document(
        d, "run", quiet=True,
        task={"exp_name": "sim_peg_insertion", "peg_shape": peg},
        method=mode, operator={"operator_id": op_id,
                               "note": "backfilled from directory name"},
        conditions=_conditions_for_mode(mode),
        extra={"backfilled": True})


def backfill_aqi_log(meta_path: Path, dry: bool) -> bool:
    """`aqi_logs/<run>/meta.json` -> run README from the logged config."""
    try:
        meta = json.load(open(meta_path))
    except (ValueError, OSError):
        return False
    d = meta_path.parent
    mode = meta.get("mode")
    cfg = meta.get("config", {})
    syn = cfg.get("synthetic_operator") or meta.get("synthetic_operator")
    op = None
    if isinstance(syn, dict):
        op = {"operator_id": syn.get("operator_id", meta.get("operator_id")),
              "tier": syn.get("tier"), "s_exec": syn.get("s_exec"),
              "s_mon": syn.get("s_mon"),
              **{k: syn.get(k) for k in ("sigma0", "v_max", "sigma_perc",
                                         "delta_in", "p_miss", "rho_fa",
                                         "l_react")}}
    elif meta.get("operator_id"):
        op = {"operator_id": meta["operator_id"]}
    if dry:
        print(f"[backfill] would document aqi_log {d} (mode={mode})")
        return True
    rn = meta.get("run_name", "")
    exp = (rn.split(f"_{mode}_")[0] if mode and f"_{mode}_" in rn
           else "sim_peg_insertion")
    return prov.document(
        d, "run", quiet=True,
        task={"exp_name": exp},
        method=mode, operator=op,
        conditions=_conditions_for_mode(mode, cfg),
        extra={"backfilled": True, "run_name": meta.get("run_name"),
               "start_time": meta.get("start_time")})


def backfill_demo_data(demo_dir: Path, dry: bool) -> bool:
    """Write a single index README summarising legacy flat demo files."""
    pkls = sorted(demo_dir.glob("*.pkl"))
    if not pkls:
        return False
    if dry:
        print(f"[backfill] would index {len(pkls)} demo files in {demo_dir}")
        return True
    L = ["# demo_data — recorded demonstrations (index)", "",
         "Legacy flat demo files (pre-provenance). New synthetic demos are "
         "stored as self-describing folders under a campaign's `demos/`.", "",
         "| file | ground truth | notes |", "|---|---|---|"]
    for p in pkls:
        gt = p.with_name(os.path.splitext(p.name)[0] + "_groundtruth.json")
        tier = "-"
        if gt.exists():
            try:
                tier = json.load(open(gt)).get("theta", {}).get("tier", "-")
            except (ValueError, OSError):
                pass
        L.append(f"| `{p.name}` | "
                 f"{'`' + gt.name + '`' if gt.exists() else '-'} | "
                 f"tier {tier} |")
    (demo_dir / "README.md").write_text("\n".join(L) + "\n")
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(REPO / "examples"),
                    help="directory to scan (default: examples/).")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args(argv)
    root = Path(args.root)
    n = 0

    run_dirs = list(root.glob("run_*"))
    runs_subdir = root / "runs"
    if runs_subdir.is_dir():
        run_dirs += list(runs_subdir.iterdir())
    for d in sorted(run_dirs):
        if _has_checkpoint(d) and backfill_run_dir(d, args.dry_run):
            n += 1
    for meta in sorted(root.glob("aqi_logs/*/meta.json")):
        if backfill_aqi_log(meta, args.dry_run):
            n += 1
    dd = root / "demo_data"
    if dd.is_dir() and backfill_demo_data(dd, args.dry_run):
        n += 1

    verb = "would document" if args.dry_run else "documented"
    print(f"[backfill] {verb} {n} artifact(s) under {root}.")
    print("[backfill] note: campaign directories already carry their own "
          "per-artifact READMEs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
