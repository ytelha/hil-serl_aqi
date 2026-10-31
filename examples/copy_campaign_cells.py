#!/usr/bin/env python3
"""Copy finished cells of one campaign into a new campaign folder, so new arms can be compared with them.

Read-only on the source. For each selected cell it copies the results and logs, not the policy:
    checkpoints/<cell>/   without checkpoint_*, certified and wandb (so the runner sees `.done`)
    eval_runs/<cell>/     (so the runner sees `cell.json`)
    aqi_logs/<cell>_*/, logs/<cell>.*
plus what the new cells must share with them: operators/<op>.json, demos/<op>_s<seed>/, eval_grid.json.
Copied files are verified by hash and listed in <dst>/copied_cells.json.

Run it BEFORE the first run in the new folder; it refuses a folder that already has a cell or a campaign.json.
The runner keeps existing operators/*.json (frozen roster) and skips cells with `.done` and `cell.json`.

    python examples/copy_campaign_cells.py --src runs/aqi_oracle_quality --dst runs/quality_novice_ablation \\
        --arms logged,quality-oracle --operators nov00,nov01,nov02
"""
import argparse
import datetime
import fnmatch
import hashlib
import json
import os
import re
import shutil
import sys

CELL = re.compile(r"^(.+)_([a-z]+\d+)_s(\d+)$")
NOT_COPIED = ("checkpoint_*", "certified", "wandb")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def select_cells(src, arms, operators, seeds):
    """(cells, problems): the source's finished cells of these arms, operators and seeds."""
    root = os.path.join(src, "checkpoints")
    cells, problems, seen = [], [], set()
    for name in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        m = CELL.match(name)
        if not m or m.group(1) not in arms:
            continue
        if (operators and m.group(2) not in operators) or (seeds and m.group(3) not in seeds):
            continue
        seen.add(m.group(1))
        if not os.path.exists(os.path.join(root, name, ".done")):
            problems.append(f"{name}: no .done (not finished)")
        elif not os.path.exists(os.path.join(src, "eval_runs", name, "cell.json")):
            problems.append(f"{name}: no eval_runs/{name}/cell.json (not evaluated)")
        cells.append((name, m.group(2), m.group(3)))
    problems += [f"arm {a}: no cell selected" for a in arms if a not in seen]
    return cells, problems


def _ignore(_dir, names):
    return [n for n in names if any(fnmatch.fnmatch(n, p) for p in NOT_COPIED)]


def copy_cells(src, dst, cells):
    """Copy, then verify by hash what the new cells rely on. Returns the provenance record."""
    os.makedirs(dst, exist_ok=True)
    shutil.copyfile(os.path.join(src, "eval_grid.json"), os.path.join(dst, "eval_grid.json"))
    for name, _op, _seed in cells:
        shutil.copytree(os.path.join(src, "checkpoints", name), os.path.join(dst, "checkpoints", name), ignore=_ignore)
        shutil.copytree(os.path.join(src, "eval_runs", name), os.path.join(dst, "eval_runs", name))
        logs = os.path.join(src, "aqi_logs")
        for d in sorted(os.listdir(logs)) if os.path.isdir(logs) else []:
            if d.startswith(name + "_"):
                shutil.copytree(os.path.join(logs, d), os.path.join(dst, "aqi_logs", d))
        logs = os.path.join(src, "logs")
        for f in sorted(os.listdir(logs)) if os.path.isdir(logs) else []:
            if f.startswith(name + "."):
                os.makedirs(os.path.join(dst, "logs"), exist_ok=True)
                shutil.copyfile(os.path.join(logs, f), os.path.join(dst, "logs", f))
    hashes = {}
    os.makedirs(os.path.join(dst, "operators"), exist_ok=True)
    for op in sorted({c[1] for c in cells}):
        rel = os.path.join("operators", op + ".json")
        shutil.copyfile(os.path.join(src, rel), os.path.join(dst, rel))
        hashes[rel.replace(os.sep, "/")] = sha256(os.path.join(src, rel))
    for op, seed in sorted({(c[1], c[2]) for c in cells}):
        rel = os.path.join("demos", f"{op}_s{seed}")
        shutil.copytree(os.path.join(src, rel), os.path.join(dst, rel))
        hashes[(rel + "/demos.pkl").replace(os.sep, "/")] = sha256(os.path.join(src, rel, "demos.pkl"))
    for rel, h in hashes.items():
        if sha256(os.path.join(dst, rel)) != h:
            raise SystemExit(f"[copy] {rel}: the copy differs from the source")
    record = {"source": os.path.abspath(src), "copied": datetime.datetime.now().isoformat(timespec="seconds"),
              "cells": [c[0] for c in cells], "sha256": hashes}
    with open(os.path.join(dst, "copied_cells.json"), "w") as f:
        json.dump(record, f, indent=2)
    return record


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="finished campaign")
    ap.add_argument("--dst", required=True, help="new campaign folder")
    ap.add_argument("--arms", required=True, help="comma-separated arms")
    ap.add_argument("--operators", default="", help="comma-separated operators (default: all)")
    ap.add_argument("--seeds", default="", help="comma-separated seeds (default: all)")
    ap.add_argument("--dry", action="store_true", help="list the cells, copy nothing")
    a = ap.parse_args(argv)
    split = lambda s: [x for x in s.split(",") if x]   # noqa: E731
    cells, problems = select_cells(a.src, split(a.arms), split(a.operators), split(a.seeds))
    for name, _o, _s in cells:
        if os.path.exists(os.path.join(a.dst, "checkpoints", name)):
            problems.append(f"{name}: already in {a.dst}")
    if os.path.exists(os.path.join(a.dst, "campaign.json")):
        problems.append(f"{a.dst} has already been started (campaign.json): copy before the first run")
    if os.path.abspath(a.src) == os.path.abspath(a.dst):
        problems.append("source and destination are the same folder")
    for p in problems:
        print("[copy] PROBLEM:", p)
    if problems or not cells:
        print(f"[copy] nothing copied ({len(cells)} cells selected)")
        return 2
    print(f"[copy] {len(cells)} cells:", " ".join(c[0] for c in cells))
    if a.dry:
        return 0
    copy_cells(a.src, a.dst, cells)
    print(f"[copy] done; hashes verified; record in {os.path.join(a.dst, 'copied_cells.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
