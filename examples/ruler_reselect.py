#!/usr/bin/env python3
"""What a shorter ruler gives for finished cells, recomputed from their saved trials (read-only by default).

The ruler scores each checkpoint on the same grid with deterministic actions, so scoring fewer checkpoints gives
the same per-checkpoint results. This tool keeps the subset a shortening would score (ruler_select.py: skip each
arm's demo pre-training, every n-th, the newest N) and recomputes each cell's score from
eval_runs/<cell>/cell.json and trials.jsonl. No simulator, GPU or JAX.

Report (read-only): per arm, the checkpoints scored, the share of ruler env steps (time) kept, and the cells whose
score would change, with the largest change. Use it before switching a campaign to a shorter ruler:

    python examples/ruler_reselect.py --campaign runs/active_ratematch --skip_pretrain --newest 8

--apply, only for the cells listed in <campaign>/copied_cells.json (reference cells copied from another campaign:
their policies were not copied, so they cannot be re-scored): rewrites their cell.json for this selection, so they
are scored like the new cells. The original stays as cell.full_ruler.json. Cells trained in that folder are never
touched, and nothing is written into the source campaign.

    python examples/ruler_reselect.py --campaign runs/synthetic_suite/C_reference --skip_pretrain --newest 8 --apply

Each cell's pre-training steps come from the learner command in checkpoints/<cell>/manifest.json
(--demo_pretrain / --demo_pretrain_steps); --pretrain_steps is used where that file is missing.

--verify checks the assumption all of this rests on: that a checkpoint's trials do not depend on which
checkpoints were scored before it. --check_cmd prints the command that scores one finished cell again with the
shorter ruler into /tmp/ruler_check/<cell> (the cell's own ruler command from logs/<cell>.eval.log, with the new
output folder and flags); run it on the GPU, then compare trial by trial (exit 0 when success and steps match):

    python examples/ruler_reselect.py --campaign runs/<campaign> --skip_pretrain --newest 8 --check_cmd <cell>
    python examples/ruler_reselect.py --verify /tmp/ruler_check/<cell> --against runs/<campaign>/eval_runs/<cell>
"""
import argparse
import datetime
import glob
import importlib.util
import json
import os
import re
import shlex
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import ruler_select as rs  # noqa: E402

CELL = re.compile(r"^(.+)_([a-z]+\d+)_s(\d+)$")
STATS = os.path.join(HERE, "..", "serl_launcher", "serl_launcher", "aqi", "stats.py")


def _stats():
    """serl_launcher/aqi/stats.py by path (the package's __init__ imports JAX; stats.py needs numpy only)."""
    spec = importlib.util.spec_from_file_location("aqi_stats_standalone", STATS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pretrain_steps(campaign, cell, default=None):
    """Demo pre-training steps of a cell's learner (0 if it opted out); ``default`` if unknown."""
    try:
        rep = json.load(open(os.path.join(campaign, "checkpoints", cell, "manifest.json"))).get("reproduce") or []
    except (OSError, ValueError):
        return default
    toks = " ".join(rep).split()
    if "--demo_pretrain_steps" not in toks:
        return default
    if "--nodemo_pretrain" in toks or "--demo_pretrain" not in toks:
        return 0
    try:
        return int(toks[toks.index("--demo_pretrain_steps") + 1])
    except (IndexError, ValueError):
        return default


def _sr(trials):
    clean = [t for t in trials if not t.get("intervened", False)]
    return sum(bool(t.get("success")) for t in clean) / len(clean) if clean else None


def summarise(select, points, k):
    """(summary SR, steps whose trials make up the cell's score) for a ruler --select mode."""
    if select == "best_window":
        return rs.best_window(points, k)
    if select == "best":
        sr, step = rs.best_single(points)
        return sr, ([step] if step is not None else None)
    if select == "mean":
        return rs.mean_all(points), [p["step"] for p in points]
    return None, None


def reselect_cell(campaign, cell, filt_args, k):
    """Recompute one cell. Returns a dict with old/new score, checkpoints and steps, or {"skip": reason}."""
    d = os.path.join(campaign, "eval_runs", cell)
    try:
        cj = json.load(open(os.path.join(d, "cell.json")))
        trials = [json.loads(line) for line in open(os.path.join(d, "trials.jsonl"), encoding="utf-8") if line.strip()]
    except (OSError, ValueError) as e:
        return {"skip": f"unreadable ({e.__class__.__name__})"}
    select = cj.get("select")
    if select not in ("best", "mean", "best_window"):
        return {"skip": f"select={select}: only the all-checkpoint sweep can be shortened"}
    if cj.get("used_certified"):
        return {"skip": "scored its certified checkpoint only"}
    points = [p for p in cj.get("sr_per_checkpoint") or [] if p.get("step") is not None]
    # self-check: the recorded score must come out of the recorded checkpoints
    old_sr, old_steps = summarise(select, points, k)
    rec_steps = {"best_window": cj.get("best_window_steps"), "best": [cj.get("best_step")]}.get(select)
    if rec_steps is not None and old_steps != rec_steps:
        return {"skip": f"recorded selection {rec_steps} not reproduced ({old_steps}); window_k?"}
    by_step = {}
    for t in trials:
        by_step.setdefault(t.get("checkpoint_step"), []).append(t)
    old_score = (cj.get("outcomes") or {}).get("sr")
    sel_old = [t for s in (old_steps or []) for t in by_step.get(s, [])]
    if old_score is not None and _sr(sel_old) is not None and abs(_sr(sel_old) - old_score) > 1e-9:
        return {"skip": "trials.jsonl does not reproduce outcomes.sr"}
    skip_pre, stride, newest, default_pre = filt_args
    min_step = 0
    if skip_pre:
        min_step = pretrain_steps(campaign, cell, default_pre)
        if min_step is None:
            return {"skip": "pre-training steps unknown (no manifest.json): pass --pretrain_steps"}
    want = rs.normalise(min_step, stride, newest)
    rec = cj.get("checkpoint_filter")
    already = False
    if rec:
        got = rs.normalise(rec.get("min_step", 0), rec.get("stride", 1), rec.get("newest", 0))
        if got != want:
            return {"skip": f"already scored with another selection {got}"}
        already = True          # scored (or re-selected) with this selection: nothing to do
    kept = ([p["step"] for p in points] if already
            else rs.keep_checkpoints([p["step"] for p in points], *want))
    kpts = [p for p in points if p["step"] in set(kept)]
    new_sr, new_steps = summarise(select, kpts, k)
    sel_new = [t for s in (new_steps or []) for t in by_step.get(s, [])]
    steps_all = sum(t.get("steps", 0) for t in trials)
    steps_kept = sum(t.get("steps", 0) for s in kept for t in by_step.get(s, []))
    return {"arm": (CELL.match(cell).group(1) if CELL.match(cell) else cell), "select": select,
            "old": old_score, "new": _sr(sel_new), "old_steps": old_steps, "new_steps": new_steps,
            "n_scored": len(points), "n_kept": len(kept), "kept": kept, "want": want,
            "ruler_steps": steps_all, "ruler_steps_kept": steps_kept, "summary_sr": new_sr,
            "cell_json": cj, "trials_score": sel_new, "already": already}


def apply_cell(campaign, cell, r):
    """Rewrite a copied cell's cell.json for the new selection; the original stays as cell.full_ruler.json."""
    d = os.path.join(campaign, "eval_runs", cell)
    orig = os.path.join(d, "cell.full_ruler.json")
    if not os.path.exists(orig):
        shutil.copyfile(os.path.join(d, "cell.json"), orig)
    cj = dict(r["cell_json"])
    stats = _stats()
    pts = [p for p in cj["sr_per_checkpoint"] if p["step"] in set(r["kept"])]
    bw_sr, bw_steps = rs.best_window(pts, r["k"])
    b_sr, b_step = rs.best_single(pts)
    m, s, n = r["want"]
    cj.update(
        best_step=b_step, best_sr=b_sr, mean_all_sr=rs.mean_all(pts),
        best_window_sr=bw_sr, best_window_steps=bw_steps,
        eval_last_k=len(pts), eval_checkpoint_steps=[p["step"] for p in pts], sr_per_checkpoint=pts,
        outcomes=stats.cell_outcomes(r["trials_score"], f_max=cj.get("f_max")),
        checkpoint_filter=dict(min_step=m, stride=s, newest=n, n_saved=r["n_scored"], n_scored=len(pts)),
        reselected={"tool": "examples/ruler_reselect.py", "from": "cell.full_ruler.json",
                    "datetime": datetime.datetime.now().isoformat(timespec="seconds"),
                    "note": "recomputed from trials.jsonl; no checkpoint was re-scored"})
    with open(os.path.join(d, "cell.json"), "w") as f:
        json.dump(cj, f, indent=2)


def _pp(x):
    return "   -  " if x is None else f"{100 * x:5.1f}"


def _read_trials(d):
    return {(t.get("checkpoint_step"), t.get("trial")): t
            for t in (json.loads(line) for line in open(os.path.join(d, "trials.jsonl"), encoding="utf-8")
                      if line.strip())}


def check_cmd(campaign, cell, want, out_root="/tmp/ruler_check"):
    """The cell's own ruler command (first line of logs/<cell>.eval.log), re-pointed to out_root/<cell> with the
    shorter ruler's flags, in the campaign's sim environment. None if the log is missing."""
    try:
        first = open(os.path.join(campaign, "logs", cell + ".eval.log"), encoding="utf-8").readline()
    except OSError:
        return None
    toks = shlex.split(first[1:].strip()) if first.startswith("#") else []
    if "eval_ruler.py" not in toks:
        return None
    i = toks.index("eval_ruler.py")
    head, args = toks[:i], toks[i + 1:]
    out, skip = [], False
    for j, t in enumerate(args):
        if skip:
            skip = False
            continue
        if t in ("--out", "--select_min_step", "--select_stride", "--select_newest"):
            skip = True
            continue
        out.append(t)
    m, s, n = want
    out += ["--out", f"{out_root}/{cell}"] + (["--select_min_step", str(m)] if m else []) \
        + (["--select_stride", str(s)] if s > 1 else []) + (["--select_newest", str(n)] if n else [])
    try:
        box = json.load(open(os.path.join(campaign, "campaign.json"))).get("sim_rpy_box") or []
    except (OSError, ValueError):
        box = []
    # head = ["source", ".../examples/experiments/sim_peg_insertion/sim_env.sh", "&&", "exec", "python"]
    if len(head) < 2 or head[0] != "source":
        return None
    inner = (f"source {shlex.quote(head[1])} && exec python eval_ruler.py "
             + " ".join(shlex.quote(t) for t in out))
    examples = head[1].rsplit("/experiments/", 1)[0] if "/experiments/" in head[1] else "examples"
    rpy = ",".join(f"{v:g}" for v in box)
    return (f"cd {shlex.quote(examples)} && SIM_DISABLE_SPACEMOUSE=1 SIM_RPY_BOX={shlex.quote(rpy)} "
            f"bash -c {shlex.quote(inner)}")


def verify(new_dir, old_dir):
    """Compare a fresh shorter-ruler run of one cell with its original trials. 0 = success and steps identical."""
    new, old = _read_trials(new_dir), _read_trials(old_dir)
    common = sorted(k for k in new if k in old)
    missing = sorted(k for k in new if k not in old)
    diff = [k for k in common if (bool(new[k].get("success")), new[k].get("steps"))
            != (bool(old[k].get("success")), old[k].get("steps"))]
    df = max((abs(float(new[k].get("f_peak") or 0) - float(old[k].get("f_peak") or 0)) for k in common), default=0.0)
    cks = sorted({k[0] for k in common})
    print(f"[verify] {len(common)} trials on {len(cks)} checkpoints {cks} compared"
          + (f"; {len(missing)} new trials have no original" if missing else ""))
    print(f"[verify] success and steps identical in {len(common) - len(diff)} of {len(common)}; "
          f"largest peak-force difference {df:.3g} N")
    for k in diff[:10]:
        print(f"  differs: checkpoint {k[0]} trial {k[1]}: original {old[k].get('success')}/{old[k].get('steps')} "
              f"steps, now {new[k].get('success')}/{new[k].get('steps')}")
    ok = bool(common) and not diff
    print("[verify] " + ("IDENTICAL: re-selecting finished cells from their trials is exact." if ok else
                         "NOT identical: a checkpoint's result depends on what ran before it; do not re-select "
                         "copied cells, score every cell of a comparison with the same ruler instead."))
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign", help="campaign folder (with eval_runs/)")
    ap.add_argument("--verify", default="", help="a fresh shorter-ruler eval folder of one cell (with trials.jsonl)")
    ap.add_argument("--against", default="", help="with --verify: that cell's original eval_runs/<cell> folder")
    ap.add_argument("--skip_pretrain", action="store_true", help="skip each cell's demo pre-training checkpoints")
    ap.add_argument("--stride", type=int, default=1, help="every n-th checkpoint, counted back from the newest")
    ap.add_argument("--newest", type=int, default=0, help="only the newest N checkpoints (0 = all)")
    ap.add_argument("--window_k", type=int, default=3, help="the campaign's eval.window_k (default 3)")
    ap.add_argument("--pretrain_steps", type=int, default=None,
                    help="pre-training steps where checkpoints/<cell>/manifest.json is missing")
    ap.add_argument("--apply", action="store_true",
                    help="rewrite cell.json of the cells in copied_cells.json (original kept as cell.full_ruler.json)")
    ap.add_argument("--json", default="", help="also write the per-cell table to this JSON file")
    ap.add_argument("--check_cmd", default="", metavar="CELL",
                    help="print the command that re-scores CELL with this selection into /tmp/ruler_check/CELL")
    a = ap.parse_args(argv)
    if a.verify:
        if not a.against:
            ap.error("--verify needs --against <original eval_runs/<cell> folder>")
        return verify(a.verify, a.against)
    if not a.campaign:
        ap.error("--campaign is required (or --verify ... --against ...)")
    if not (a.skip_pretrain or a.stride > 1 or a.newest > 0):
        ap.error("choose at least one of --skip_pretrain, --stride N (> 1), --newest N")
    camp = a.campaign
    if a.check_cmd:
        m = pretrain_steps(camp, a.check_cmd, a.pretrain_steps) if a.skip_pretrain else 0
        cmd = None if m is None else check_cmd(camp, a.check_cmd, rs.normalise(m, a.stride, a.newest))
        if cmd is None:
            print(f"[reselect] no ruler command for {a.check_cmd}: needs {camp}/logs/{a.check_cmd}.eval.log"
                  + (" and its manifest.json (or --pretrain_steps)" if m is None else ""))
            return 2
        print(cmd)
        return 0
    cells = sorted(os.path.basename(os.path.dirname(p))
                   for p in glob.glob(os.path.join(camp, "eval_runs", "*", "cell.json")))
    copied = set()
    if a.apply:
        try:
            copied = set(json.load(open(os.path.join(camp, "copied_cells.json")))["cells"])
        except (OSError, ValueError, KeyError):
            print(f"[reselect] --apply needs {os.path.join(camp, 'copied_cells.json')} (only copied cells are re-selected)")
            return 2
    filt = (a.skip_pretrain, a.stride, a.newest, a.pretrain_steps)
    rows, skipped = {}, {}
    for c in cells:
        r = reselect_cell(camp, c, filt, a.window_k)
        if "skip" in r:
            skipped[c] = r["skip"]
        else:
            r["k"] = a.window_k
            rows[c] = r
    print(f"[reselect] {camp}: {len(rows)} cells recomputed, {len(skipped)} skipped; selection: "
          + ", ".join((["skip pre-training"] if a.skip_pretrain else []) + ([f"every {a.stride}-th"] if a.stride > 1 else [])
                      + ([f"newest {a.newest}"] if a.newest else [])) + f", window {a.window_k}")
    for c, why in sorted(skipped.items()):
        print(f"  skipped {c}: {why}")
    if not rows:
        return 1
    print(f"\n  {'arm':22s} {'cells':>5s} {'checkpoints':>13s} {'ruler time':>10s} {'score now':>9s} {'shorter':>7s} "
          f"{'changed':>7s} {'largest':>8s}")
    arms = sorted({r["arm"] for r in rows.values()})
    for arm in arms + ["all"]:
        rr = [r for r in rows.values() if arm in ("all", r["arm"])]
        ch = [r for r in rr if r["old"] is not None and r["new"] is not None and abs(r["new"] - r["old"]) > 1e-9]
        big = max((r["new"] - r["old"] for r in ch), key=abs, default=0.0)
        mean = lambda xs: sum(xs) / len(xs) if xs else None  # noqa: E731
        t = sum(r["ruler_steps_kept"] for r in rr) / max(sum(r["ruler_steps"] for r in rr), 1)
        print(f"  {arm:22s} {len(rr):5d} {mean([r['n_scored'] for r in rr]):6.1f} -> {mean([r['n_kept'] for r in rr]):4.1f} "
              f"{100 * t:9.0f}% {_pp(mean([r['old'] for r in rr if r['old'] is not None]))} %"
              f" {_pp(mean([r['new'] for r in rr if r['new'] is not None]))} % {len(ch):7d} {100 * big:+7.1f}pp")
    changed = [(c, r) for c, r in rows.items()
               if r["old"] is not None and r["new"] is not None and abs(r["new"] - r["old"]) > 1e-9]
    if changed:
        print("\n  cells whose score changes (now -> shorter ruler):")
        for c, r in changed:
            print(f"    {c:28s} {_pp(r['old'])} -> {_pp(r['new'])} %   window {r['old_steps']} -> {r['new_steps']}")
    else:
        print("\n  no cell's score changes.")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({c: {k: v for k, v in r.items() if k not in ("cell_json", "trials_score")}
                       for c, r in rows.items()}, f, indent=2)
    if a.apply:
        todo = [c for c in rows if c in copied and not rows[c]["already"]]
        done = [c for c in rows if c in copied and rows[c]["already"]]
        refused = [c for c in rows if c not in copied]
        for c in todo:
            apply_cell(camp, c, rows[c])
        print(f"\n[reselect] --apply: rewrote cell.json of {len(todo)} copied cell(s) (originals in cell.full_ruler.json)"
              + (f"; {len(done)} already had this selection" if done else "")
              + (f"; left {len(refused)} cell(s) alone that were not copied" if refused else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
