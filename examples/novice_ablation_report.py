#!/usr/bin/env python3
"""Which part of the quality arms collapses the novice? Read-only; docs/novice_ablation.md explains the test.

    python examples/novice_ablation_report.py --count runs/aqi_oracle_quality/aqi_logs
        what each arm dropped, per tier, and why: reads an existing campaign's segments.jsonl
    python examples/novice_ablation_report.py --campaign runs/quality_novice_ablation
        the results of the test against the rule fixed in advance -> <campaign>/report/novice_ablation.{md,json}

Each arm's mean ruler SR is compared with the midpoint between ensemble extended and the oracle campaign's quality
arm: at or above it the arm has recovered, below it it has collapsed. Six cells per arm show direction, not
significance.
"""
import argparse
import collections
import glob
import json
import math
import os
import re
import sys

CELL = re.compile(r"^(.+)_([a-z]+\d+)_s(\d+)$")
ARMS = (("logged", "ensemble extended", "-", "-", "-"),
        ("quality-oracle", "as in the oracle campaign", "oracle", "default", "0.1"),
        ("oracle-nofilter", "without the hard filters", "oracle", "none", "0.1"),
        ("oracle-nobc", "replay priority only", "oracle", "none", "0"),
        ("filter-only", "the hard filters alone", "none", "default", "0"),
        ("bc-only", "the imitation loss alone", "none", "none", "0.1"))
MIN_GAP = 0.15   # below this, the two reference arms differ too little for a midpoint to mean anything


def _jsonl(path):
    out = []
    if os.path.exists(path):
        for line in open(path, encoding="utf-8", errors="replace"):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def count_dropped(log_root):
    """{(arm, tier prefix): counters} from aqi_logs/<arm>_<op>_s<seed>_actor/segments.jsonl."""
    t = collections.defaultdict(collections.Counter)
    for run in sorted(glob.glob(os.path.join(log_root, "*_actor"))):
        m = CELL.match(os.path.basename(run)[:-len("_actor")])
        if not m:
            continue
        c = t[(m.group(1), m.group(2)[:3])]
        for s in _jsonl(os.path.join(run, "segments.jsonl")):
            c["takeovers"] += 1
            c["steps"] += int(s.get("length", 0))
            c["in_failed_episode"] += not s.get("episode_success", False)
            c["q_positive"] += float(s.get("q_int_mean", 0) or 0) > 0
            if s.get("excluded"):
                c["dropped"] += 1
                c["steps_dropped"] += int(s.get("length", 0))
                c["why_" + str(s.get("exclusion_reason") or "unknown")] += 1
    return t


def render_count(t):
    L = ["| arm | tier | takeovers | dropped | operator steps dropped | in failed episodes | score above 0 | why |",
         "|---|---|---|---|---|---|---|---|"]
    for (arm, tier), c in sorted(t.items()):
        n, st = max(c["takeovers"], 1), max(c["steps"], 1)
        why = ", ".join(f"{k[4:]} {v}" for k, v in sorted(c.items()) if k.startswith("why_")) or "-"
        L.append(f"| {arm} | {tier} | {c['takeovers']} | {100 * c['dropped'] / n:.0f} % | {100 * c['steps_dropped'] / st:.0f} % | "
                 f"{100 * c['in_failed_episode'] / n:.0f} % | {100 * c['q_positive'] / n:.0f} % | {why} |")
    return "\n".join(L)


def read_cells(campaign):
    """{arm: {(op, seed): {sr, last_sr, converged}}}."""
    out = collections.defaultdict(dict)
    for cj in sorted(glob.glob(os.path.join(campaign, "eval_runs", "*", "cell.json"))):
        name = os.path.basename(os.path.dirname(cj))
        m = CELL.match(name)
        if not m:
            continue
        d = json.load(open(cj))
        sr = (d.get("outcomes") or {}).get("sr")
        if sr is None or sr != sr:
            continue
        per = d.get("sr_per_checkpoint") or [{}]
        done = os.path.join(campaign, "checkpoints", name, ".done")
        out[m.group(1)][(m.group(2), m.group(3))] = {
            "sr": float(sr), "last_sr": per[-1].get("sr"),
            "converged": bool(json.load(open(done)).get("converged")) if os.path.exists(done) else False}
    return out


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def sign_p(w, l):
    n = w + l
    if n == 0:
        return None
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(min(w, l) + 1)) / 2 ** n)


def paired(a, b):
    keys = sorted(set(a) & set(b))
    d = [a[k]["sr"] - b[k]["sr"] for k in keys]
    w, l = sum(x > 1e-9 for x in d), sum(x < -1e-9 for x in d)
    return {"pairs": len(d), "mean": _mean(d), "wins": w, "losses": l, "ties": len(d) - w - l, "p": sign_p(w, l)}


def reading(sr):
    """The rule of docs/novice_ablation.md, applied to {arm: mean SR}."""
    if any(sr.get(a[0]) is None for a in ARMS):
        return "incomplete", "Not all six arms have results yet."
    low, high = sr["quality-oracle"], sr["logged"]
    if high - low < MIN_GAP:
        return "no_gap", (f"Ensemble extended and the oracle campaign's quality arm differ by only "
                          f"{100 * (high - low):.0f} points in these cells, so there is no collapse to explain here.")
    mid = (low + high) / 2.0
    ok = {a: sr[a] >= mid for a in ("oracle-nofilter", "oracle-nobc", "filter-only", "bc-only")}
    F, B = not ok["filter-only"], not ok["bc-only"]
    if F and B:
        return "filter_and_bc", "The hard filters and the imitation loss each collapse the novice alone."
    if F and ok["oracle-nofilter"]:
        return "filter", ("The hard filters cause the collapse: they collapse the novice alone, and the oracle "
                          "without them recovers. The result is about filtering by episode outcome, not about the score.")
    if B and not ok["oracle-nofilter"] and ok["oracle-nobc"]:
        return "bc", ("The imitation loss causes the collapse: it collapses the novice alone, and the oracle "
                      "recovers once it is off. The result is about imitating novice corrections, not about the score.")
    if not F and not B and not ok["oracle-nobc"]:
        return "score", ("The oracle's replay priority causes the collapse: it collapses the novice with no filter and "
                         "no imitation loss, while those two alone do not. This supports the oracle campaign's conclusion.")
    if not F and not B and ok["oracle-nobc"] and not ok["oracle-nofilter"]:
        return "score_and_bc", "The collapse needs the oracle score together with the imitation loss; neither does it alone."
    if not F and not B and ok["oracle-nofilter"]:
        return "filter_with_score", "No single part collapses the novice; it takes the hard filters together with the other parts."
    return "mixed", "The arms do not fit one explanation; read the table."


def _pct(x, signed=False):
    return "–" if x is None else (f"{100 * x:+.1f}" if signed else f"{100 * x:.1f} %")


def _pair(p):
    return (f"{_pct(p['mean'], True)} ({p['wins']}/{p['losses']}/{p['ties']}, p {p['p']:.2f})"
            if p["pairs"] and p["p"] is not None else ("0.0 (ties)" if p["pairs"] else "–"))


def analyse(campaign):
    cells = read_cells(campaign)
    dropped = count_dropped(os.path.join(campaign, "aqi_logs"))
    res = {"arms": {}}
    for arm, *_ in ARMS:
        c = cells.get(arm, {})
        d = collections.Counter()
        for (a, _t), v in dropped.items():
            if a == arm:
                d.update(v)
        res["arms"][arm] = {
            "cells": len(c), "sr": _mean([x["sr"] for x in c.values()]), "last_sr": _mean([x["last_sr"] for x in c.values()]),
            "converged": sum(x["converged"] for x in c.values()),
            "dropped": d["dropped"] / d["takeovers"] if d["takeovers"] else None,
            "steps_dropped": d["steps_dropped"] / d["steps"] if d["steps"] else None,
            "vs_logged": paired(c, cells.get("logged", {})), "vs_quality": paired(c, cells.get("quality-oracle", {}))}
    res["verdict"], res["reading"] = reading({a: v["sr"] for a, v in res["arms"].items()})
    return res


def render(res, campaign):
    L = ["# Novice test: which part of the quality arms collapses the novice?", "", f"Source: `{campaign}`.", "",
         "| arm | what it is | score | hard filters | BC | cells | ruler SR | last ckpt SR | converged | takeovers dropped "
         "| operator steps dropped | vs ensemble extended | vs quality-oracle |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm, what, score, filters, bc in ARMS:
        v = res["arms"][arm]
        L.append(f"| {arm} | {what} | {score} | {filters} | {bc} | {v['cells']} | {_pct(v['sr'])} | {_pct(v['last_sr'])} | "
                 f"{v['converged']} | {_pct(v['dropped'])} | {_pct(v['steps_dropped'])} | {_pair(v['vs_logged'])} | "
                 f"{_pair(v['vs_quality'])} |")
    L += ["", "Paired: mean difference in points (wins/losses/ties, exact sign test) per operator and seed.", "",
          f"**Reading (`{res['verdict']}`):** {res['reading']}", "",
          "Six cells per arm show direction, not significance."]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", default=None, help="an aqi_logs folder: what each arm dropped, per tier")
    ap.add_argument("--campaign", default=None, help="the novice test's campaign folder")
    a = ap.parse_args(argv)
    if a.count:
        t = count_dropped(a.count)
        print(render_count(t))
        return 0 if t else 1
    if not a.campaign:
        ap.error("give --count or --campaign")
    res = analyse(a.campaign)
    text = render(res, a.campaign)
    out = os.path.join(a.campaign, "report")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "novice_ablation.md"), "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.join(out, "novice_ablation.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
