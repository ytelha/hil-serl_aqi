"""Phase 3 of the extension study: per-run measures and the fixed contrasts (docs/extension_phase3.md).

    python examples/phase3_analysis.py --root runs/phase3 --out results/phase3

Reads the raw run folders only (no JAX). Per run:
  success at B     mean ruler success of the last three environment-step checkpoints (16 000, 18 000 and the final
                   one at the budget B = 20 000): 3 x 9 = 27 trials
  AULC             mean ruler success over all environment-step checkpoints (2 000 ... 20 000)
  would-be stop    the environment step where the mini-evaluation would have stopped the run (WOULD_STOP.json)
  operator steps   until the would-be stop (the old effort measure; runs without one: over the whole budget, marked)
                   and over the whole budget (secondary: asking keeps asking until the end)
  careless share   careless takeovers / all takeovers (episodes.jsonl), every arm
  E2HiL keep       mean keep fraction of robot, demonstration, careful-takeover and careless-takeover states
                   (learner_metrics.jsonl), E2HiL arms only
Contrasts are paired by operator and seed; exact two-sided sign and Wilcoxon signed-rank tests; Holm within each
hypothesis on its primary measure. Diverged runs (DIVERGED.json, also after their one retry) and crashed runs never
enter a contrast; their pair is left out and listed.
"""

import argparse
import csv
import itertools
import json
import math
import os
import re
from pathlib import Path

B_LAST = 3                     # checkpoints averaged for success at B
CELL_RE = re.compile(r"^(?P<arm>.+)_(?P<op>(?P<tier>exp|int|nov)\d+)_s(?P<seed>\d+)$")
TIERS = {"exp": "expert", "int": "intermediate", "nov": "novice"}

# (hypothesis, label, measure, kind, spec); kind "diff": A - B; "inter": (A - B) - (C - D)
# spec entries are (campaign, arm); the first contrast of each hypothesis is its primary one where noted in the doc
CONTRASTS = [
    ("H4b grading under imitation", "imitate, discard lowest 40 % vs imitate all", "success_at_B", "diff",
     [("X1_careless", "bc-discard"), ("X1_careless", "bc")]),
    ("H4b grading under imitation", "imitate, discard lowest 40 % vs imitate, random 40 % out", "success_at_B", "diff",
     [("X1_careless", "bc-discard"), ("X1_careless", "bc-random")]),
    ("H4b grading under imitation", "careless vs careful operator, imitate all (cost)", "success_at_B", "diff",
     [("X1_careless", "bc"), ("X2_careful", "bc")]),
    ("H3' asking", "asks 15 % vs operator decides (operator steps until the would-be stop)", "op_steps_to_stop", "diff",
     [("X1_careless", "asks-15"), ("X1_careless", "keepall")]),
    ("H3' asking", "asks 15 % vs random requests at the same rate (operator steps until the would-be stop)",
     "op_steps_to_stop", "diff", [("X1_careless", "asks-15"), ("X1_careless", "random-15")]),
    ("H3' asking", "asks 15 % vs operator decides (success at B)", "success_at_B", "diff",
     [("X1_careless", "asks-15"), ("X1_careless", "keepall")]),
    ("H5' combination", "interaction: (AQI - asks 15 % with imitation) - (discard - imitate all)", "success_at_B",
     "inter", [("X1_careless", "aqi"), ("X1_careless", "asks-15-bc"), ("X1_careless", "bc-discard"),
               ("X1_careless", "bc")]),
    ("H5' combination", "AQI vs HIL-SERL", "success_at_B", "diff",
     [("X1_careless", "aqi"), ("X1_careless", "hilserl")]),
    ("H6 E2HiL", "E2HiL vs HIL-SERL, careless operator", "success_at_B", "diff",
     [("X1_careless", "e2hil"), ("X1_careless", "hilserl")]),
    ("H6 E2HiL", "E2HiL vs HIL-SERL, careful operator", "success_at_B", "diff",
     [("X2_careful", "e2hil"), ("X2_careful", "hilserl")]),
    ("H6 E2HiL", "interaction: E2HiL's gain with the careless minus with the careful operator", "success_at_B",
     "inter", [("X1_careless", "e2hil"), ("X1_careless", "hilserl"), ("X2_careful", "e2hil"),
               ("X2_careful", "hilserl")]),
    ("H6 E2HiL", "E2HiL (minimum over critics) vs mask only (mean over critics)", "success_at_B", "diff",
     [("X1_careless", "e2hil"), ("X1_careless", "e2hil-mean")]),
]
SECONDARY = ("AULC", "success_at_B", "op_steps_to_stop", "op_steps_total")
KEEP_CAMPAIGN = "X1_careless"   # where E2HiL meets careful and careless takeovers
SKIP_CAMPAIGNS = ("X0_probe",)  # training only, no ruler: its own report (PROBE.md)
DESCRIPTIVE = [("X1_careless", "bc-perfect", "the ceiling of grading under imitation (reads the label)")]


# ---------------------------------------------------------------------------- reading
def _json(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _jsonl(p):
    out = []
    try:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        pass
    except OSError:
        pass
    return out


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def env_curve(ckpt_dir, cell_json):
    """[(env step, ruler success, final?)] of the scored environment-step checkpoints, sorted by env step."""
    rows = _jsonl(Path(ckpt_dir) / "env_checkpoints.jsonl")
    by_step = {}
    for r in rows:
        ls = int(r["learner_step"])
        if ls not in by_step or r.get("final"):
            by_step[ls] = r
    scored = {int(c["step"]): float(c["sr"]) for c in (cell_json or {}).get("sr_per_checkpoint") or []
              if _num(c.get("sr")) is not None}
    out = [(int(r["env_steps"]), scored[ls], bool(r.get("final"))) for ls, r in by_step.items() if ls in scored]
    return sorted(out)


def success_at_b(curve, last=B_LAST):
    if len(curve) < last:
        return None
    return sum(v for _, v, _ in curve[-last:]) / last


def aulc(curve):
    return sum(v for _, v, _ in curve) / len(curve) if curve else None


def operator_steps(episodes, until=None):
    """(operator steps, env steps, takeovers, careless takeovers) over the episodes that ended at or before `until`."""
    op = env = tk = ck = 0
    for e in episodes:
        if until is not None and int(e.get("step", 0)) > until:
            break
        op += int(e.get("intervention_steps", 0))
        env += int(e.get("length", 0))
        tk += int(e.get("intervention_count", 0))
        ck += int(e.get("careless_takeovers", 0))
    return op, env, tk, ck


def _suffix(m, name):
    """The value of a learner metric by its own name, whatever group it is logged under ("actor/<name>")."""
    for k, v in m.items():
        if k == name or k.endswith("/" + name):
            return v
    return None


def e2hil_keep(metrics_rows):
    """Mean keep fraction per source over the online learner metrics, where the source was in the batch."""
    acc = {}
    for r in metrics_rows:
        if r.get("phase") != "online":
            continue
        m = r.get("metrics") or {}
        for src in ("online", "demo", "intervention", "careless", "takeover"):
            n, k = _num(_suffix(m, f"e2hil_{src}_count")), _num(_suffix(m, f"e2hil_{src}_keep_frac"))
            if n and k is not None:
                acc.setdefault(src, []).append(k)
    return {src: sum(v) / len(v) for src, v in acc.items() if v}


def read_run(camp, cid):
    camp = Path(camp)
    m = CELL_RE.match(cid)
    if not m:
        return None
    ck = camp / "checkpoints" / cid
    cj = _json(camp / "eval_runs" / cid / "cell.json")
    eps = _jsonl(ck / "episodes.jsonl")
    ws = _json(ck / "WOULD_STOP.json")
    div = _json(ck / "DIVERGED.json")
    crashed = (ck / ".crashed").exists() and not (ck / ".done").exists()
    archived = sorted(p.name for p in (camp / "diverged").glob(f"{cid}.*.json")) if (camp / "diverged").exists() else []
    curve = env_curve(ck, cj)
    stop = int(ws["step"]) if ws and ws.get("step") is not None else None
    op_stop, env_stop, _, _ = operator_steps(eps, stop)
    op_all, env_all, tk_all, ck_all = operator_steps(eps)
    ok = cj is not None and div is None and not crashed and len(curve) >= B_LAST
    why = ("diverged" if div is not None else "crashed" if crashed else "no ruler result" if cj is None
           else f"only {len(curve)} scored env-step checkpoints" if len(curve) < B_LAST else "")
    return {"campaign": camp.name, "cell": cid, "arm": m["arm"], "operator": m["op"], "tier": TIERS[m["tier"]],
            "seed": int(m["seed"]), "valid": ok, "why_not": why, "retried_after_divergence": len(archived),
            "success_at_B": success_at_b(curve) if ok else None, "AULC": aulc(curve) if ok else None,
            "curve": [[s, round(v, 4)] for s, v, _ in curve], "would_stop": stop,
            "op_steps_to_stop": op_stop if ok else None, "stop_censored": stop is None,
            "op_steps_total": op_all if ok else None, "env_steps": env_all, "takeovers": tk_all,
            "careless_share": (ck_all / tk_all) if tk_all else None,
            "operator_share": (op_all / env_all) if env_all else None,
            "e2hil_keep": e2hil_keep(_jsonl(ck / "learner_metrics.jsonl")) if m["arm"].startswith("e2hil") else {}}


def read_root(root):
    runs = []
    for camp in sorted(Path(root).iterdir()) if Path(root).exists() else []:
        ck = camp / "checkpoints"
        if not ck.is_dir() or camp.name in SKIP_CAMPAIGNS:
            continue
        for d in sorted(ck.iterdir()):
            if d.is_dir():
                r = read_run(camp, d.name)
                if r:
                    runs.append(r)
    return runs


# ---------------------------------------------------------------------------- statistics
def sign_test(diffs):
    pos, neg = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    p = 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p)


def wilcoxon_exact(diffs):
    """Exact two-sided Wilcoxon signed-rank p (zeros dropped, ties get average ranks; enumeration, n <= 16)."""
    d = [x for x in diffs if x != 0]
    n = len(d)
    if n == 0:
        return 1.0
    a = sorted(abs(x) for x in d)
    rank = {}
    i = 0
    while i < n:
        j = i
        while j + 1 < n and a[j + 1] == a[i]:
            j += 1
        rank[a[i]] = (i + j + 2) / 2.0
        i = j + 1
    r = [rank[abs(x)] for x in d]
    w = sum(ri for ri, x in zip(r, d) if x > 0)
    mean = sum(r) / 2.0
    if n > 16:                                          # normal approximation
        sd = math.sqrt(sum(ri * ri for ri in r) / 4.0)
        z = abs(w - mean) / sd if sd else 0.0
        return min(1.0, math.erfc(z / math.sqrt(2)))
    dev = abs(w - mean)
    hits = 0
    for signs in itertools.product((0, 1), repeat=n):
        if abs(sum(ri for ri, s in zip(r, signs) if s) - mean) >= dev - 1e-9:
            hits += 1
    return hits / 2 ** n


def holm(ps):
    """Holm-adjusted p values, same order."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [None] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def _index(runs):
    return {(r["campaign"], r["arm"], r["operator"], r["seed"]): r for r in runs}


def contrast(runs, kind, spec, measure):
    idx = _index(runs)
    keys = sorted({(r["operator"], r["seed"]) for r in runs if (r["campaign"], r["arm"]) == spec[0]})
    diffs, pairs, left_out = [], [], []
    for op, seed in keys:
        rs = [idx.get((c, a, op, seed)) for c, a in spec]
        if any(r is None or not r["valid"] or r.get(measure) is None for r in rs):
            left_out.append(f"{op} s{seed}")
            continue
        v = [r[measure] for r in rs]
        d = v[0] - v[1] if kind == "diff" else (v[0] - v[1]) - (v[2] - v[3])
        diffs.append(d)
        pairs.append({"pair": f"{op} s{seed}", "values": v, "diff": d})
    n = len(diffs)
    return {"n": n, "mean_diff": (sum(diffs) / n) if n else None,
            "better": sum(d > 0 for d in diffs), "worse": sum(d < 0 for d in diffs),
            "same": sum(d == 0 for d in diffs), "sign_p": sign_test(diffs), "wilcoxon_p": wilcoxon_exact(diffs),
            "pairs": pairs, "left_out": left_out}


def analyse(runs):
    out, by_h = [], {}
    for h, label, measure, kind, spec in CONTRASTS:
        c = contrast(runs, kind, spec, measure)
        c.update(hypothesis=h, label=label, measure=measure, kind=kind,
                 spec=[f"{a} ({camp})" for camp, a in spec])
        c["secondary"] = {m: contrast(runs, kind, spec, m) for m in SECONDARY if m != measure}
        for m in c["secondary"].values():
            m.pop("pairs", None)
        out.append(c)
        by_h.setdefault(h, []).append(c)
    for cs in by_h.values():
        for c, p in zip(cs, holm([c["wilcoxon_p"] for c in cs])):
            c["holm_p"] = p
    e2 = {}
    for arm in ("e2hil", "e2hil-mean"):           # the main campaign only: careful and careless takeovers
        keep = []
        for r in runs:
            if r["campaign"] == KEEP_CAMPAIGN and r["arm"] == arm and r["valid"]:
                k = r["e2hil_keep"]
                if k.get("careless") is not None and k.get("intervention") is not None:
                    keep.append({"run": r["cell"], "careless": k["careless"], "careful": k["intervention"],
                                 "diff": k["careless"] - k["intervention"]})
        kd = [x["diff"] for x in keep]
        e2[arm] = {"n": len(kd), "mean_diff": (sum(kd) / len(kd)) if kd else None,
                   "lower": sum(d < 0 for d in kd), "higher": sum(d > 0 for d in kd),
                   "sign_p": sign_test(kd), "wilcoxon_p": wilcoxon_exact(kd), "runs": keep}
    return out, e2


# ---------------------------------------------------------------------------- report
def _f(v, pct=False, nd=1):
    if v is None:
        return "–"
    return f"{100 * v:.{nd}f} %" if pct else (f"{v:,.0f}".replace(",", " ") if abs(v) >= 100 else f"{v:.{nd}f}")


def _p(v):
    return "–" if v is None else f"{v:.3f}"


def tier_means(runs):
    """{(campaign, arm): {tier: mean success at B}} over valid runs."""
    acc = {}
    for r in runs:
        if r["valid"] and r["success_at_B"] is not None:
            acc.setdefault((r["campaign"], r["arm"]), {}).setdefault(r["tier"], []).append(r["success_at_B"])
    return {k: {t: sum(v) / len(v) for t, v in by.items()} for k, by in acc.items()}


def arm_table(runs):
    rows = {}
    for r in runs:
        rows.setdefault((r["campaign"], r["arm"]), []).append(r)
    L = ["| campaign | arm | runs (valid) | success at B | AULC | operator steps until the would-be stop | "
         "no would-be stop | operator steps, whole budget | careless share |", "|---|---|---|---|---|---|---|---|---|"]
    for (camp, arm), rs in sorted(rows.items()):
        v = [r for r in rs if r["valid"]]

        def mean(k):
            xs = [r[k] for r in v if r.get(k) is not None]
            return sum(xs) / len(xs) if xs else None
        L.append(f"| {camp} | {arm} | {len(rs)} ({len(v)}) | {_f(mean('success_at_B'), True)} | "
                 f"{_f(mean('AULC'), True)} | {_f(mean('op_steps_to_stop'))} | "
                 f"{sum(r['stop_censored'] for r in v)} | {_f(mean('op_steps_total'))} | "
                 f"{_f(mean('careless_share'), True)} |")
    return L


def render(runs, cs, e2):
    L = ["# Phase 3 of the extension study: results", "",
         "Generated by `examples/phase3_analysis.py` from the raw runs. Budget B = 20 000 environment steps for every "
         "run; success at B = mean ruler success of the last three environment-step checkpoints (27 trials). "
         "Contrasts are paired by operator and seed (exact sign and Wilcoxon tests; Holm within each hypothesis on "
         "its listed measure). Diverged or crashed runs never enter a contrast.", "", "## Arms", ""]
    L += arm_table(runs)
    L += ["", "## Operator dependency (H1'): success at B by tier", "",
          "| campaign | arm | expert | intermediate | novice | expert - novice |", "|---|---|---|---|---|---|"]
    for (camp, arm), by in sorted(tier_means(runs).items()):
        gap = (by["expert"] - by["novice"]) if by.get("expert") is not None and by.get("novice") is not None else None
        L.append(f"| {camp} | {arm} | {_f(by.get('expert'), True)} | {_f(by.get('intermediate'), True)} | "
                 f"{_f(by.get('novice'), True)} | {'–' if gap is None else f'{100 * gap:+.1f} pp'} |")
    L += ["", "## Contrasts", "",
          "| hypothesis | contrast | measure | pairs | better / worse / same | mean difference | sign p | Wilcoxon p "
          "| Holm p | left out |", "|---|---|---|---|---|---|---|---|---|---|"]
    for c in cs:
        pct = c["measure"] in ("success_at_B", "AULC")
        md = None if c["mean_diff"] is None else (100 * c["mean_diff"] if pct else c["mean_diff"])
        L.append(f"| {c['hypothesis']} | {c['label']} | {c['measure']} | {c['n']} | {c['better']} / {c['worse']} / "
                 f"{c['same']} | {'–' if md is None else (f'{md:+.1f} pp' if pct else f'{md:+.0f}')} | "
                 f"{_p(c['sign_p'])} | {_p(c['wilcoxon_p'])} | {_p(c.get('holm_p'))} | "
                 f"{', '.join(c['left_out']) or '–'} |")
    L += ["", "## Does E2HiL's entropy criterion drop careless corrections?", "",
          "Per run of the main campaign: the share of careless-takeover states that E2HiL keeps in the policy update "
          "minus the share of careful-takeover states it keeps (both averaged over the logged learner steps).", "",
          "| arm | runs | lower for careless | higher | mean difference | sign p | Wilcoxon p |",
          "|---|---|---|---|---|---|---|"]
    for arm, e in e2.items():
        md = "–" if e["mean_diff"] is None else f"{100 * e['mean_diff']:+.1f} pp"
        L.append(f"| {arm} | {e['n']} | {e['lower']} | {e['higher']} | {md} | "
                 f"{_p(e['sign_p'])} | {_p(e['wilcoxon_p'])} |")
    L.append("")
    bad = [r for r in runs if not r["valid"]]
    if bad:
        L += ["## Runs without a result", ""] + [f"- {r['campaign']}/{r['cell']}: {r['why_not']}" for r in bad] + [""]
    L += ["## Every run", "", "| campaign | run | tier | success at B | AULC | ruler per env checkpoint | would-be stop | "
          "operator steps to it | whole budget | careless share | retried after divergence |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        curve = " ".join(f"{s // 1000}k:{round(100 * v)}" for s, v in r["curve"])
        L.append(f"| {r['campaign']} | {r['cell']} | {r['tier']} | {_f(r['success_at_B'], True)} | "
                 f"{_f(r['AULC'], True)} | {curve or '–'} | {_f(r['would_stop'])} | {_f(r['op_steps_to_stop'])} | "
                 f"{_f(r['op_steps_total'])} | {_f(r['careless_share'], True)} | {r['retried_after_divergence']} |")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default="runs/phase3")
    ap.add_argument("--out", default="results/phase3")
    a = ap.parse_args(argv)
    runs = read_root(a.root)
    cs, e2 = analyse(runs)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "PHASE3_REPORT.md").write_text(render(runs, cs, e2), encoding="utf-8")
    (out / "phase3_contrasts.json").write_text(json.dumps({"contrasts": cs, "e2hil_keep": e2}, indent=2),
                                               encoding="utf-8")
    cols = ["campaign", "cell", "arm", "operator", "tier", "seed", "valid", "why_not", "success_at_B", "AULC",
            "would_stop", "op_steps_to_stop", "stop_censored", "op_steps_total", "env_steps", "takeovers",
            "careless_share", "operator_share", "retried_after_divergence"]
    with open(out / "phase3_runs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in runs:
            w.writerow(r)
    print(f"[phase3] {len(runs)} runs, {sum(r['valid'] for r in runs)} valid -> {out / 'PHASE3_REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
