#!/usr/bin/env python3
"""How well do the quality scores find the bad corrections of an inconsistent operator?

With --synthetic_inconsistent the synthetic operator is attentive or inattentive at every
takeover, drawn at random and independent of the situation, and every segment in
aqi_logs/<run>/segments.jsonl carries that hidden `condition`. It is a ground-truth label of
correction quality that does not depend on when or how long the operator intervened (the
timing leakage that invalidated the earlier score audit). This script rebuilds every score
of examples/quality_signal_audit.py from the logs and reports its AUROC for "attentive"
(higher score = better correction), with a 95 % CI that resamples whole episodes, per tier.

    python examples/inconsistent_operator_audit.py --campaign runs/inconsistent_operator/Q1_validity_orig
    python examples/inconsistent_operator_audit.py --campaign runs/inconsistent_operator/Q2_pilot_orig
    python examples/inconsistent_operator_audit.py --log_root <aqi_logs> --out report_dir

Pre-registered reading (docs/inconsistent_operator.md):
    valid     AUROC >= 0.70 and the CI's lower bound > 0.5
    weak      0.5 < AUROC < 0.70, or the CI includes 0.5
    invalid   AUROC <= 0.5 (no information or inverted)
The onset yardsticks (takeover start time, critic value at the onset) must come out near 0.5;
if not, the condition is not independent of the situation and the label is confounded (report,
do not use). The end-time yardstick may differ: inattentive takeovers last longer by design.

On a Q2 or Q3 campaign it also reports, per discard arm, what was dropped after the quality
warm-up: precision (dropped takeovers that were inattentive) and recall (inattentive takeovers
that were dropped).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import quality_signal_audit as qsa  # noqa: E402

# score of the audit -> quality_score_mode that uses it online (None = no online mode)
ONLINE_MODE = {
    "a_hat_mean": "counterfactual",
    "B = z(dV)-z(t)-z(F)": "behavioral",
    "a_hind (hindsight)": "hindsight",
    "d_progress_post": "progress",
    "outcome (episode success)": "outcome",
}
YARDSTICKS = ("-start_step (time only)", "-end_step (time only)", "V(onset) (difficulty only)")
# the onset yardsticks must be ~0.5: the condition is drawn at the onset, independent of the
# situation. The end of a takeover is not: inattentive takeovers last longer, by design.
ONSET_YARDSTICKS = ("-start_step (time only)", "V(onset) (difficulty only)")


def verdict(a, ci):
    if a != a:
        return "n/a"
    if a <= 0.5:
        return "invalid"
    if a >= 0.70 and ci[0] == ci[0] and ci[0] > 0.5:
        return "valid"
    return "weak"


def analyse(rows, n_boot=500):
    rows = [r for r in rows if r.get("condition") in ("attentive", "inattentive")]
    last_ep = {}
    for r in rows:
        r["attentive"] = r["condition"] == "attentive"
        last_ep[r["run"]] = max(last_ep.get(r["run"], 0), int(r.get("episode", 0) or 0))
    # the second half of each run: critic-based scores are noise before the critic has seen
    # rewards (finding F4), so the early takeovers pull their AUROC towards 0.5
    late = [r for r in rows if int(r.get("episode", 0) or 0) >= last_ep[r["run"]] / 2.0]
    scores = list(qsa.SCORES) + [("outcome (episode success)", lambda r: float(r["success"]),
                                  "+1 if the episode succeeded")]
    out = {"n_segments": len(rows),
           "n_runs": len({r["run"] for r in rows}),
           "inattentive_share": {t: (float(np.mean([not r["attentive"] for r in rows if r["tier"] == t]))
                                     if any(r["tier"] == t for r in rows) else None) for t in qsa.TIERS},
           "length_by_condition": {c: (float(np.mean([r["t_int"] for r in rows if r["condition"] == c]))
                                       if any(r["condition"] == c for r in rows) else None)
                                   for c in ("attentive", "inattentive")},
           "success_by_condition": {c: (float(np.mean([r["success"] for r in rows if r["condition"] == c]))
                                        if any(r["condition"] == c for r in rows) else None)
                                    for c in ("attentive", "inattentive")},
           "scores": {}}
    for name, get, note in scores:
        s, y, idx = qsa._score_vectors(rows, get, label_key="attentive")
        a = qsa.auroc(s, y) if len(s) else float("nan")
        ci = (qsa.cluster_bootstrap_auroc(s, y, [rows[i]["cluster"] for i in idx], n_boot=n_boot)
              if len(s) else (float("nan"),) * 2)
        ls, ly, _ = qsa._score_vectors(late, get, label_key="attentive")
        by_tier = {}
        for t in qsa.TIERS:
            ts, ty, _ = qsa._score_vectors([r for r in rows if r["tier"] == t], get, label_key="attentive")
            by_tier[t] = qsa.auroc(ts, ty) if len(ts) else float("nan")
        out["scores"][name] = {"note": note, "n": int(len(s)), "auroc": a, "ci95": list(ci),
                               "auroc_late": qsa.auroc(ls, ly) if len(ls) else float("nan"),
                               "by_tier": by_tier, "online_mode": ONLINE_MODE.get(name),
                               "verdict": "yardstick" if name in YARDSTICKS else verdict(a, ci)}
    usable = [(v["auroc"], n) for n, v in out["scores"].items()
              if v["online_mode"] and v["auroc"] == v["auroc"]]
    best = max(usable) if usable else None
    out["best_online_score"] = ({"score": best[1], "auroc": best[0],
                                 "quality_score_mode": ONLINE_MODE[best[1]],
                                 "verdict": out["scores"][best[1]]["verdict"]} if best else None)
    ys = [out["scores"][n]["auroc"] for n in ONSET_YARDSTICKS if n in out["scores"]]
    out["label_confounded"] = bool(any(abs(a - 0.5) > 0.1 for a in ys if a == a))
    out["discard"] = discard_quality(rows)
    return out


DISCARD_REASONS = ("score_filter", "oracle_filter", "random_filter")


def discard_quality(rows):
    """Per arm with a discard filter (Q2, Q3): how well it drops the inattentive takeovers and keeps the
    attentive ones, after the quality warm-up (before it nothing is dropped)."""
    out = {}
    for arm in sorted({r["arm"] for r in rows}):
        rs = [r for r in rows if r["arm"] == arm and r.get("drop_warmup") is False]
        if not rs:
            continue
        drop = np.array([r.get("exclusion_reason") in DISCARD_REASONS for r in rs])
        bad = np.array([not r["attentive"] for r in rs])
        mean = lambda a: float(np.mean(a)) if a.size else float("nan")   # noqa: E731
        out[arm] = {"n": int(len(rs)), "inattentive_share": mean(bad), "dropped_share": mean(drop),
                    "precision": mean(bad[drop]),                  # dropped ones that were inattentive
                    "recall": mean(drop[bad]),                     # inattentive ones that were dropped
                    "attentive_dropped": mean(drop[~bad]),         # good ones lost
                    "kept_inattentive_share": mean(bad[~drop])}    # bad ones left in the buffer
    return out


def _f(x, nd=2):
    return "–" if x is None or x != x else f"{x:.{nd}f}"


def render(res, source):
    L = ["# Inconsistent operator – do the quality scores find the bad corrections?", "",
         f"Source: `{source}` · {res['n_segments']} takeovers with a condition label from {res['n_runs']} runs.", "",
         "Label: the operator's hidden condition at the takeover (attentive = 1). AUROC = chance that a random "
         "attentive takeover scores higher than a random inattentive one; 0.5 = no information, < 0.5 = inverted. "
         "CI: 95 %, resampling whole episodes. 2nd half: the takeovers of each run's second half of episodes "
         "(critic-based scores need a trained critic).", "",
         "| | expert | intermediate | novice |", "|---|---|---|---|",
         "| inattentive share | " + " | ".join(_f(res["inattentive_share"][t]) for t in qsa.TIERS) + " |", "",
         f"Mean takeover length: attentive {_f(res['length_by_condition']['attentive'], 1)} steps, inattentive "
         f"{_f(res['length_by_condition']['inattentive'], 1)} steps. Episode success: attentive "
         f"{_f(res['success_by_condition']['attentive'])}, inattentive {_f(res['success_by_condition']['inattentive'])}.", ""]
    if res["label_confounded"]:
        L += ["**Warning: an onset yardstick (start time or V at the onset) is far from 0.5 – the condition is not "
              "independent of the situation, so these AUROCs are confounded. Report; do not choose a score from them.**", ""]
    L += ["| score | online mode | n | AUROC | 95 % CI | 2nd half | expert | intermediate | novice | verdict |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for name, v in sorted(res["scores"].items(), key=lambda kv: -(kv[1]["auroc"] if kv[1]["auroc"] == kv[1]["auroc"] else -1)):
        L.append(f"| {name} | {v['online_mode'] or '–'} | {v['n']} | {_f(v['auroc'])} | "
                 f"{_f(v['ci95'][0])}–{_f(v['ci95'][1])} | {_f(v['auroc_late'])} | " + " | ".join(_f(v['by_tier'][t]) for t in qsa.TIERS)
                 + f" | {v['verdict']} |")
    b = res["best_online_score"]
    L += ["", "**Best score that can run online:** " + (
        f"`{b['score']}` (AUROC {_f(b['auroc'])}, {b['verdict']}) → `quality_score_mode: {b['quality_score_mode']}` "
        "for the discard and re-weight arms." if b else "none with data.")]
    if res.get("discard"):
        L += ["", "## Discard arms: what was dropped (after the warm-up)", "",
              "precision = dropped takeovers that were inattentive; recall = inattentive takeovers that were "
              "dropped. A random drop has precision = the inattentive share; the oracle has precision and recall 1.",
              "", "| arm | n | inattentive | dropped | precision | recall | attentive lost | inattentive kept |",
              "|---|---|---|---|---|---|---|---|"]
        for arm, d in res["discard"].items():
            L.append(f"| {arm} | {d['n']} | {_f(d['inattentive_share'])} | {_f(d['dropped_share'])} | "
                     f"{_f(d['precision'])} | {_f(d['recall'])} | {_f(d['attentive_dropped'])} | "
                     f"{_f(d['kept_inattentive_share'])} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign", default=None, help="campaign folder (reads <campaign>/aqi_logs)")
    ap.add_argument("--log_root", default=None, help="an aqi_logs folder")
    ap.add_argument("--out", default=None, help="output folder (default <campaign>/report)")
    ap.add_argument("--n_boot", type=int, default=500)
    args = ap.parse_args()
    if not (args.campaign or args.log_root):
        ap.error("give --campaign or --log_root")
    root = args.log_root or os.path.join(args.campaign, "aqi_logs")
    out = args.out or os.path.join(args.campaign or root, "report")
    rows = qsa.load_segments(root)
    res = analyse(rows, n_boot=args.n_boot)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "inconsistent_audit.json"), "w") as f:
        json.dump(res, f, indent=2)
    text = render(res, root)
    with open(os.path.join(out, "inconsistent_audit.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print("wrote", os.path.join(out, "inconsistent_audit.md"))
    return 0 if res["n_segments"] else 1


if __name__ == "__main__":
    sys.exit(main())
