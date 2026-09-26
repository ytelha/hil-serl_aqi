#!/usr/bin/env python3
"""Check that every cell of a campaign trained against the operator its
demos were recorded with.

Why: until the fix in operator_model.jitter_theta the per-operator jitter
depended on Python's per-process string hashing, so every run_campaign.py
process drew a different theta for the same operator id, and every
non-shard run rewrote operators/<id>.json with it. Within one uninterrupted
run all cells agree; a RESUMED run (or a separate --stages analyze run)
rewrote the roster, and cells trained after a resume used a different
operator than their demos and the earlier cells.

The records that tell what each step really used:
    demos/<op>_s<seed>/demos_groundtruth.json   theta that recorded the demos
    checkpoints/<cell>/manifest.json            theta the cell trained with
                                                (written right after training)
    aqi_logs/<cell>_actor/meta.json             full theta seen by the actor
                                                (AQI modes and HiRIL)
    operators/<op>.json                         the roster file as it is NOW

    python examples/roster_audit.py runs/<campaign> [runs/<campaign2> ...]
    python examples/roster_audit.py --restore runs/<campaign>

Exit code 0 if every cell matches its demos, 1 otherwise. Without
--restore no file is changed. --restore writes the demos' theta back to
every rewritten operators/<op>.json (the old file kept as .json.bak), so a
resumed or extended campaign trains its remaining cells against the
operator its demos came from. Plain python3 (json only).
"""

import glob
import json
import os
import sys

# the jittered parameters (operator_model._SCALE_JITTERED); the manifests
# carry only some of them, so each comparison uses the keys both sides have
KEYS = ("sigma0", "k_sdn", "sigma_tr", "sigma_perc", "sigma_mon", "rho_fa",
        "pause_len_mean")


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _theta_in(d):
    """First dict in a (nested) record that looks like an operator theta."""
    if isinstance(d, dict):
        if "sigma0" in d and "rho_fa" in d:
            return d
        for v in d.values():
            t = _theta_in(v)
            if t is not None:
                return t
    elif isinstance(d, list):
        for v in d:
            t = _theta_in(v)
            if t is not None:
                return t
    return None


def _same(a, b):
    """(equal on the shared jittered keys, number of keys compared)."""
    ks = [k for k in KEYS if k in a and k in b
          and isinstance(a[k], (int, float)) and isinstance(b[k], (int, float))]
    ok = all(abs(float(a[k]) - float(b[k])) <= 1e-9 * max(1.0, abs(float(b[k])))
             for k in ks)
    return ok, len(ks)


def _brief(t):
    return " ".join(f"{k}={float(t[k]):.3f}" for k in ("sigma0", "rho_fa",
                                                        "sigma_perc") if k in t)


def audit(camp, restore=False):
    print(f"\n=== {camp}")
    ops = {os.path.basename(p)[:-5]: _load(p)
           for p in sorted(glob.glob(os.path.join(camp, "operators", "*.json")))}
    demos = {}
    for p in sorted(glob.glob(os.path.join(camp, "demos", "*",
                                           "demos_groundtruth.json"))):
        t = _theta_in((_load(p) or {}).get("theta"))
        if t is not None:
            op = t.get("operator_id") or os.path.basename(
                os.path.dirname(p)).rsplit("_s", 1)[0]
            demos.setdefault(op, []).append((os.path.basename(
                os.path.dirname(p)), t))
    bad_cells, n_cells, file_rewritten = [], 0, []
    for op in sorted(set(ops) | set(demos)):
        ref = demos.get(op, [(None, None)])[0][1]
        # all demo seeds of one operator must share a theta
        for name, t in demos.get(op, [])[1:]:
            if not _same(t, ref)[0]:
                print(f"  {op}: demos {name} used a different theta than "
                      f"{demos[op][0][0]}")
        if ref is None:
            print(f"  {op}: no demos ground truth; reference = roster file")
            ref = ops.get(op)
        elif ops.get(op) is not None and not _same(ops[op], ref)[0]:
            file_rewritten.append(op)
        if ref is None:
            continue
        for mp in sorted(glob.glob(os.path.join(camp, "checkpoints", "*",
                                                "manifest.json"))):
            cell = os.path.basename(os.path.dirname(mp))
            m = _load(mp) or {}
            o = m.get("operator") or {}
            if o.get("operator_id") != op:
                continue
            n_cells += 1
            used = [("manifest", o)]
            meta = _theta_in(_load(os.path.join(camp, "aqi_logs",
                                                f"{cell}_actor", "meta.json")))
            if meta is not None:
                used.append(("actor meta", meta))
            for src, t in used:
                ok, nk = _same(t, ref)
                if nk and not ok:
                    bad_cells.append(cell)
                    print(f"  MISMATCH {cell} ({src}): {_brief(t)}  vs demos "
                          f"{_brief(ref)}")
                    break
    for op in file_rewritten:
        print(f"  {op}: operators/{op}.json was rewritten after the demos "
              f"({_brief(ops[op])} vs demos {_brief(demos[op][0][1])})")
        if restore:
            p = os.path.join(camp, "operators", f"{op}.json")
            os.replace(p, p + ".bak")
            with open(p, "w", encoding="utf-8") as f:
                json.dump(demos[op][0][1], f, indent=2, sort_keys=True)
                f.write("\n")
            print(f"     restored from the demos (old file: {op}.json.bak)")
    bad = sorted(set(bad_cells))
    print(f"  cells checked: {n_cells}   trained against a different operator "
          f"than their demos: {len(bad)}   roster files rewritten: "
          f"{len(file_rewritten)}")
    if bad:
        print("  -> these cells are not comparable with the rest of their "
              "operator (between-operator variance / ICC); retrain them or "
              "drop them from operator-level analyses.")
    elif file_rewritten:
        print("  -> training is consistent; only the roster files (and any "
              "report that reads them) show a later draw. The demos' "
              "demos_groundtruth.json holds the operator that was used.")
    else:
        print("  -> consistent.")
    return not bad


def main(argv):
    restore = "--restore" in argv
    camps = [a for a in argv if a != "--restore"]
    if not camps or camps[0] in ("-h", "--help"):
        print(__doc__)
        return 2
    ok = [audit(os.path.abspath(c), restore) for c in camps]
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
