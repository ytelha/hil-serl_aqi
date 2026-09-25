#!/usr/bin/env python3
"""Operator-study pipeline for the AQI-HIL-SERL experimental protocol.

Implements the run-book of the thesis document "Experimental Design -
Measuring Operator Dependency and Comparing AQI-HIL-SERL to Vanilla":

    init      Part E.1  freeze the protocol: campaign manifest, the
                        fixed eval grid (+SHA-256), the counterbalancing
                        table, operator roster, decision rules & delta -
                        the date-stamped pre-registration.
    session   Part E.2  one operator session: verbatim briefing, the
                        exact training + ruler commands for this cell,
                        the 3-question exit form, artifact verification,
                        and the committed session sheet.
    evaluate  Parts B-D, F  ingest every cell (ruler trials + training
                        logs), compute the full metric catalogue,
                        Experiment-1 dependency statistics, Experiment-2
                        comparison battery, the pre-registered decision
                        tables, and the thesis figures.
    simulate            fill a campaign with synthetic cells for an
                        end-to-end dry run; --preset doc_example
                        reproduces the document's worked numbers.

A cell is (method, operator, seed); its id is <method>_<operator>_s<seed>.
Campaign layout:
    <campaign>/campaign.json, eval_grid.json, counterbalancing.json,
    PROTOCOL_FREEZE.md, operators.csv, sessions/<cell>.json,
    aqi_logs/<cell>/..., eval_runs/<cell>/..., checkpoints/<cell>/,
    report/ (from evaluate).
"""

import argparse
import datetime
import glob
import hashlib
import json
import math
import os
import shutil
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "serl_launcher"))
from serl_launcher.aqi import stats as S  # noqa: E402

VANILLA_NOTE = (
    "'logged' is the recommended vanilla arm: HIL-SERL training with the "
    "gate suppressed and quality off, plus the instrumentation the process "
    "metrics need. A pure 'hilserl' arm is supported but writes no "
    "training logs, so its process metrics (S80, r(t), AUC, intervention "
    "counts) are unavailable; outcome metrics from the ruler still work.")


# ===========================================================================
# helpers
# ===========================================================================
def _jsonify(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and math.isinf(o):
        return "inf"
    raise TypeError(f"not JSON serializable: {type(o)}")


def _dump(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_jsonify)


def _load(path):
    with open(path) as f:
        return json.load(f)


def _read_jsonl(path):
    out = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    return out


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _grid_sha(grid: dict) -> str:
    payload = {k: grid[k] for k in sorted(grid) if k != "sha256"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()


def cell_id(method, operator, seed):
    return f"{method}_{operator}_s{seed}"


# ===========================================================================
# init (E.1)
# ===========================================================================
def make_eval_grid(dx_mm, dyaw_rad, reps):
    """The ruler's initial-state grid (A.2): every (dx, yaw) pair x reps,
    in a FIXED order (pose-major, repetition-minor), generated once."""
    trials = [dict(dx_mm=float(dx), dyaw_rad=float(dy), rep=r)
              for r in range(reps)
              for dx in dx_mm for dy in dyaw_rad]
    grid = dict(dx_mm=[float(x) for x in dx_mm],
                dyaw_rad=[float(x) for x in dyaw_rad],
                reps=reps, n_trials=len(trials), trials=trials)
    grid["sha256"] = _grid_sha(grid)
    return grid


def make_counterbalancing(operators, methods, order_seed):
    """Balanced session order per operator (E.1): rows of a cyclic Latin
    square over the methods, assigned to operators in a seeded shuffle -
    with 2 methods this is the AB/BA split of the document's example."""
    rows = [list(methods[i:]) + list(methods[:i]) for i in range(len(methods))]
    rng = np.random.default_rng(order_seed)
    assignment = {}
    perm = rng.permutation(len(operators))
    for rank, op_idx in enumerate(perm):
        assignment[operators[op_idx]] = rows[rank % len(rows)]
    return assignment


def cmd_init(args):
    camp = args.campaign
    os.makedirs(camp, exist_ok=False)
    for d in ("sessions", "aqi_logs", "eval_runs", "checkpoints"):
        os.makedirs(os.path.join(camp, d))

    operators = args.operators.split(",")
    methods = args.methods.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    dx = [float(x) for x in args.dx_mm.split(",")]
    dyaw = [float(x) for x in args.dyaw_rad.split(",")]

    grid = make_eval_grid(dx, dyaw, args.reps)
    _dump(os.path.join(camp, "eval_grid.json"), grid)

    order = make_counterbalancing(operators, methods, args.order_seed)
    _dump(os.path.join(camp, "counterbalancing.json"), order)

    cells = []
    for op in operators:
        idx = 0
        for seed in seeds:
            for method in order[op]:
                cells.append(dict(cell_id=cell_id(method, op, seed),
                                  operator=op, method=method, seed=seed,
                                  order_index=idx))
                idx += 1

    protocol_sha = (_sha256_file(args.protocol_file)
                    if args.protocol_file else None)
    manifest = dict(
        created=datetime.datetime.now().isoformat(timespec="seconds"),
        exp_name=args.exp_name, methods=methods,
        vanilla_method=args.vanilla or methods[0],
        operators=operators, seeds=seeds,
        delta_noninferiority=args.delta,
        budget=args.budget, rig_epoch=args.rig_epoch,
        control_hz=args.control_hz,
        eval_grid_sha256=grid["sha256"],
        protocol_file=args.protocol_file, protocol_sha256=protocol_sha,
        order_seed=args.order_seed, cells=cells,
    )
    _dump(os.path.join(camp, "campaign.json"), manifest)

    with open(os.path.join(camp, "operators.csv"), "w") as f:
        f.write("operator_id,age_range,teleop_experience(none/some/expert),"
                "consent_recorded\n")
        f.writelines(f"{op},,,\n" for op in operators)

    with open(os.path.join(camp, "PROTOCOL_FREEZE.md"), "w") as f:
        f.write(f"""# Pre-registration - campaign `{os.path.basename(camp)}`

Frozen {manifest['created']}. Nothing below changes after this date.

* Task: **{args.exp_name}**; rig epoch: {args.rig_epoch}
* Training budget per cell (one rule, never both): **{args.budget}**
* Methods: {', '.join(methods)} (vanilla arm: **{manifest['vanilla_method']}**)
* Operators: {', '.join(operators)}; seeds: {seeds}
* Eval ruler: {len(dx)}x{len(dyaw)} pose grid x {args.reps} reps = \
{grid['n_trials']} autonomous trials, fixed order, gate disabled \
(sha256 `{grid['sha256'][:16]}...`)
* Counterbalancing: see counterbalancing.json (order seed {args.order_seed})
* Protocol document: {args.protocol_file or '(not attached)'} \
{f"(sha256 `{protocol_sha[:16]}...`)" if protocol_sha else ''}

## Pre-registered decision rules
* **Experiment 1 (sim):** ICC >= 0.3 AND permutation p < 0.05.
* **Experiment 1 (real):** operator range >= 15 pp with non-overlapping
  extreme Wilson CIs.
* **Experiment 2 PRIMARY:** RG_rel > 0 with the 95% paired-operator
  bootstrap CI excluding 0.
* **Experiment 2 GUARDRAIL:** mean-SR non-inferiority at
  delta = {args.delta} - a method that fails the guardrail fails,
  whatever RG says.
* **Secondary (reported with CIs, no accept/reject):** worst-operator
  gain > 0; interventions-to-80% reduced; force-violation rate not
  increased; ablation ordering full >= single component >= vanilla on
  RG_rel.

Exactly ONE primary endpoint: RG_rel on SR. {VANILLA_NOTE}
""")
    print(f"[init] campaign frozen at {camp}")
    print(f"[init] {len(cells)} cells; next: "
          f"python examples/experiment_pipeline.py session "
          f"--campaign {camp} --operator {operators[0]}")


# ===========================================================================
# session (E.2)
# ===========================================================================
BRIEFING = """
------------------------------------------------------------------------
BRIEFING (read verbatim - 5 min)
------------------------------------------------------------------------
"You will control training sessions of a robot learning a precise
insertion task. Whenever you believe the robot needs help, take over
with the SpaceMouse, correct it, and release control when you are done.
{prompt_line}
There are no right or wrong interventions - use your own judgement.
We will not coach you during the session. You may stop at any time.
The session is recorded and anonymised."

Then: 10 minutes familiarisation on PRACTICE poses (disjoint from the
evaluation grid). Familiarisation is NOT logged.
------------------------------------------------------------------------"""

PROMPT_LINE = {
    True: 'The robot may also PROMPT you for help; treat a prompt as a\n'
          'suggestion - you decide whether and how to intervene.',
    False: 'The robot will not prompt you; intervene at your own '
           'initiative.',
}


def _pending_sessions(camp, manifest, operator):
    done = {os.path.splitext(os.path.basename(p))[0]
            for p in glob.glob(os.path.join(camp, "sessions", "*.json"))}
    return [c for c in manifest["cells"]
            if c["operator"] == operator and c["cell_id"] not in done]


def _ask(prompt, default=None, cast=str, valid=None, non_interactive=False):
    if non_interactive:
        return default
    while True:
        raw = input(f"{prompt}{f' [{default}]' if default is not None else ''}: ").strip()
        if not raw and default is not None:
            return default
        try:
            v = cast(raw)
        except ValueError:
            print("  invalid, try again")
            continue
        if valid and not valid(v):
            print("  out of range, try again")
            continue
        return v


def cmd_session(args):
    camp = args.campaign
    manifest = _load(os.path.join(camp, "campaign.json"))
    pending = _pending_sessions(camp, manifest, args.operator)
    if args.cell:
        cellrec = next(c for c in manifest["cells"]
                       if c["cell_id"] == args.cell)
    elif pending:
        cellrec = pending[0]
    else:
        print(f"[session] no pending sessions for {args.operator}")
        return
    cid, method, seed = (cellrec["cell_id"], cellrec["method"],
                         cellrec["seed"])
    active = method in ("active", "aqi")
    exp = manifest["exp_name"]
    task_dir = os.path.join(_HERE, "experiments", exp)
    ckpt = os.path.abspath(os.path.join(camp, "checkpoints", cid))
    demo = f" --demo_path={args.demo_path}" if args.demo_path else ""

    print(f"\n=== SESSION: cell {cid} (operator {args.operator}, "
          f"session {cellrec['order_index'] + 1} of their sequence) ===")
    print("The operator is NOT told which method this session uses.")
    print(BRIEFING.format(prompt_line=PROMPT_LINE[active]))
    print("TRAINING BLOCK - run these (two terminals), budget "
          f"{manifest['budget']}; take one 5-min break at the midpoint; "
          "we sit behind, silent; only safety interaction is permitted:\n")
    print(f"  cd {task_dir}")
    print(f"  bash run_learner_aqi.sh --aqi_mode={method} --seed={seed} "
          f"--checkpoint_path={ckpt}{demo}")
    print(f"  bash run_actor_aqi.sh --aqi_mode={method} --seed={seed} "
          f"--operator_id={args.operator} --run_name={cid} "
          f"--checkpoint_path={ckpt}{demo}\n")
    _ask("Press Enter when the training block has ENDED and the "
         "checkpoint is frozen", default="", non_interactive=args.non_interactive)
    budget_used = _ask("Budget actually used (e.g. '45 min' or '25000 steps')",
                       default=manifest["budget"],
                       non_interactive=args.non_interactive)
    anomalies = _ask("Anomalies during training (blank = none)", default="",
                     non_interactive=args.non_interactive)

    eval_out = os.path.join(camp, "eval_runs", cid)
    print("\nFREEZE + EVALUATE - run the ruler now (the operator fills the "
          "exit form meanwhile):\n")
    print(f"  cd {_HERE}")
    print(f"  python eval_ruler.py --exp_name={exp} "
          f"--checkpoint_path={ckpt} "
          f"--grid_file={os.path.abspath(os.path.join(camp, 'eval_grid.json'))} "
          f"--out={os.path.abspath(eval_out)} "
          f"--control_hz={manifest['control_hz']}\n")

    effort = _ask("Exit form - perceived effort (1-7)", default=4, cast=int,
                  valid=lambda v: 1 <= v <= 7,
                  non_interactive=args.non_interactive)
    competence = _ask("Exit form - perceived robot competence (1-7)",
                      default=4, cast=int, valid=lambda v: 1 <= v <= 7,
                      non_interactive=args.non_interactive)
    comments = _ask("Exit form - free comments", default="",
                    non_interactive=args.non_interactive)
    _ask("Press Enter when the ruler has finished", default="",
         non_interactive=args.non_interactive)

    # ---- artifact verification + log ingestion ----------------------------
    log_dst = os.path.join(camp, "aqi_logs", cid)
    if not os.path.exists(os.path.join(log_dst, "episodes.jsonl")):
        for cand in (os.path.join(task_dir, "aqi_logs", cid),
                     os.path.join(os.getcwd(), "aqi_logs", cid)):
            if os.path.exists(os.path.join(cand, "episodes.jsonl")):
                shutil.copytree(cand, log_dst, dirs_exist_ok=True)
                print(f"[session] ingested training logs from {cand}")
                break
    checks = dict(
        eval_cell_json=os.path.exists(os.path.join(eval_out, "cell.json")),
        eval_trials=len(_read_jsonl(os.path.join(eval_out, "trials.jsonl"))),
        training_episodes=len(_read_jsonl(
            os.path.join(log_dst, "episodes.jsonl"))),
    )
    if method == "hilserl":
        checks["note"] = "hilserl arm: no training logs by design"
    meta_path = os.path.join(log_dst, "meta.json")
    hp_hash = (hashlib.sha256(json.dumps(
        _load(meta_path).get("config", {}), sort_keys=True).encode()
    ).hexdigest() if os.path.exists(meta_path) else None)

    sheet = dict(
        cell_id=cid, operator=args.operator, method=method, seed=seed,
        order_index=cellrec["order_index"],
        date=datetime.datetime.now().isoformat(timespec="seconds"),
        rig_epoch=manifest["rig_epoch"], budget_planned=manifest["budget"],
        budget_used=budget_used, anomalies=anomalies,
        exit_form=dict(effort=effort, competence=competence,
                       comments=comments),
        hyperparameter_hash=hp_hash,
        eval_grid_sha256=manifest["eval_grid_sha256"],
        artifact_checks=checks,
    )
    _dump(os.path.join(camp, "sessions", f"{cid}.json"), sheet)
    ok = checks["eval_cell_json"] and (checks["training_episodes"] > 0
                                       or method == "hilserl")
    status = ("artifacts OK" if ok
              else "WARNING: artifacts incomplete - see artifact_checks")
    print(f"[session] sheet committed: sessions/{cid}.json ({status})")
    rest = _pending_sessions(camp, manifest, args.operator)
    if rest:
        print(f"[session] next for {args.operator}: {rest[0]['cell_id']}")


# ===========================================================================
# evaluate (Parts B, C, D, F)
# ===========================================================================
def _cell_metrics(camp, cellrec, manifest, window, bin_steps):
    cid = cellrec["cell_id"]
    dt = 1.0 / manifest["control_hz"]
    trials = _read_jsonl(os.path.join(camp, "eval_runs", cid, "trials.jsonl"))
    cell_json = os.path.join(camp, "eval_runs", cid, "cell.json")
    f_max = _load(cell_json).get("f_max") if os.path.exists(cell_json) else None
    out = dict(cellrec)
    out["outcomes"] = S.cell_outcomes(trials, f_max=f_max) if trials else None

    eps = _read_jsonl(os.path.join(camp, "aqi_logs", cid, "episodes.jsonl"))
    if eps:
        succ = [float(e.get("success", 0)) for e in eps]
        ln = [int(e.get("length", 0)) for e in eps]
        iv = [int(e.get("n_intervention_steps", 0)) for e in eps]
        segs = [int(e.get("n_segments", 0)) for e in eps]
        out["process"] = dict(
            n_episodes=len(eps),
            s80=S.s80(succ, ln, window=window),
            auc=S.learning_curve_auc(succ),
            intervention=S.intervention_summary(iv, ln, segs, dt=dt),
            r_curve=S.intervention_rate_curve(iv, ln, bin_steps=bin_steps),
            interventions_to_80=S.steps_to_threshold(succ, segs,
                                                     window=window),
            human_seconds_to_80=S.steps_to_threshold(
                succ, [x * dt for x in iv], window=window),
        )
    else:
        out["process"] = None
    return out


def _group(cells, key):
    g = {}
    for c in cells:
        g.setdefault(c[key], []).append(c)
    return g


def _sr(c):
    return (c["outcomes"]["sr"] if c["outcomes"] else float("nan"))


def _pooled_kn(cells):
    k = sum(c["outcomes"]["k_success"] for c in cells if c["outcomes"])
    n = sum(c["outcomes"]["n_trials"] for c in cells if c["outcomes"])
    return k, n


def _exp1_for_method(cells_m, B, rng):
    by_op = _group(cells_m, "operator")
    groups = {op: [_sr(c) for c in cc if c["outcomes"]]
              for op, cc in by_op.items()}
    groups = {op: v for op, v in groups.items() if v}
    op_means = {op: float(np.mean(v)) for op, v in groups.items()}
    spread = S.operator_spread(op_means)
    icc = S.icc_oneway(groups)
    seeds_per_op = max((len(v) for v in groups.values()), default=0)
    res = dict(operator_means=op_means, spread=spread, icc=icc,
               wilson_per_operator={
                   op: dict(zip(("k", "n"), _pooled_kn(by_op[op]))) | dict(
                       ci=S.wilson_interval(*_pooled_kn(by_op[op])))
                   for op in by_op},
               seeds_per_operator=seeds_per_op)
    if seeds_per_op >= 2:
        vals, ops = [], []
        for op, v in groups.items():
            vals += v
            ops += [op] * len(v)
        res["permutation"] = S.permutation_sigma2(vals, ops, B=B, rng=rng)
        res["icc_bootstrap"] = S.bootstrap_icc(groups, B=B, rng=rng)
    else:
        res["permutation"] = dict(
            note="single seed per operator: descriptive (real-robot) path")
    # real-robot criterion: extreme operators' pooled Wilson CIs
    if len(op_means) >= 2:
        wo, bo = spread["worst_operator"], spread["best_operator"]
        ci_w = res["wilson_per_operator"][wo]["ci"]
        ci_b = res["wilson_per_operator"][bo]["ci"]
        res["extremes_nonoverlap"] = not S.intervals_overlap(ci_w, ci_b)
    res["decision"] = S.exp1_decision(
        icc["icc"], res.get("permutation", {}).get("p_value", float("nan")),
        range_pp=spread["range"],
        extremes_nonoverlap=res.get("extremes_nonoverlap"))
    return res


def _mean_finite(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else float("nan")


def _exp2(cells, vanilla, treat, delta, B, rng):
    def op_means(m):
        by_op = _group([c for c in cells if c["method"] == m], "operator")
        return {op: float(np.mean([_sr(c) for c in cc if c["outcomes"]]))
                for op, cc in by_op.items()
                if any(c["outcomes"] for c in cc)}
    yv, ya = op_means(vanilla), op_means(treat)
    res = dict(vanilla=vanilla, treatment=treat,
               operator_means={vanilla: yv, treat: ya},
               rg=S.robustness_gain(yv, ya),
               rg_bootstrap=S.paired_bootstrap_rg_rel(yv, ya, B=B, rng=rng),
               guardrail=S.mean_diff_guardrail(yv, ya, delta=delta, B=B,
                                               rng=rng),
               worst_gain=S.worst_operator_gain(yv, ya),
               wilcoxon=S.wilcoxon_paired(yv, ya),
               brown_forsythe=S.brown_forsythe(
                   {vanilla: list(yv.values()), treat: list(ya.values())}))

    def proc(m, key):
        return _mean_finite([(c["process"] or {}).get(key)
                             for c in cells if c["method"] == m])
    i_v = proc(vanilla, "interventions_to_80")
    i_a = proc(treat, "interventions_to_80")
    res["intervention_efficiency"] = dict(
        vanilla_interventions_to_80=i_v, treat_interventions_to_80=i_a,
        vanilla_human_seconds_to_80=proc(vanilla, "human_seconds_to_80"),
        treat_human_seconds_to_80=proc(treat, "human_seconds_to_80"),
        reduced=bool(i_a < i_v) if np.isfinite(i_a) and np.isfinite(i_v)
        else None)

    def vr(m):
        return _mean_finite([c["outcomes"]["violation_rate"]
                             for c in cells
                             if c["method"] == m and c["outcomes"]])
    vr_v, vr_a = vr(vanilla), vr(treat)
    res["violation_rate"] = dict(vanilla=vr_v, treat=vr_a,
                                 not_increased=bool(vr_a <= vr_v + 1e-9)
                                 if np.isfinite(vr_a) and np.isfinite(vr_v)
                                 else None)

    glmm_cells = [dict(method=c["method"], operator=c["operator"],
                       k=c["outcomes"]["k_success"],
                       n=c["outcomes"]["n_trials"])
                  for c in cells if c["outcomes"]
                  and c["method"] in (vanilla, treat)]
    res["glmm"] = S.glmm_success(glmm_cells, vanilla_method=vanilla)
    return res


def _figures(camp, cells, manifest, exp2_all, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[evaluate] matplotlib missing - figures skipped")
        return

    def save(fig, name):
        for ext in ("png", "svg"):
            fig.savefig(os.path.join(out_dir, f"{name}.{ext}"),
                        bbox_inches="tight", dpi=150)
        plt.close(fig)

    methods = manifest["methods"]
    vanilla = manifest["vanilla_method"]
    by_mo = {}
    for c in cells:
        if c["outcomes"]:
            by_mo.setdefault((c["method"], c["operator"]), []).append(c)

    # ---- money plot (F): paired dots + arrows + Wilson whiskers ---------
    treats = [m for m in methods if m != vanilla]
    if treats:
        treat = treats[-1]  # full system panel
        fig, ax = plt.subplots(figsize=(5, 4.2))
        ops = sorted({c["operator"] for c in cells})
        for op in ops:
            pts = {}
            for x, m in ((0, vanilla), (1, treat)):
                cc = by_mo.get((m, op))
                if not cc:
                    continue
                k, n = _pooled_kn(cc)
                sr = k / n if n else float("nan")
                lo, hi = S.wilson_interval(k, n)
                ax.errorbar([x], [sr], yerr=[[sr - lo], [hi - sr]],
                            fmt="o", capsize=4, color="C0" if x else "C3")
                pts[x] = sr
            if 0 in pts and 1 in pts:
                ax.annotate("", xy=(1, pts[1]), xytext=(0, pts[0]),
                            arrowprops=dict(arrowstyle="->", alpha=0.5))
                ax.text(-0.06, pts[0], op, ha="right", va="center",
                        fontsize=8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([vanilla, treat])
        ax.set_ylabel("success rate (ruler, Wilson 95%)")
        ax.set_ylim(0, 1.02)
        ax.set_title("Per-operator SR: convergence IS the claim")
        save(fig, "money_plot")

    # ---- interaction plot: one line per operator across methods ---------
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ops = sorted({c["operator"] for c in cells})
    for op in ops:
        ys = []
        for m in methods:
            cc = by_mo.get((m, op))
            ys.append(float(np.mean([_sr(c) for c in cc])) if cc
                      else float("nan"))
        ax.plot(range(len(methods)), ys, "o-", label=op)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods)
    ax.set_ylabel("mean SR per operator")
    ax.set_title("Interaction plot: converging lines = operator-robustness")
    ax.legend(fontsize=8)
    save(fig, "interaction_plot")

    # ---- learning + intervention-rate curves, one panel per method ------
    with_logs = [m for m in methods
                 if any(c["method"] == m and c["process"] for c in cells)]
    if with_logs:
        fig, axes = plt.subplots(2, len(with_logs),
                                 figsize=(4.2 * len(with_logs), 6),
                                 squeeze=False, sharey="row")
        for j, m in enumerate(with_logs):
            for c in cells:
                if c["method"] != m or not c["process"]:
                    continue
                eps = _read_jsonl(os.path.join(
                    camp, "aqi_logs", c["cell_id"], "episodes.jsonl"))
                succ = [float(e.get("success", 0)) for e in eps]
                if len(succ) < 3:
                    continue
                w = min(20, max(3, len(succ) // 5))
                roll = np.convolve(succ, np.ones(w) / w, mode="valid")
                axes[0][j].plot(np.arange(len(roll)) + w, roll, alpha=0.8,
                                label=c["operator"])
                rc = c["process"]["r_curve"]
                axes[1][j].plot(rc["bin_ends"], rc["r"], alpha=0.8)
            axes[0][j].set_title(m)
            axes[0][j].set_ylim(-0.02, 1.02)
            axes[1][j].set_ylim(-0.02, 1.02)
            axes[1][j].set_xlabel("env steps")
        axes[0][0].set_ylabel("rolling success")
        axes[1][0].set_ylabel("intervention rate r(t)")
        axes[0][0].legend(fontsize=7)
        save(fig, "learning_curves")


def _decision_table_md(exp1, exp2_all, interaction, manifest):
    lines = ["## Decision-rule table (pre-registered vs observed)", "",
             "| Rule | Criterion | Observed | Met |",
             "|---|---|---|---|"]
    vanilla = manifest["vanilla_method"]
    e1 = exp1.get(vanilla, {})
    d1 = e1.get("decision", {})
    icc_v = e1.get("icc", {}).get("icc", float("nan"))
    p_v = e1.get("permutation", {}).get("p_value")
    p_str = f"{p_v:.4f}" if p_v is not None else "n/a"
    lines.append(f"| Exp1 (sim) | ICC>=0.3 and perm p<0.05 | "
                 f"ICC={icc_v:.3f}, p={p_str} | "
                 f"{'YES' if d1.get('sim_supported') else 'no'} |")
    if "real_supported" in d1:
        lines.append(
            f"| Exp1 (real) | range>=15pp, extreme CIs disjoint | "
            f"range={d1['range_pp']:.2f}, "
            f"disjoint={d1['extremes_nonoverlap']} | "
            f"{'YES' if d1.get('real_supported') else 'no'} |")
    for treat, e2 in exp2_all.items():
        ci = e2["rg_bootstrap"]["ci95"]
        d2 = e2["decision"]
        lines.append(f"| Exp2 PRIMARY ({treat}) | RG_rel CI95 excludes 0 | "
                     f"RG_rel={e2['rg']['rg_rel']:.3f} "
                     f"CI=[{ci[0]:.3f},{ci[1]:.3f}] | "
                     f"{'YES' if d2['primary_rg_rel_ci_excludes_0'] else 'no'} |")
        g = e2["guardrail"]
        lines.append(f"| Exp2 GUARDRAIL ({treat}) | CI lower of dbar > "
                     f"-{g['delta']} | dbar={g['dbar']:.3f} "
                     f"CI=[{g['ci95_bootstrap'][0]:.3f},"
                     f"{g['ci95_bootstrap'][1]:.3f}] | "
                     f"{'YES' if d2['guardrail_noninferiority'] else 'no'} |")
        sec = d2["secondary"]
        lines.append(f"| Exp2 secondary ({treat}) | worst-op gain>0; "
                     f"intv-to-80 reduced; VR not increased | "
                     f"gain={e2['worst_gain']['gain']:.3f}; "
                     f"reduced={sec['interventions_to_80_reduced']}; "
                     f"VR ok={sec['violation_rate_not_increased']} | "
                     f"reported |")
    if interaction and len(manifest["methods"]) > 2:
        lines.append("| Ablation ordering | full >= single >= vanilla "
                     "(RG_rel) | see Exp2 blocks per method | reported |")
    return "\n".join(lines)


def cmd_evaluate(args):
    camp = args.campaign
    manifest = _load(os.path.join(camp, "campaign.json"))
    rng = np.random.default_rng(args.stat_seed)
    out_dir = args.out or os.path.join(camp, "report")
    os.makedirs(out_dir, exist_ok=True)

    cells = [_cell_metrics(camp, c, manifest, args.window, args.bin_steps)
             for c in manifest["cells"]]
    have = [c for c in cells if c["outcomes"]]
    print(f"[evaluate] {len(have)}/{len(cells)} cells with ruler data")

    vanilla = args.vanilla or manifest["vanilla_method"]
    exp1 = {m: _exp1_for_method(
        [c for c in have if c["method"] == m], args.B, rng)
        for m in manifest["methods"]
        if any(c["method"] == m for c in have)}

    exp2_all = {}
    for treat in manifest["methods"]:
        if treat == vanilla or not any(c["method"] == treat for c in have):
            continue
        e2 = _exp2(have, vanilla, treat, manifest["delta_noninferiority"],
                   args.B, rng)
        exp2_all[treat] = e2

    # ablation ordering on RG_rel (D.3 secondary), then decisions
    full = "aqi" if "aqi" in exp2_all else None
    singles = [m for m in exp2_all if m not in (full,)]
    ablation_ok = None
    if full and singles:
        rg_full = exp2_all[full]["rg"]["rg_rel"]
        ablation_ok = all(np.isfinite(exp2_all[s]["rg"]["rg_rel"])
                          and rg_full >= exp2_all[s]["rg"]["rg_rel"] >= 0
                          for s in singles)
    for treat, e2 in exp2_all.items():
        e2["decision"] = S.exp2_decision(
            e2["rg_bootstrap"]["ci95"], e2["guardrail"]["passed"],
            e2["worst_gain"]["gain"],
            e2["intervention_efficiency"]["reduced"],
            e2["violation_rate"]["not_increased"],
            ablation_ok if treat == full else None)

    op_mean_table = {m: exp1[m]["operator_means"] for m in exp1
                     if len(exp1[m]["operator_means"]) >= 2}
    interaction = (S.interaction_decomposition(op_mean_table)
                   if len(op_mean_table) >= 2 else None)

    results = dict(campaign=manifest, generated=datetime.datetime.now()
                   .isoformat(timespec="seconds"),
                   cells=cells, exp1=exp1, exp2=exp2_all,
                   interaction=interaction,
                   ablation_ordering_ok=ablation_ok)
    _dump(os.path.join(out_dir, "results.json"), results)

    # ---- report.md --------------------------------------------------------
    L = [f"# Experiment report - campaign `{os.path.basename(camp)}`", "",
         f"Generated {results['generated']}; vanilla arm: **{vanilla}**; "
         f"delta = {manifest['delta_noninferiority']}; B = {args.B}.",
         "", "## Per-cell metric catalogue", "",
         "| cell | SR (Wilson95) | T_succ [s] | F_peak [N] | VR | S80 "
         "| AUC | interventions (steps) |", "|---|---|---|---|---|---|---|---|"]
    for c in cells:
        o, p = c["outcomes"], c["process"]
        if o:
            sr = (f"{o['sr']:.2f} [{o['sr_wilson95'][0]:.2f},"
                  f"{o['sr_wilson95'][1]:.2f}]")
            t = ("-" if math.isnan(o["time_to_success_mean"])
                 else f"{o['time_to_success_mean']:.1f}")
            fp, vr = f"{o['f_peak_mean']:.1f}", f"{o['violation_rate']:.2f}"
        else:
            sr = t = fp = vr = "-"
        s80 = ("-" if not p else ("inf" if math.isinf(p["s80"])
                                  else f"{p['s80']:.0f}"))
        auc = "-" if not p else f"{p['auc']:.2f}"
        iv = ("-" if not p else
              f"{p['intervention']['events_takeovers']} "
              f"({p['intervention']['intervention_steps']})")
        L.append(f"| {c['cell_id']} | {sr} | {t} | {fp} | {vr} | {s80} "
                 f"| {auc} | {iv} |")
    L += ["", "## Experiment 1 - operator dependency (per method)", ""]
    for m, e1 in exp1.items():
        sp, icc = e1["spread"], e1["icc"]
        L += [f"### method = {m}",
              f"- operator means: " + ", ".join(
                  f"{o}={v:.2f}" for o, v in sorted(
                      e1["operator_means"].items())),
              f"- sigma_op={sp['sigma_op']:.3f}  CV={sp['cv']:.3f}  "
              f"range={sp['range']:.2f}  worst={sp['worst']:.2f} "
              f"({sp['worst_operator']})",
              f"- ICC(1,1)={icc['icc']:.3f} "
              f"(sigma2_op={icc['sigma2_op']:.5f}, "
              f"sigma2_seed={icc['sigma2_seed']:.5f})"]
        if "p_value" in e1.get("permutation", {}):
            L.append(f"- permutation p={e1['permutation']['p_value']:.4f}; "
                     f"ICC CI95={np.round(e1['icc_bootstrap']['icc_ci95'], 3).tolist()}")
        else:
            L.append(f"- {e1['permutation']['note']}")
        L.append("")
    L += ["## Experiment 2 - method comparison (paired on operator)", ""]
    for treat, e2 in exp2_all.items():
        rgb = e2["rg_bootstrap"]
        L += [f"### {vanilla} -> {treat}",
              f"- **RG_rel = {e2['rg']['rg_rel']:.3f}**  "
              f"CI95=[{rgb['ci95'][0]:.3f},{rgb['ci95'][1]:.3f}]  "
              f"(RG_abs={e2['rg']['rg']:.5f})",
              f"- guardrail dbar={e2['guardrail']['dbar']:+.3f} "
              f"(pass={e2['guardrail']['passed']}); "
              f"worst-operator gain={e2['worst_gain']['gain']:+.3f}",
              f"- Wilcoxon p={e2['wilcoxon'].get('p_value', float('nan')):.3f} "
              f"rank-biserial={e2['wilcoxon'].get('rank_biserial', float('nan')):.2f}; "
              f"Brown-Forsythe p={e2['brown_forsythe'].get('p_value', float('nan')):.3f}",
              f"- interventions-to-80: {e2['intervention_efficiency']}",
              f"- GLMM: " + (e2["glmm"].get("note") or
                             f"method coef={e2['glmm']['method_coef_logodds']:.3f} "
                             f"CI={np.round(e2['glmm']['method_coef_ci95'], 3).tolist()}, "
                             f"random SDs={ {k: round(v, 3) for k, v in e2['glmm']['random_sd'].items()} }"),
              ""]
    if interaction:
        L += ["## Interaction model (exact decomposition)",
              f"- mu={interaction['mu']:.4f}; alpha=" + ", ".join(
                  f"{m}={a:+.4f}" for m, a in interaction["alpha"].items()),
              f"- Var(operator effects) per method: " + ", ".join(
                  f"{m}={v:.5f}" for m, v in
                  interaction["var_operator_effects"].items()),
              f"- reconstruction max err = "
              f"{interaction['reconstruction_max_err']:.2e}", ""]
    L += [_decision_table_md(exp1, exp2_all, interaction, manifest), "",
          "Figures: money_plot, interaction_plot, learning_curves "
          "(.png/.svg alongside this report)."]
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write("\n".join(L))

    _figures(camp, have, manifest, exp2_all, out_dir)
    print(f"[evaluate] wrote {out_dir}/report.md, results.json, figures")


# ===========================================================================
# simulate - synthetic campaigns for dry runs
# ===========================================================================
DOC_EXAMPLE = dict(
    vanilla={"O1": [0.88, 0.90, 0.85, 0.92, 0.85],
             "O2": [0.74, 0.70, 0.78, 0.72, 0.76],
             "O3": [0.60, 0.55, 0.65, 0.58, 0.62],
             "O4": [0.52, 0.48, 0.58, 0.50, 0.62]},
    aqi={"O1": 0.89, "O2": 0.82, "O3": 0.78, "O4": 0.76},
)


def _sim_cell_sr(preset, method, operator, seed_idx, methods, rng):
    if preset == "doc_example":
        base_v = DOC_EXAMPLE["vanilla"].get(
            operator, [0.7] * 5)[seed_idx % 5]
        base_a = DOC_EXAMPLE["aqi"].get(operator, 0.8)
        if method == methods[0]:
            return base_v
        if method == "aqi" or method == methods[-1]:
            return float(np.clip(base_a + 0.02 * rng.standard_normal(), 0, 1))
        frac = (methods.index(method)) / (len(methods) - 1)
        return float(np.clip(base_v + frac * (base_a - base_v)
                             + 0.02 * rng.standard_normal(), 0, 1))
    skill = {"O1": 0.18, "O2": 0.04, "O3": -0.10, "O4": -0.16}.get(
        operator, 0.05 * rng.standard_normal())
    compress = (methods.index(method)) / max(len(methods) - 1, 1)
    mu = 0.70 + 0.10 * compress
    return float(np.clip(mu + skill * (1 - 0.85 * compress)
                         + 0.03 * rng.standard_normal(), 0.02, 0.98))


def cmd_assess(args):
    """Operator skill assessment (methodology doc Part B) over the
    campaign's aqi_logs: cells are grouped by operator, each operator's
    runs feed the Table-7 feature pipeline, and Methods A (composite) +
    D (Beta-Bernoulli) - plus B (clustering) when the cohort is large
    enough - type every operator. Writes report/skill.md + skill.json
    and adds an assessed_tier column to operators.csv."""
    import numpy as np
    from serl_launcher.aqi import skill as SK
    camp = args.campaign
    log_root = os.path.join(camp, "aqi_logs")
    by_op = {}
    for cid in sorted(os.listdir(log_root)) if os.path.isdir(log_root) \
            else []:
        run = os.path.join(log_root, cid)
        if not os.path.isdir(run):
            continue
        r = SK.features_from_aqi_logs(run, budget_s=args.budget_s)
        if not r["episode_rows"]:
            continue
        op = (r["meta"].get("operator_id")
              or cid.split("_")[1] if "_" in cid else cid)
        d = by_op.setdefault(op, dict(rows=[], segs=[]))
        d["rows"].extend(r["episode_rows"])
        d["ivs"] = r["interventions"]      # last run's aggregate basis
    if not by_op:
        print("no aqi_logs found under", log_root)
        return
    vecs = {o: SK.aggregate_operator(d["rows"], args.budget_s,
                                     d.get("ivs"))
            for o, d in by_op.items()}
    feats = [f for f in SK.CORE7
             if any(np.isfinite(v.get(f, np.nan)) for v in vecs.values())]
    Z = SK.normalize_features(vecs, feats)
    S = {o: SK.composite_score(Z[o], feats) for o in Z}
    comp = SK.composite_tiers(S)
    lines = ["# Operator skill (campaign assessment)", "",
             f"features: {', '.join(feats)}; cuts "
             f"{np.round(comp['cuts'], 3).tolist()}", "",
             "| operator | episodes | SR | composite S | tier |",
             "|---|---|---|---|---|"]
    for o in sorted(vecs):
        lines.append(f"| {o} | {vecs[o]['n_episodes']} | "
                     f"{vecs[o]['sr']:.2f} | {S[o]:+.2f} | "
                     f"{comp['tiers'][o]} |")
    if len(vecs) >= 8:
        cl = SK.cluster_operators(Z, feats,
                                  sr={o: vecs[o]["sr"] for o in vecs})
        lines += ["", f"k-means (stability {cl['stability']:.2f}"
                  + (", <15-operator warning"
                     if cl["small_cohort_warning"] else "") + "): "
                  + json.dumps(cl["tiers"])]
    os.makedirs(os.path.join(camp, "report"), exist_ok=True)
    with open(os.path.join(camp, "report", "skill.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(camp, "report", "skill.json"), "w") as f:
        json.dump(dict(features=feats, composite=comp,
                       scores=S, vectors={o: {k: v for k, v in
                                              vecs[o].items()
                                              if np.isscalar(v)}
                                          for o in vecs}),
                  f, indent=1, default=float)
    # annotate operators.csv
    ops_csv = os.path.join(camp, "operators.csv")
    if os.path.exists(ops_csv):
        with open(ops_csv) as f:
            rows = [l.rstrip("\n") for l in f]
        if rows and "assessed_tier" not in rows[0]:
            rows[0] += ",assessed_tier"
            for i in range(1, len(rows)):
                oid = rows[i].split(",")[0]
                rows[i] += "," + comp["tiers"].get(oid, "")
            with open(ops_csv, "w") as f:
                f.write("\n".join(rows) + "\n")
    print("skill assessment -> report/skill.md "
          f"({len(vecs)} operators)")


def cmd_simulate(args):
    camp = args.campaign
    manifest = _load(os.path.join(camp, "campaign.json"))
    grid = _load(os.path.join(camp, "eval_grid.json"))
    n_trials = grid["n_trials"]
    methods = manifest["methods"]
    rng = np.random.default_rng(args.sim_seed)
    dt = 1.0 / manifest["control_hz"]

    for cellrec in manifest["cells"]:
        cid, m, op = cellrec["cell_id"], cellrec["method"], cellrec["operator"]
        seed_idx = manifest["seeds"].index(cellrec["seed"])
        sr = _sim_cell_sr(args.preset, m, op, seed_idx, methods, rng)

        # ---- ruler trials -------------------------------------------------
        ed = os.path.join(camp, "eval_runs", cid)
        os.makedirs(ed, exist_ok=True)
        k_target = int(round(sr * n_trials))
        succ_flags = rng.permutation(
            [True] * k_target + [False] * (n_trials - k_target))
        trials = []
        with open(os.path.join(ed, "trials.jsonl"), "w") as f:
            for i, spec in enumerate(grid["trials"]):
                s = bool(succ_flags[i])
                fp = float(np.clip(8 + 6 * rng.random()
                                   + (10 if rng.random() < 0.03 else 0),
                                   0, 60))
                rec = dict(trial=i, dx_mm=spec["dx_mm"],
                           dyaw_rad=spec["dyaw_rad"], rep=spec["rep"],
                           success=s, steps=int(60 + 40 * rng.random()),
                           time_s=float(4 + 4 * rng.random()) if s else None,
                           f_peak=fp, f_mean=fp * 0.45,
                           tau_peak=float(0.3 + 0.3 * rng.random()),
                           violation=bool(fp > 30.0), intervened=False,
                           pose_control="sim")
                trials.append(rec)
                f.write(json.dumps(rec) + "\n")
        _dump(os.path.join(ed, "cell.json"),
              dict(cell_id=cid, f_max=30.0, grid_sha256=grid["sha256"],
                   simulated=True,
                   outcomes=S.cell_outcomes(trials, f_max=30.0)))

        # ---- training episode logs ---------------------------------------
        if m != "hilserl":
            ld = os.path.join(camp, "aqi_logs", cid)
            os.makedirs(ld, exist_ok=True)
            n_ep, ln = 60, 100
            with open(os.path.join(ld, "episodes.jsonl"), "w") as f:
                for ep in range(n_ep):
                    ramp = min(ep / 35.0, 1.0)
                    p = 0.15 + ramp * (sr - 0.15)
                    s = bool(rng.random() < p)
                    quality_boost = 0.6 if m in ("quality", "aqi") else 1.0
                    iv = int(np.clip((1 - ramp) * 80 * quality_boost
                                     + rng.integers(0, 8), 0, ln))
                    f.write(json.dumps(dict(
                        episode=ep, operator_id=op, mode=m, success=s,
                        episode_return=float(s), length=ln,
                        n_segments=int(iv > 0) + int(iv > 40),
                        n_intervention_steps=iv, n_excluded_segments=0,
                        n_requests=int(iv > 0 and m in ("active", "aqi")),
                        n_request_steps=0, n_hard_violations=0,
                        f_peak=float(8 + 8 * rng.random()), tau_peak=0.4,
                        n_scored_transitions=0)) + "\n")
            _dump(os.path.join(ld, "meta.json"),
                  dict(mode=m, operator_id=op, config=dict(simulated=True)))

        # ---- session sheet -------------------------------------------------
        _dump(os.path.join(camp, "sessions", f"{cid}.json"),
              dict(cell_id=cid, operator=op, method=m,
                   seed=cellrec["seed"], order_index=cellrec["order_index"],
                   date=datetime.datetime.now().isoformat(timespec="seconds"),
                   rig_epoch=manifest["rig_epoch"], simulated=True,
                   budget_used=manifest["budget"], anomalies="",
                   exit_form=dict(effort=int(rng.integers(2, 7)),
                                  competence=int(rng.integers(2, 7)),
                                  comments="(simulated)"),
                   eval_grid_sha256=manifest["eval_grid_sha256"],
                   artifact_checks=dict(simulated=True)))
    print(f"[simulate] filled {len(manifest['cells'])} cells "
          f"(preset={args.preset}); now run evaluate.")


# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="freeze the campaign (E.1)")
    p.add_argument("--campaign", required=True)
    p.add_argument("--exp_name", required=True)
    p.add_argument("--operators", required=True,
                   help="comma list, e.g. O1,O2,O3,O4")
    p.add_argument("--methods", default="logged,aqi",
                   help=f"comma list of aqi_mode arms. {VANILLA_NOTE}")
    p.add_argument("--vanilla", default=None,
                   help="vanilla arm (default: first of --methods)")
    p.add_argument("--seeds", default="0", help="comma list, e.g. 0,1,2,3,4")
    p.add_argument("--dx_mm", default="-20,-10,0,10,20")
    p.add_argument("--dyaw_rad", default="-0.10,-0.05,0,0.05,0.10")
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--delta", type=float, default=0.05,
                   help="non-inferiority margin (pre-registered)")
    p.add_argument("--budget", default="45 min wall-clock")
    p.add_argument("--rig_epoch", default="rig-v1")
    p.add_argument("--control_hz", type=float, default=10.0)
    p.add_argument("--protocol_file", default=None,
                   help="the frozen protocol document to hash")
    p.add_argument("--order_seed", type=int, default=0)
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("session", help="run one operator session (E.2)")
    p.add_argument("--campaign", required=True)
    p.add_argument("--operator", required=True)
    p.add_argument("--cell", default=None,
                   help="explicit cell id (default: operator's next pending)")
    p.add_argument("--demo_path", default=None)
    p.add_argument("--non_interactive", action="store_true",
                   help="skip prompts with defaults (testing)")
    p.set_defaults(func=cmd_session)

    p = sub.add_parser("evaluate", help="metrics + statistics + report (B-F)")
    p.add_argument("--campaign", required=True)
    p.add_argument("--vanilla", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--B", type=int, default=10_000)
    p.add_argument("--window", type=int, default=20,
                   help="rolling window for S80 / thresholds")
    p.add_argument("--bin_steps", type=int, default=1000)
    p.add_argument("--stat_seed", type=int, default=0)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("assess", help="operator skill typing (doc Part B)")
    p.add_argument("--campaign", required=True)
    p.add_argument("--budget_s", type=float, default=15.0)
    p.set_defaults(func=cmd_assess)

    p = sub.add_parser("simulate", help="synthetic campaign dry-run")
    p.add_argument("--campaign", required=True)
    p.add_argument("--preset", default="realistic",
                   choices=["realistic", "doc_example"])
    p.add_argument("--sim_seed", type=int, default=0)
    p.set_defaults(func=cmd_simulate)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
