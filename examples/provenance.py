#!/usr/bin/env python3
"""Artifact provenance: self-describing storage for AQI-HIL-SERL runs.

Every artifact this project produces -- a set of demonstrations, a trained
policy (checkpoint), or a ruler evaluation -- should be a folder that carries,
next to its payload, both a machine-readable ``manifest.json`` and a
human-readable ``README.md`` explaining *what task produced it* and *under what
conditions*.

The interesting part is the **conditions table**. AQI composes its behaviour
from switches (`--aqi_mode`, trigger terms, quality filters, ensemble size, ...)
that are passed in *every* mode but are only *active* in some -- e.g. a baseline
``hilserl`` run still accepts an uncertainty-trigger setting, but that setting
does nothing. ``condition_rows`` reads the SAME ``AQIConfig`` the runtime reads
(``active_on`` / ``quality_on`` / ``logging_on`` / ``trigger_terms``) and labels
each choice **active** or **inert (with the reason)**, so the annotation can
never drift from the code's real behaviour.

This module is deliberately dependency-light (standard library only) and every
public writer is wrapped so a provenance failure can NEVER break the training,
evaluation or data-collection it documents.
"""

from __future__ import annotations

import datetime
import json
import platform
import socket
import subprocess
from pathlib import Path

SCHEMA = "aqi-artifact/1"

MODE_ONE_LINER = {
    "hilserl": "plain HIL-SERL baseline (active gate off, quality off, "
               "training logs off)",
    "logged": "baseline + full trigger/intervention logging (gate off, "
              "quality off)",
    "active": "active query gate A(s_t) ON; replay/loss unchanged",
    "quality": "quality-aware replay + BC loss ON; active gate off "
               "(manual/idle interventions)",
    "aqi": "full system: active gate + quality-aware replay/loss",
    "e2hil": "EXTERNAL baseline E2HiL (Deng et al., RA-L 2026): plain "
             "HIL-SERL + entropy-bounded actor loss (gate off, quality off, "
             "training logs off)",
}


def _is_baseline_arch(mode) -> bool:
    return mode in ("hilserl", "e2hil")


# ===========================================================================
# conditions: derive active/inert from the SAME config the runtime uses
# ===========================================================================
def _fmt_val(v):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "on" if v else "off"
    if isinstance(v, (list, tuple)):
        return ", ".join(str(x) for x in v) if v else "()"
    return str(v)


def _row(choice, value, is_active, active_note, inert_note, status=None):
    return {
        "choice": choice,
        "value": _fmt_val(value),
        "status": status or ("active" if is_active else "inert"),
        "note": active_note if is_active else inert_note,
    }


def condition_rows(cfg) -> list:
    """Turn an ``AQIConfig`` into annotated rows. Duck-typed: any object with
    the AQIConfig attributes works, so this module needs no serl imports."""
    def g(name):
        return getattr(cfg, name, None)

    mode = g("mode") or "?"
    active = bool(getattr(cfg, "active_on", mode in ("active", "aqi")))
    quality = bool(getattr(cfg, "quality_on", mode in ("quality", "aqi")))
    logging_on = bool(getattr(cfg, "logging_on", not _is_baseline_arch(mode)))
    terms = tuple(g("trigger_terms") or ())
    uses_uncertainty = active and ("uncertainty" in terms)

    rows = [{
        "choice": "aqi_mode", "value": mode, "status": "mode",
        "note": MODE_ONE_LINER.get(mode, ""),
    }]

    # --- active query gate -------------------------------------------------
    gate_ok = "feeds the query gate A(s_t) that requests interventions"
    gate_no = (f"INERT - the active gate is OFF in '{mode}', so this is never "
               f"consulted (no interventions are requested from it)")
    for name in ("trigger_combiner", "trigger_terms", "tau_A"):
        rows.append(_row(name, g(name), active, gate_ok, gate_no))

    # --- critic ensemble / epistemic uncertainty --------------------------
    K = g("critic_ensemble_size")
    if _is_baseline_arch(mode):
        rows.append(_row(
            "critic_ensemble_size", K, False, "",
            f"INERT for epistemics - '{mode}' is the plain 2-critic SAC "
            "baseline (bootstrap + per-member targets disabled), so the "
            "ensemble size carries no disagreement signal", status="baseline"))
    else:
        note = f"trains {_fmt_val(K)} critics"
        if uses_uncertainty:
            note += "; their disagreement sigma_Q feeds the uncertainty trigger"
        elif active:
            note += "; disagreement is computed but NOT triggered " \
                    "('uncertainty' absent from trigger_terms)"
        else:
            note += "; shapes learning, but disagreement is not used to " \
                    "trigger (gate off in this mode)"
        rows.append(_row("critic_ensemble_size", K, True, note, ""))

    # --- quality-aware module ---------------------------------------------
    q_ok = ("scores intervention segments (counterfactual advantage) for "
            "weighted replay + BC")
    q_no = f"INERT - the quality module is OFF in '{mode}'"
    for name in ("quality_disagreement_gate", "quality_filters",
                 "bc_loss_weight", "outcome_filter"):
        rows.append(_row(name, g(name), quality, q_ok, q_no))

    # --- unsolicited-takeover routing (needs a gate/quality to matter) ----
    rows.append(_row(
        "unsolicited_policy", g("unsolicited_policy"), active or quality,
        "routes unrequested takeovers (kept as RL data either way)",
        f"INERT - '{mode}' has no gate/quality, every takeover is treated "
        f"the same regardless of this"))

    # --- E2HiL external baseline -------------------------------------------
    e2hil = mode == "e2hil"
    for name in ("e2hil_K", "e2hil_q_low", "e2hil_q_high", "e2hil_q_agg"):
        rows.append(_row(
            name, g(name), e2hil,
            "E2HiL entropy-bounded actor loss (MC samples / |c| percentile "
            "band / critic reduction)",
            f"INERT - the E2HiL actor loss is only used in 'e2hil', not "
            f"'{mode}'"))

    # --- logging -----------------------------------------------------------
    rows.append(_row(
        "training logs", "on" if logging_on else "off", logging_on,
        "per-step / per-segment / per-episode JSONL logs are written",
        f"INERT - '{mode}' writes no training logs by design (its ruler "
        "outcome metrics still work)"))
    return rows


# ===========================================================================
# environment provenance
# ===========================================================================
def git_info(repo=None):
    repo = str(repo or Path(__file__).resolve().parent.parent)
    out = {}
    try:
        out["commit"] = subprocess.check_output(
            ["git", "-C", repo, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
        dirty = subprocess.run(
            ["git", "-C", repo, "diff", "--quiet"],
            stderr=subprocess.DEVNULL).returncode
        out["dirty"] = bool(dirty)
    except Exception:
        out["commit"] = None
        out["dirty"] = None
    return out


def host_info():
    try:
        return {"host": socket.gethostname(),
                "python": platform.python_version(),
                "platform": platform.platform()}
    except Exception:
        return {}


def _now():
    return datetime.datetime.now().isoformat(timespec="seconds")


# ===========================================================================
# manifest builders
# ===========================================================================
def build_manifest(kind, *, task=None, method=None, operator=None, seeds=None,
                   demos=None, termination=None, evaluation=None,
                   conditions=None, reproduce=None, extra=None):
    """Assemble a manifest dict. ``kind`` in {run, demo, eval, campaign}."""
    m = {
        "schema": SCHEMA,
        "kind": kind,
        "created": _now(),
        "provenance": {**git_info(), **host_info()},
    }
    if task is not None:
        m["task"] = task
    if method is not None:
        m["method"] = method
    if operator is not None:
        m["operator"] = operator
    if seeds is not None:
        m["seeds"] = seeds
    if demos is not None:
        m["demos"] = demos
    if termination is not None:
        m["termination"] = termination
    if evaluation is not None:
        m["evaluation"] = evaluation
    if conditions is not None:
        m["conditions"] = conditions
    if reproduce is not None:
        m["reproduce"] = reproduce
    if extra:
        m.update(extra)
    return m


# ===========================================================================
# README renderers
# ===========================================================================
def _task_line(task):
    if not task:
        return ""
    exp = task.get("exp_name", "?")
    peg = task.get("peg_shape")
    return f"`{exp}`" + (f" (peg `{peg}`)" if peg else "")


def _conditions_table(conditions):
    if not conditions:
        return []
    badge = {"active": "**active**", "inert": "inert", "baseline": "baseline",
             "mode": "-"}
    L = ["| choice | value | in this run | notes |",
         "|---|---|---|---|"]
    for r in conditions:
        L.append(f"| `{r['choice']}` | {r['value']} | "
                 f"{badge.get(r['status'], r['status'])} | {r['note']} |")
    return L


def _operator_block(op):
    if not op:
        return []
    if op.get("human"):
        return ["## Operator", "", "- human teleoperator (SpaceMouse)", ""]
    L = ["## Operator (ground truth)", ""]
    line = f"- id `{op.get('operator_id', '?')}` · tier {op.get('tier', '?')}"
    if op.get("s_exec") is not None:
        line += f" · s_exec {op['s_exec']:.2f} / s_mon {op.get('s_mon', 0):.2f}"
    L.append(line)
    keys = ["sigma0", "v_max", "sigma_perc", "delta_in", "p_miss", "rho_fa",
            "l_react"]
    bits = [f"{k} {op[k]:.3g}" for k in keys
            if isinstance(op.get(k), (int, float))]
    if bits:
        L.append("- θ: " + ", ".join(bits))
    if op.get("theta_path"):
        L.append(f"- full θ: `{op['theta_path']}`")
    return L + [""]


def _reproduce_block(reproduce):
    if not reproduce:
        return []
    cmds = reproduce if isinstance(reproduce, (list, tuple)) else [reproduce]
    return ["## Reproduce", "", "```bash"] + [str(c) for c in cmds] + ["```", ""]


def _provenance_block(m):
    p = m.get("provenance", {})
    git = (f"git {p.get('commit')}" + (" (dirty)" if p.get("dirty") else "")
           if p.get("commit") else "git n/a")
    seeds = m.get("seeds") or {}
    seed_s = " · ".join(f"{k}={v}" for k, v in seeds.items()) if seeds else ""
    line = f"{git} · host {p.get('host', '?')} · created {m.get('created', '')}"
    if seed_s:
        line += f" · seeds {seed_s}"
    return ["## Provenance", "", line, ""]


def render_run_readme(m):
    task = m.get("task", {})
    method = m.get("method", "?")
    op = m.get("operator") or {}
    term = m.get("termination") or {}
    L = [f"# Run — `{method}` · operator `{op.get('operator_id', '?')}` · "
         f"seed {(m.get('seeds') or {}).get('seed', '?')}", ""]
    L.append(f"*{_task_line(task)} · " + MODE_ONE_LINER.get(method, "") + "*")
    L += ["", "## Task & environment", ""]
    L.append(f"- experiment: {_task_line(task)}")
    if task.get("reward"):
        L.append(f"- reward: {task['reward']}")
    if task.get("train_reset"):
        L.append(f"- training reset: {task['train_reset']} (randomized start)")
    L.append("- evaluation: measured separately by the fixed ruler (see the "
             "matching `eval_runs/` artifact), not here")
    L += ["", f"## Method — `{method}`", ""]
    L.append("Every AQI switch below is passed in every mode; the **in this "
             "run** column is derived from the live config, so choices that "
             "do nothing in this mode are marked *inert*.")
    L.append("")
    L += _conditions_table(m.get("conditions"))
    L.append("")
    L += _operator_block(m.get("operator"))
    demos = m.get("demos")
    if demos:
        L += ["## Demonstrations", "",
              f"- {demos.get('n', '?')} "
              f"{'filtered ' if demos.get('filtered') else ''}"
              f"demos ← `{demos.get('path', '?')}`"
              + (f" (sha {demos['sha']})" if demos.get("sha") else ""), ""]
    if term:
        L += ["## Termination", ""]
        reason = term.get("reason")
        if term.get("converged"):
            L.append(f"- **converged** at step {term.get('step')} "
                     f"(rolling {'autonomous ' if term.get('autonomous') else ''}"
                     f"success {term.get('rolling_success')} over "
                     f"{term.get('window')} episodes)")
        elif reason == "time_budget":
            L.append(f"- stopped at the **time budget** "
                     f"({term.get('time_budget_min')} min) without converging")
        elif term.get("converged") is False:
            L.append(f"- did NOT converge; stopped at the "
                     f"{term.get('cap', '?')}-step hard cap")
        if term.get("cap"):
            L.append(f"- step cap: {term['cap']} env steps")
        if term.get("time_budget_min"):
            used = (f" (used {term['wall_minutes']} min)"
                    if term.get("wall_minutes") is not None else "")
            L.append(f"- time budget: {term['time_budget_min']} min{used}")
        L.append("")
    L += _reproduce_block(m.get("reproduce"))
    L += _provenance_block(m)
    return "\n".join(L)


def render_demo_readme(m):
    task = m.get("task", {})
    demos = m.get("demos") or {}
    op = m.get("operator") or {}
    L = [f"# Demonstrations — operator `{op.get('operator_id', '?')}` · "
         f"seed {(m.get('seeds') or {}).get('seed', '?')}", ""]
    L.append(f"*{_task_line(task)}*")
    L += ["", "## What this is", ""]
    L.append(f"- {demos.get('n', '?')} "
             f"{'filtered (successful only)' if demos.get('filtered') else 'raw'}"
             " training-format demonstration episode(s) recorded by a synthetic "
             "operator teleoperating the task through its geometric oracle.")
    L.append(f"- payload: `{demos.get('file', 'demos.pkl')}` · ground truth: "
             f"`{demos.get('ground_truth', 'ground_truth.json')}`")
    if demos.get("attempts") is not None:
        L.append(f"- collection: {demos.get('kept', demos.get('n'))} kept of "
                 f"{demos['attempts']} attempts")
    L += ["", "## Task & environment", ""]
    L.append(f"- experiment: {_task_line(task)}")
    L.append("- reset: randomized start pose (the operator sees varied starts)")
    L.append("- the SAME operator θ recorded here drives this operator's live "
             "interventions during training (Role 1 == Role 2).")
    L.append("")
    L += _operator_block(m.get("operator"))
    L += _reproduce_block(m.get("reproduce"))
    L += _provenance_block(m)
    return "\n".join(L)


def render_eval_readme(m):
    task = m.get("task", {})
    ev = m.get("evaluation") or {}
    op = m.get("operator") or {}
    L = [f"# Evaluation (ruler) — `{m.get('method', '?')}` · operator "
         f"`{op.get('operator_id', '?')}` · "
         f"seed {(m.get('seeds') or {}).get('seed', '?')}", ""]
    L.append(f"*{_task_line(task)}*")
    L += ["", "## Conditions (fixed ruler)", ""]
    L.append("- policy is **frozen**; there is **no operator** and no "
             "randomization here — every cell is measured on the identical, "
             "hash-locked pose grid.")
    L.append(f"- grid: {ev.get('n_trials', '?')} trials, "
             f"sha256 `{str(ev.get('grid_sha256', ''))[:16]}...`")
    L.append("- action selection: "
             + ("argmax (deterministic)" if ev.get("argmax", True)
                else "stochastic"))
    if ev.get("checkpoint_step") is not None:
        L.append(f"- checkpoint step: {ev['checkpoint_step']}")
    if ev.get("f_max") is not None:
        L.append(f"- force violation threshold f_max: {ev['f_max']} N")
    L.append("")
    out = ev.get("outcomes") or {}
    if out:
        sr = out.get("sr")
        L += ["## Result", "",
              (f"- success rate: {100 * sr:.1f}%"
               if isinstance(sr, (int, float)) else "- success rate: n/a")]
        if out.get("n_contaminated"):
            L.append(f"- {out['n_contaminated']} contaminated trial(s) excluded")
        L.append("")
    L.append("Note: the training method (`" + str(m.get("method", "?")) +
             "`) only shaped the policy being tested; the ruler itself is "
             "method- and operator-independent by design.")
    L.append("")
    L += _reproduce_block(m.get("reproduce"))
    L += _provenance_block(m)
    return "\n".join(L)


def render_campaign_readme(m):
    c = m.get("campaign", {})
    L = ["# Campaign — operator-dependency study", ""]
    L.append(f"*{_task_line(m.get('task', {}))} · created {m.get('created', '')}*")
    L += ["", "Automated study of how much a HIL-SERL outcome depends on WHICH "
          "operator trained it, and whether AQI reduces that dependency.", ""]
    L += ["## Design", ""]
    L.append(f"- operators: {c.get('n_operators', '?')} "
             f"({c.get('ops_per_tier', '?')}/tier × tiers "
             f"{', '.join(c.get('tiers', []))})")
    L.append(f"- methods: {', '.join(c.get('methods', []))}")
    L.append(f"- seeds/operator: {c.get('n_seeds', '?')}")
    L.append(f"- demos/(operator,seed): {c.get('demos', '?')}")
    L.append(f"- per-cell budget: converge @ {c.get('converge_success', '?')} "
             f"(cap {c.get('steps', '?')} env steps)")
    if c.get("minutes_per_cell"):
        L.append(f"- fixed time budget: {c['minutes_per_cell']} min/cell")
    if c.get("max_hours"):
        L.append(f"- campaign wall-clock cap: {c['max_hours']} h")
    L.append(f"- ruler grid: {c.get('n_grid_trials', '?')} trials, sha256 "
             f"`{str(c.get('grid_sha256', ''))[:16]}...`")
    L += ["", "## Layout", "", "| folder | what |", "|---|---|"]
    L.append("| `operators/` | per-operator ground-truth θ (`<id>.json`) |")
    L.append("| `demos/<op>_s<seed>/` | that operator's demos + README |")
    L.append("| `checkpoints/<cell>/` | trained policy per cell + README |")
    L.append("| `eval_runs/<cell>/` | fixed-ruler outcomes per cell + README |")
    L.append("| `report/` | the operator-dependency assessment |")
    L += ["", "A *cell* is `<method>_<operator>_s<seed>`. Each artifact folder "
          "carries its own README explaining the exact conditions (and which "
          "choices were inert for that method).", ""]
    L += _provenance_block(m)
    return "\n".join(L)


_RENDER = {
    "run": render_run_readme,
    "demo": render_demo_readme,
    "eval": render_eval_readme,
    "campaign": render_campaign_readme,
}


def render_readme(manifest) -> str:
    return _RENDER.get(manifest.get("kind"), render_run_readme)(manifest)


# ===========================================================================
# safe writers -- a provenance failure must never break the real work
# ===========================================================================
def write_docs(directory, manifest, quiet=False) -> bool:
    """Write ``manifest.json`` + ``README.md`` into ``directory``. Never
    raises; returns True on success."""
    try:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        (d / "manifest.json").write_text(json.dumps(manifest, indent=2))
        (d / "README.md").write_text(render_readme(manifest))
        return True
    except Exception as exc:  # provenance is best-effort
        if not quiet:
            print(f"[provenance] could not write docs to {directory}: {exc}")
        return False


def document(directory, kind, quiet=False, **fields) -> bool:
    """Build a manifest from ``fields`` and write the docs. Never raises."""
    try:
        manifest = build_manifest(kind, **fields)
    except Exception as exc:
        if not quiet:
            print(f"[provenance] could not build manifest: {exc}")
        return False
    return write_docs(directory, manifest, quiet=quiet)
