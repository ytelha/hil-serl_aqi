#!/usr/bin/env python3
"""Automated synthetic-operator campaign runner (operator-dependency study).

This is the end-to-end orchestrator for the thesis' core question: **how much
does a HIL-SERL training outcome depend on WHICH operator trained it, and does
AQI reduce that dependency versus the baseline?**  It runs, unattended:

  1. sample  -- draw 5-10 *distinct* synthetic operators per skill tier (each a
                jittered theta, doc S3.5) and freeze the fixed evaluation "ruler"
                (a hash-locked pose grid);
  2. demos   -- once per (operator, seed): the operator teleoperates the sim via
                its geometric oracle and records training-format demonstrations
                (reused across all methods so the method is the only thing that
                changes between cells);
  3. train   -- for every cell (method x operator x seed): launch the distributed
                learner+actor exactly as a human run would (synchronous agentlace
                loop, unchanged), with the synthetic operator standing in for the
                SpaceMouse and a per-cell env-step budget;
  4. eval    -- score each trained policy on the identical fixed grid with
                eval_ruler.py (policy frozen, no operator, argmax) -> cell.json;
  5. analyze -- for each method compute WITHIN-operator variance (across seeds),
                BETWEEN-operator variance (across operators) and the one-way ICC,
                then compare methods (does AQI shrink the ICC / between-operator
                variance vs baseline HIL-SERL?).

A *cell* is one (method, operator, seed); its id is ``<method>_<operator>_s<seed>``.
A *method* is an ARM: a base ``--aqi_mode`` plus optional per-arm overrides
(YAML ``arms:``), so e.g. two ``quality`` variants can run side by side; see
the "arms" section below and docs/campaign_runner.md.

Layout created under ``--campaign``::

    campaign.json                frozen manifest (params + grid sha256)
    eval_grid.json               the fixed ruler grid (dx x dyaw poses)
    operators.csv                the sampled operator roster
    operators/<op>.json          per-operator ground-truth theta (drives BOTH
                                 demo collection and live interventions);
                                 frozen on first write, never re-sampled
    demos/<op>_s<seed>.pkl       synthetic demonstrations (+ _groundtruth.json)
    checkpoints/<cell>/          trained policy per cell (+ .done marker)
    eval_runs/<cell>/cell.json   ruler outcomes per cell
    logs/<...>.log               every subprocess' stdout+stderr
    report/dependency.{json,md}  the operator-dependency assessment

Every stage is IDEMPOTENT (skips a unit whose output already exists), so the
campaign resumes cleanly after an interruption.  ``--plan`` prints the full plan
without running anything heavy; ``--dry_run`` prints the exact subprocess
commands; ``--stages`` runs a subset.

The learner and actor keep talking over the ordinary synchronous agentlace loop
-- this runner only *starts* and *stops* them; it never changes how they train.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXAMPLES = REPO / "examples"
SIM_ENV_SH = EXAMPLES / "experiments" / "sim_peg_insertion" / "sim_env.sh"

# Make the in-repo packages importable so the sampling/analysis (which run
# in THIS process, not a subprocess) can use them.
for _p in ("serl_launcher", "franka_sim", "serl_robot_infra", "examples"):
    _pp = str(REPO / _p)
    if _pp not in sys.path:
        sys.path.insert(0, _pp)

from serl_launcher.aqi.operator_model import (  # noqa: E402
    make_operator_theta, jitter_theta, theta_to_json, theta_from_json,
    theta_from_scale, TIER_S)
from serl_launcher.aqi import stats as S  # noqa: E402
from serl_launcher.aqi.config import MODES as AQI_MODES  # noqa: E402
from serl_launcher.aqi.config import QUALITY_SCORE_MODES  # noqa: E402
from serl_launcher.hiril.config import CLI_FIELDS as HIRIL_FIELDS  # noqa: E402
import provenance as prov  # noqa: E402


# ===========================================================================
# small helpers
# ===========================================================================
def _green(msg):
    print(f"\033[92m{msg}\033[0m", flush=True)


def _yellow(msg):
    print(f"\033[93m{msg}\033[0m", flush=True)


def _red(msg):
    print(f"\033[91m{msg}\033[0m", flush=True)


def _sha256_of_grid(grid: dict) -> str:
    """Byte-identical to eval_ruler._sha256_of_grid: sha256 over the JSON of
    every key except ``sha256``, sorted."""
    payload = {k: grid[k] for k in sorted(grid) if k != "sha256"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _parse_floats(csv_str: str):
    return [float(x) for x in str(csv_str).split(",") if x.strip() != ""]


# ---- YAML/JSON task-definition config -------------------------------------
def load_config(path: str) -> dict:
    """Load a campaign task-definition file. Prefers YAML (PyYAML); falls back
    to JSON so a config works even without PyYAML installed."""
    text = Path(path).read_text()
    try:
        import yaml
        data = yaml.safe_load(text)
    except ImportError:
        try:
            data = json.loads(text)
        except Exception:
            raise SystemExit(
                "[config] PyYAML is not installed and the file is not valid "
                "JSON. Install it (pip install pyyaml) or use a JSON config.")
    if not isinstance(data, dict):
        raise SystemExit(f"[config] {path} did not parse to a mapping.")
    return data


def _explicit_dests(parser, argv):
    """Return the set of argparse dests the user passed on the command line, so
    those always win over the config file."""
    if argv is None:
        argv = sys.argv[1:]
    toks = {a.split("=", 1)[0] for a in argv}
    dests = set()
    for act in parser._actions:
        if any(opt in toks for opt in act.option_strings):
            dests.add(act.dest)
    return dests


def _csv(v):
    if isinstance(v, (list, tuple)):
        return ",".join(str(x) for x in v)
    return str(v)


def apply_config(args, cfg: dict, explicit) -> dict:
    """Overlay a parsed config onto ``args`` for every field the CLI did NOT set
    explicitly (CLI > config > defaults). Returns the per-tier operator-count
    map (or ``{}`` if the config used a single count)."""
    def setk(dest, value):
        if value is not None and dest not in explicit:
            setattr(args, dest, value)

    task = cfg.get("task") or {}
    setk("exp_name", task.get("exp_name"))
    setk("peg_shape", task.get("peg_shape"))
    if task.get("rpy_box") is not None and "sim_rpy_box" not in explicit:
        v = task["rpy_box"]
        args.sim_rpy_box = (",".join(str(x) for x in v)
                            if isinstance(v, (list, tuple)) else str(v))

    ops = cfg.get("operators") or {}
    tier_keys = [k for k in ops if k not in ("jitter", "sample_seed",
                                             "unified_monitoring",
                                             "tier_s_exec", "tier_s_mon")]
    ops_map = {}
    if tier_keys and "ops_per_tier" not in explicit:
        # per-tier populations from the config (unless the CLI forced a scalar)
        ops_map = {k: int(ops[k]) for k in tier_keys}
    if tier_keys and "tiers" not in explicit:
        args.tiers = ",".join(tier_keys)
    setk("jitter", ops.get("jitter"))
    setk("sample_seed", ops.get("sample_seed"))
    setk("unified_monitoring", ops.get("unified_monitoring"))
    if ops.get("tier_s_exec") is not None and "tier_s_exec" not in explicit:
        args.tier_s_exec = {str(k): float(v)
                            for k, v in dict(ops["tier_s_exec"]).items()}
    if ops.get("tier_s_mon") is not None and "tier_s_mon" not in explicit:
        args.tier_s_mon = {str(k): float(v)
                           for k, v in dict(ops["tier_s_mon"]).items()}

    demos = cfg.get("demos") or {}
    setk("demos", demos.get("per_operator"))
    if demos.get("filtered") is not None and "filtered" not in explicit:
        args.filtered = bool(demos["filtered"])
    setk("demo_max_attempts", demos.get("max_attempts"))

    methods = cfg.get("methods")
    if methods and "methods" not in explicit:
        args.methods = _csv(methods)

    rep = cfg.get("replicates") or {}
    setk("seeds", rep.get("seeds"))
    setk("seed_base", rep.get("seed_base"))

    bud = cfg.get("budget") or {}
    setk("cell_budget_mode", bud.get("mode"))
    setk("converge_success", bud.get("converge_success"))
    setk("converge_window", bud.get("converge_window"))
    setk("converge_min_steps", bud.get("converge_min_steps"))
    setk("converge_eval_threshold", bud.get("converge_eval_threshold"))
    setk("converge_eval_max_steps", bud.get("converge_eval_max_steps"))
    setk("converge_mode", bud.get("converge_mode"))
    setk("converge_eval_every", bud.get("converge_eval_every"))
    setk("converge_eval_consecutive", bud.get("converge_eval_consecutive"))
    setk("utd_ratio", bud.get("utd_ratio"))
    if bud.get("converge_eval_poses") is not None and "converge_eval_poses" not in explicit:
        v = bud["converge_eval_poses"]
        args.converge_eval_poses = (";".join(str(x) for x in v)
                                    if isinstance(v, (list, tuple)) else str(v))
    setk("steps", bud.get("steps_cap"))
    setk("learner_max_steps", bud.get("learner_max_steps"))
    setk("demo_pretrain", bud.get("demo_pretrain"))
    setk("demo_pretrain_steps", bud.get("demo_pretrain_steps"))
    setk("demo_pretrain_batch_size", bud.get("demo_pretrain_batch_size"))
    setk("learning_rate", bud.get("learning_rate"))
    setk("minutes_per_cell", bud.get("minutes_per_cell"))
    setk("max_hours", bud.get("total_hours"))
    setk("buffer_period", bud.get("buffer_period"))
    setk("checkpoint_keep", bud.get("checkpoint_keep"))

    ev = cfg.get("eval") or {}
    if ev.get("dx_mm") is not None and "eval_dx_mm" not in explicit:
        args.eval_dx_mm = _csv(ev["dx_mm"])
    if ev.get("yaw") is not None and "eval_yaw" not in explicit:
        args.eval_yaw = _csv(ev["yaw"])
    setk("eval_reps", ev.get("reps"))
    setk("eval_max_steps", ev.get("max_steps"))
    setk("eval_last_k", ev.get("last_k"))
    setk("eval_select", ev.get("select"))
    setk("eval_window_k", ev.get("window_k"))

    gpu = cfg.get("gpu") or {}
    setk("jax_platform", gpu.get("jax_platform"))
    if gpu.get("mem_learner") is not None and "gpu_mem_learner" not in explicit:
        args.gpu_mem_learner = str(gpu["mem_learner"])
    if gpu.get("mem_actor") is not None and "gpu_mem_actor" not in explicit:
        args.gpu_mem_actor = str(gpu["mem_actor"])

    ana = cfg.get("analysis") or {}
    setk("baseline", ana.get("baseline"))
    if ana.get("plots") is not None and "plots" not in explicit:
        args.plots = bool(ana["plots"])

    log = cfg.get("logging") or {}
    if log.get("wandb") is not None and "wandb" not in explicit:
        # PyYAML parses bare off/no/on/yes as booleans (YAML 1.1); normalize
        # those and any string to the valid off/offline/online choices.
        w = log["wandb"]
        if isinstance(w, bool):
            w = "online" if w else "off"
        w = str(w).strip().lower()
        if w not in ("off", "offline", "online"):
            raise SystemExit(f"[config] logging.wandb must be off/offline/online, "
                             f"got {log['wandb']!r}.")
        args.wandb = w
    if log.get("wandb_tags") is not None and "wandb_tags" not in explicit:
        args.wandb_tags = _csv(log["wandb_tags"])
    if (log.get("takeover_channels") is not None
            and "log_takeover_channels" not in explicit):
        args.log_takeover_channels = _to_bool(log["takeover_channels"])

    aqi = cfg.get("aqi") or {}
    if aqi.get("trigger_terms") is not None and "trigger_terms" not in explicit:
        args.trigger_terms = _csv(aqi["trigger_terms"])
    if (aqi.get("trigger_combiner") is not None
            and "trigger_combiner" not in explicit):
        args.trigger_combiner = str(aqi["trigger_combiner"])
    if aqi.get("quality_filters") is not None and "quality_filters" not in explicit:
        # an explicit empty list means NO filters ("none"); an empty string
        # would emit no flag and silently fall back to the default filters.
        args.quality_filters = _csv(aqi["quality_filters"]) or "none"
    if (aqi.get("quality_disagreement_gate") is not None
            and "quality_disagreement_gate" not in explicit):
        args.quality_disagreement_gate = bool(aqi["quality_disagreement_gate"])
    if aqi.get("bc_loss_weight") is not None and "bc_loss_weight" not in explicit:
        args.bc_loss_weight = float(aqi["bc_loss_weight"])
    if aqi.get("per_standardize") is not None and "per_standardize" not in explicit:
        args.per_standardize = bool(aqi["per_standardize"])
    if aqi.get("per_delta_init") is not None and "per_delta_init" not in explicit:
        args.per_delta_init = float(aqi["per_delta_init"])
    if aqi.get("per_eta") is not None and "per_eta" not in explicit:
        args.per_eta = float(aqi["per_eta"])
    if aqi.get("per_eps") is not None and "per_eps" not in explicit:
        args.per_eps = float(aqi["per_eps"])
    if (aqi.get("quality_warmup_steps") is not None
            and "quality_warmup_steps" not in explicit):
        args.quality_warmup_steps = int(aqi["quality_warmup_steps"])
    if aqi.get("per_scale_decay") is not None and "per_scale_decay" not in explicit:
        args.per_scale_decay = float(aqi["per_scale_decay"])
    if (aqi.get("per_refresh_period") is not None
            and "per_refresh_period" not in explicit):
        args.per_refresh_period = int(aqi["per_refresh_period"])
    if aqi.get("per_is_beta") is not None and "per_is_beta" not in explicit:
        args.per_is_beta = float(aqi["per_is_beta"])
    if (aqi.get("quality_score_mode") is not None
            and "quality_score_mode" not in explicit):
        args.quality_score_mode = str(aqi["quality_score_mode"])
    if (aqi.get("quality_behavioral_terms") is not None
            and "quality_behavioral_terms" not in explicit):
        args.quality_behavioral_terms = _csv(aqi["quality_behavioral_terms"])
    for key, cast in (("quality_post_release_steps", int),
                      ("quality_hindsight_discount", float),
                      ("quality_consistency_weight", bool),
                      ("quality_consistency_k", int),
                      ("quality_consistency_sigma_ref", float),
                      ("query_target_rate", float),
                      ("query_cooldown_steps", int),
                      ("query_min_on_steps", int),
                      ("quality_cf_diag", bool),
                      ("quality_drop_fraction", float)):
        if aqi.get(key) is not None and key not in explicit:
            setattr(args, key, cast(aqi[key]))
    if (aqi.get("synthetic_inconsistent") is not None
            and "synthetic_inconsistent" not in explicit):
        args.synthetic_inconsistent = _to_bool(aqi["synthetic_inconsistent"])
    for k in ("synthetic_bad_share", "synthetic_bad_persistence"):
        if aqi.get(k) is not None and k not in explicit:
            setattr(args, k, float(aqi[k]))
    if (aqi.get("synthetic_intervention_mode") is not None
            and "synthetic_intervention_mode" not in explicit):
        args.synthetic_intervention_mode = str(
            aqi["synthetic_intervention_mode"])
    if (aqi.get("synthetic_rotation_control") is not None
            and "synthetic_rotation_control" not in explicit):
        args.synthetic_rotation_control = _to_bool(
            aqi["synthetic_rotation_control"])
    if (aqi.get("synthetic_orientation_monitor") is not None
            and "synthetic_orientation_monitor" not in explicit):
        args.synthetic_orientation_monitor = _to_bool(
            aqi["synthetic_orientation_monitor"])
    for k in ("synthetic_orientation_tilt_rad", "synthetic_orientation_yaw_rad"):
        if aqi.get(k) is not None and k not in explicit:
            setattr(args, k, float(aqi[k]))
    if (aqi.get("critic_ensemble_size") is not None
            and "critic_ensemble_size" not in explicit):
        args.critic_ensemble_size = int(aqi["critic_ensemble_size"])
    if (aqi.get("critic_subsample_size") is not None
            and "critic_subsample_size" not in explicit):
        args.critic_subsample_size = int(aqi["critic_subsample_size"])
    if (aqi.get("per_member_targets") is not None
            and "per_member_targets" not in explicit):
        args.per_member_targets = bool(aqi["per_member_targets"])
    for key, cast in (("e2hil_K", int), ("e2hil_q_low", float),
                      ("e2hil_q_high", float), ("e2hil_q_agg", str)):
        if aqi.get(key) is not None and key not in explicit:
            setattr(args, key, cast(aqi[key]))

    mo = cfg.get("method_overrides")
    if mo is not None and "method_overrides" not in explicit:
        args.method_overrides = dict(mo)
    hir = cfg.get("hiril") or {}
    for name, cast in HIRIL_FIELDS.items():
        if hir.get(name) is not None and f"hiril_{name}" not in explicit:
            setattr(args, f"hiril_{name}",
                    _to_bool(hir[name]) if cast is bool else cast(hir[name]))
    unknown = sorted(set(hir) - set(HIRIL_FIELDS))
    if unknown:
        raise SystemExit(f"[config] unknown hiril keys {unknown}; valid: "
                         f"{sorted(HIRIL_FIELDS)}")
    arms = cfg.get("arms")
    if arms is not None:
        if not isinstance(arms, dict):
            raise SystemExit("[config] `arms:` must map arm names to "
                             "{mode: <aqi_mode>, <knob>: <value>, ...}.")
        args.arms = {str(k): dict(v or {}) for k, v in arms.items()}

    por = cfg.get("policy_oracle") or {}
    if por.get("demo_source") is not None and "demo_source" not in explicit:
        args.demo_source = str(por["demo_source"])
    setk("policy_oracle_checkpoint", por.get("checkpoint_path"))
    if por.get("aqi_mode") is not None and "policy_oracle_mode" not in explicit:
        args.policy_oracle_mode = str(por["aqi_mode"])
    setk("policy_oracle_step", por.get("checkpoint_step"))
    setk("policy_mm_per_unit", por.get("mm_per_unit"))
    if por.get("training_oracle") is not None and "training_oracle" not in explicit:
        args.training_oracle = str(por["training_oracle"])
    return ops_map


def _ops_per_tier(args, tier):
    m = getattr(args, "ops_per_tier_map", None)
    if m:
        return int(m.get(tier, 0))
    return int(args.ops_per_tier)


def _ops_effective(args):
    """int (uniform) or dict (per-tier) to hand to sample_operators."""
    m = getattr(args, "ops_per_tier_map", None)
    return m if m else args.ops_per_tier


def _actor_budget(args):
    """Return (actor_steps, actor_minutes) for the selected cell budget mode.

    Exactly one limiter is active:
      - steps mode: env-step cap only
      - time mode: wall-clock cap only
    """
    mode = getattr(args, "cell_budget_mode", "steps")
    if mode == "steps":
        return int(args.steps), 0.0
    # Actor loop still needs a finite max_steps; keep it effectively unbounded
    # so wall-clock is the real stop criterion.
    return 1_000_000_000, float(args.minutes_per_cell)


def _has_checkpoint(ckpt_dir: Path) -> bool:
    return ckpt_dir.is_dir() and any(
        p.name.startswith("checkpoint_") and p.is_dir()
        for p in ckpt_dir.iterdir())


# ---- provenance helpers ---------------------------------------------------
def _demo_dir(campaign: Path, operator: str, seed: int) -> Path:
    """Each (operator, seed) demo set is its OWN folder (payload + docs)."""
    return campaign / "demos" / f"{operator}_s{seed}"


def _demo_pkl(campaign: Path, operator: str, seed: int) -> Path:
    return _demo_dir(campaign, operator, seed) / "demos.pkl"


def _roster_theta(campaign: Path, operator: str):
    p = campaign / "operators" / f"{operator}.json"
    try:
        return theta_from_json(str(p)) if p.exists() else None
    except Exception:
        return None


def _operator_profile(campaign: Path, operator: str) -> str:
    """Map roster operator id to a valid synthetic profile for demo tools."""
    th = _roster_theta(campaign, operator)
    if th is not None:
        return str(th.tier)
    low = operator.lower()
    if low.startswith("exp"):
        return "expert"
    if low.startswith("int"):
        return "intermediate"
    if low.startswith("nov"):
        return "novice"
    return "expert"


def _task_dict(args):
    d = {"exp_name": args.exp_name, "peg_shape": args.peg_shape,
         "reward": "sim ground truth (lateral < 12 mm and depth > 50 mm)"
         if args.exp_name == "sim_peg_insertion" else None,
         "train_reset": "randomized +/-15 mm xy, +/-0.08 rad yaw"
         if args.exp_name == "sim_peg_insertion" else None}
    if _rpy_box(args):              # sim orientation box: part of the task
        d["rpy_box"] = _rpy_box(args)
    return d


def _operator_dict(th, campaign: Path):
    d = {"operator_id": th.operator_id, "tier": th.tier,
         "s_exec": getattr(th, "s_exec", None), "s_mon": getattr(th, "s_mon", None),
         "theta_path": f"../../operators/{th.operator_id}.json"}
    for k in ("sigma0", "v_max", "sigma_perc", "delta_in", "p_miss", "rho_fa",
              "l_react"):
        d[k] = getattr(th, k, None)
    return d


def _method_conditions(method, args):
    """Conditions rows for an ARM, derived from a real AQIConfig built with this
    arm's resolved knobs (campaign-wide values + per-arm overrides), so inert
    choices are labelled exactly as the runtime sees them, plus rows for the
    learner knobs that can differ per arm (warm start, learning rate).
    Unlisted AQIConfig fields show dataclass defaults, not the experiment's
    get_aqi_config() values."""
    mode = _arm_mode(args, method)

    def val(key):
        return _arm_val(args, method, key, getattr(args, key, None))

    if mode in PIPELINES:
        from serl_launcher.hiril.config import HiRILConfig
        kw = {n: val(f"hiril_{n}") for n in HIRIL_FIELDS
              if val(f"hiril_{n}") is not None}
        rows = prov.hiril_condition_rows(HiRILConfig(**kw).as_dict())
        dp = bool(val("demo_pretrain"))
        dps = int(val("demo_pretrain_steps") or 0)
        rows.append({"choice": "demo_pretrain",
                     "value": f"{dps} steps" if dp and dps > 0 else "off",
                     "status": "active" if dp and dps > 0 else "inert",
                     "note": "demo-only learner warm start (not in the paper)"})
        return rows

    def as_tuple(v):
        v = "" if v is None else str(v)
        return () if v in ("", "none") else tuple(t for t in v.split(",") if t)

    try:
        from serl_launcher.aqi import AQIConfig
        kw = {"mode": mode}
        for key in ("trigger_terms", "quality_filters",
                    "quality_behavioral_terms"):
            if val(key):
                kw[key] = as_tuple(val(key))
        for key, field in (
                ("trigger_combiner", "trigger_combiner"),
                ("quality_disagreement_gate", "quality_disagreement_gate"),
                ("bc_loss_weight", "bc_loss_weight"),
                ("quality_score_mode", "quality_score_mode"),
                ("per_standardize", "per_standardize"),
                ("per_delta_init", "per_delta_init"),
                ("per_eta", "per_eta"), ("per_eps", "per_eps"),
                ("quality_warmup_steps", "quality_warmup_steps"),
                ("per_scale_decay", "per_scale_decay"),
                ("per_refresh_period", "per_refresh_period"),
                ("per_is_beta", "per_is_beta"),
                ("quality_post_release_steps", "quality_post_release_steps"),
                ("quality_hindsight_discount", "quality_hindsight_discount"),
                ("quality_consistency_weight", "quality_consistency_weight"),
                ("quality_consistency_k", "quality_consistency_k"),
                ("quality_consistency_sigma_ref",
                 "quality_consistency_sigma_ref"),
                ("quality_cf_diag", "quality_cf_diag"),
                ("critic_ensemble_size", "critic_ensemble_size"),
                ("per_member_targets", "critic_per_member_targets"),
                ("e2hil_K", "e2hil_K"), ("e2hil_q_low", "e2hil_q_low"),
                ("e2hil_q_high", "e2hil_q_high"),
                ("e2hil_q_agg", "e2hil_q_agg")):
            if val(key) not in (None, ""):
                kw[field] = val(key)
        # mirror train_aqi_rlpd: M defaults to min(2, K), an explicit M wins,
        # always clamped to K
        k = int(kw.get("critic_ensemble_size", 10))
        m = val("critic_subsample_size")
        kw["critic_subsample_size"] = min(int(m) if m is not None else 2, k)
        rows = prov.condition_rows(AQIConfig(**kw))
    except Exception:
        rows = [{"choice": "aqi_mode", "value": mode, "status": "mode",
                 "note": prov.MODE_ONE_LINER.get(mode, "")}]
    lr = float(val("learning_rate") or 0)
    rows.append({"choice": "learning_rate",
                 "value": f"{lr:g}" if lr > 0 else "3e-4 (built-in)",
                 "status": "active",
                 "note": "Adam LR of actor, critic and temperature"})
    dp, dps = bool(val("demo_pretrain")), int(val("demo_pretrain_steps") or 0)
    rows.append({"choice": "demo_pretrain",
                 "value": f"{dps} steps" if dp and dps > 0 else "off",
                 "status": "active" if dp and dps > 0 else "inert",
                 "note": "demo-only learner warm start before online RL"})
    ovr = _arm_overrides(args, method)
    if ovr:
        rows.insert(1, {"choice": "arm", "value": method, "status": "arm",
                        "note": "per-arm overrides: " + ", ".join(
                            f"{k}={v}" for k, v in ovr.items())})
    return rows


# ===========================================================================
# stage 1a: sample the operator roster (distinct jittered operators per tier)
# ===========================================================================
_TIER_ABBR = {"expert": "exp", "intermediate": "int", "novice": "nov"}

# columns surfaced in operators.csv (a readable slice of the frozen theta)
_ROSTER_COLS = ["operator_id", "tier", "s_exec", "s_mon", "sigma0",
                "sigma_perc", "v_max", "latency", "delta_in", "rho_fa",
                "p_miss", "l_react"]


# execution-only scale-like params (exclude the monitoring params sigma_mon /
# rho_fa) so a unified monitoring signal stays identical across operators.
_EXEC_JITTER_KEYS = {"sigma0", "k_sdn", "sigma_tr", "sigma_perc",
                     "pause_len_mean"}


def sample_operators(tiers, ops_per_tier, sample_seed, jitter,
                     unified_monitoring="", tier_s_exec=None, tier_s_mon=None):
    """Draw ``ops_per_tier`` DISTINCT operators for each tier by jittering the
    tier anchor theta (doc S3.5): tiers become overlapping clouds, so within a
    tier operators genuinely differ (that is the between-operator variance the
    study measures). Deterministic in ``sample_seed``.

    When ``unified_monitoring`` names a tier, every operator's MONITORING skill
    (the intervention signal: when/whether to take over) is pinned to that tier
    and left un-jittered, so operators differ ONLY in EXECUTION skill (how well
    the correction is carried out). This isolates the execution channel.

    ``tier_s_exec`` / ``tier_s_mon`` map a tier name to a continuous skill scale
    in [0, 1] that OVERRIDES the tier anchor's execution / monitoring skill
    (0 = novice anchor, 1 = expert anchor). Use e.g. {"novice": 0.25} to make a
    weak-but-informative novice (sigma0 ~ 5 mm/s instead of 10) whose
    corrections are worse than an expert's yet still better than the untrained
    policy -- so a quality/active gate can LIFT it rather than only ignore it."""
    import numpy as np
    rng = np.random.default_rng(sample_seed)
    tier_s_exec = tier_s_exec or {}
    tier_s_mon = tier_s_mon or {}
    roster = []
    for tier in tiers:
        se = float(tier_s_exec.get(tier, TIER_S[tier]))
        sm_default = (TIER_S[unified_monitoring] if unified_monitoring
                      else TIER_S[tier])
        sm = float(tier_s_mon.get(tier, sm_default))
        if unified_monitoring or tier in tier_s_exec or tier in tier_s_mon:
            anchor = theta_from_scale(se, sm)
            # pin + un-jitter monitoring only when it is unified across tiers
            jit_keys = _EXEC_JITTER_KEYS if unified_monitoring else None
        else:
            anchor = make_operator_theta(tier)
            jit_keys = None
        anchor.tier = tier
        abbr = _TIER_ABBR.get(tier, tier[:3])
        n = ops_per_tier[tier] if isinstance(ops_per_tier, dict) else ops_per_tier
        for i in range(int(n)):
            th = jitter_theta(anchor, rng, strength=jitter, keys=jit_keys)
            th.tier = tier
            th.operator_id = f"{abbr}{i:02d}"
            roster.append(th)
    return roster


def freeze_roster(roster, campaign: Path):
    """The roster is FROZEN on first write, like the eval ruler: an operator
    whose operators/<id>.json already exists keeps that theta -- the demos
    and every trained cell used it -- and only new operators take the
    freshly sampled one. Returns (roster, ids whose file differs from what
    the sampler draws now)."""
    out, differs = [], []
    for th in roster:
        old = _roster_theta(campaign, th.operator_id)
        if old is None:
            out.append(th)
            continue
        a, b = old.as_dict(), th.as_dict()
        if any(not (a.get(k) == v or (isinstance(v, float)
                                      and isinstance(a.get(k), float)
                                      and abs(a[k] - v) <= 1e-9 * max(1.0, abs(v))))
               for k, v in b.items()):
            differs.append(th.operator_id)
        out.append(old)
    return out, differs


def write_roster(roster, campaign: Path):
    """Write operators/<id>.json for operators that have none (existing files
    are the frozen roster, see freeze_roster) and refresh operators.csv."""
    (campaign / "operators").mkdir(parents=True, exist_ok=True)
    for th in roster:
        p = campaign / "operators" / f"{th.operator_id}.json"
        if not p.exists():
            theta_to_json(th, str(p))
    with open(campaign / "operators.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(_ROSTER_COLS)
        for th in roster:
            w.writerow([getattr(th, c) for c in _ROSTER_COLS])


# ===========================================================================
# stage 1b: freeze the ruler grid
# ===========================================================================
def build_grid(dx_mm, dyaw_rad, reps, peg_shape):
    """A fixed pose grid: the outer product of lateral offsets dx and yaw
    offsets dyaw, repeated ``reps`` times, in a fixed order. The policy+env are
    deterministic under argmax so one rep already pins the outcome; reps>1 only
    helps if you later evaluate stochastically."""
    trials = []
    for rep in range(int(reps)):
        for dx in dx_mm:
            for yaw in dyaw_rad:
                trials.append({"dx_mm": float(dx), "dyaw_rad": float(yaw),
                               "rep": int(rep)})
    grid = {
        "trials": trials,
        "dx_mm": [float(x) for x in dx_mm],
        "dyaw_rad": [float(x) for x in dyaw_rad],
        "reps": int(reps),
        "n_poses": len(dx_mm) * len(dyaw_rad),
        "peg_shape": peg_shape,
        "note": "fixed operator-independent ruler; every cell is scored on "
                "exactly these poses (set_eval_reset_pose hook).",
    }
    grid["sha256"] = _sha256_of_grid(grid)
    return grid


# ===========================================================================
# cells
# ===========================================================================
class Cell:
    __slots__ = ("method", "operator", "tier", "seed")

    def __init__(self, method, operator, tier, seed):
        self.method = method
        self.operator = operator
        self.tier = tier
        self.seed = seed

    @property
    def id(self):
        return f"{self.method}_{self.operator}_s{self.seed}"


def _parse_shard(spec):
    """'i/n' -> (i, n) with 0 <= i < n; '' -> None."""
    spec = (spec or "").strip()
    if not spec:
        return None
    try:
        i, n = (int(x) for x in spec.split("/"))
    except ValueError:
        raise SystemExit(f"[shard] expected 'i/n', got {spec!r}")
    if not (n >= 1 and 0 <= i < n):
        raise SystemExit(f"[shard] need 0 <= i < n, got {spec!r}")
    return i, n


def enumerate_cells(methods, roster, seeds):
    tier_of = {th.operator_id: th.tier for th in roster}
    cells = []
    for method in methods:
        for th in roster:
            for s in seeds:
                cells.append(Cell(method, th.operator_id, tier_of[th.operator_id], s))
    return cells


# ===========================================================================
# subprocess plumbing (everything runs inside the sim env via sim_env.sh)
# ===========================================================================
def _launch(pyargs, log_path: Path, extra_env=None, background=False):
    """Run ``python <pyargs>`` inside the sim environment (sources sim_env.sh so
    the venv, PYTHONPATH, MUJOCO_GL and hidapi are set exactly as an interactive
    run). Returns (proc, logfile) when background else the return code."""
    inner = ("source " + shlex.quote(str(SIM_ENV_SH)) + " && exec python "
             + " ".join(shlex.quote(str(a)) for a in pyargs))
    env = dict(os.environ)
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logf = open(log_path, "w")
    logf.write("# " + inner + "\n")
    logf.flush()
    proc = subprocess.Popen(["bash", "-c", inner], cwd=str(EXAMPLES), env=env,
                            stdout=logf, stderr=subprocess.STDOUT,
                            start_new_session=True)
    if background:
        return proc, logf
    rc = proc.wait()
    logf.close()
    return rc


def _wait_for_banner(proc, log_path: Path, needle: str, timeout: float) -> bool:
    """Poll a background process' log until ``needle`` appears (learner READY),
    the process dies, or the timeout elapses."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if log_path.exists():
            try:
                if needle in log_path.read_text(errors="ignore"):
                    return True
            except OSError:
                pass
        if proc.poll() is not None:
            return False               # exited before the banner -> failure
        time.sleep(1.0)
    return False


def _terminate(proc, grace=15.0):
    """Stop a background process group politely, then hard."""
    if proc.poll() is not None:
        return proc.returncode
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        return proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        return proc.wait()


# ===========================================================================
# GPU pre-flight (the GPU is shared -- never clobber someone else's run)
# ===========================================================================
def gpu_report():
    """Return (used_mb_by_others, [(pid, mb, name), ...]) from nvidia-smi, or
    (None, []) if nvidia-smi is unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-compute-apps=pid,used_memory,process_name",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None, []
    mine = os.getpid()
    procs, other_mb = [], 0
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        pid_s, mb_s, name = (x.strip() for x in line.split(",", 2))
        try:
            pid, mb = int(pid_s), int(float(mb_s))
        except ValueError:
            continue
        procs.append((pid, mb, name))
        if pid != mine:
            other_mb += mb
    return other_mb, procs


def gpu_preflight(args):
    if args.jax_platform == "cpu" or args.force or args.dry_run:
        return
    other_mb, procs = gpu_report()
    if other_mb is None:
        _yellow("[gpu] nvidia-smi not found; skipping the GPU pre-flight check.")
        return
    _green(f"[gpu] {other_mb} MB in use by other processes "
           f"({len(procs)} compute app(s) visible).")
    if other_mb > args.gpu_free_mb:
        try:
            mem_frac = float(args.gpu_mem_learner) + float(args.gpu_mem_actor)
        except (TypeError, ValueError):
            mem_frac = None
        _red(f"[gpu] {other_mb} MB already used by other processes exceeds the "
             f"--gpu_free_mb={args.gpu_free_mb} budget. The learner+actor need "
             f"~{mem_frac:.0%} of VRAM. Wait for the GPU to free up, lower "
             f"--gpu_mem_learner/--gpu_mem_actor, or pass --force to proceed "
             f"anyway." if mem_frac is not None else
             f"the configured GPU memory fractions could not be parsed. Wait "
             f"for the GPU to free up, lower --gpu_mem_learner/--gpu_mem_actor, "
             f"or pass --force to proceed anyway.")
        raise SystemExit(2)


# ===========================================================================
# stage 2: demos (once per operator x seed, reused across methods)
# ===========================================================================
def collect_demos_job(args, campaign: Path, operator: str, seed: int):
    ddir = _demo_dir(campaign, operator, seed)
    out = _demo_pkl(campaign, operator, seed)
    if out.exists():
        _green(f"[demos] {operator}_s{seed} exists -- skip")
        return True
    theta_json = campaign / "operators" / f"{operator}.json"
    profile = _operator_profile(campaign, operator)
    if args.demo_source == "policy":
        if not args.policy_oracle_checkpoint:
            _red("[demos] demo_source=policy requires --policy_oracle_checkpoint")
            return False
        pyargs = ["collect_policy_synthetic_demos.py",
                  "--exp_name", args.exp_name,
                  "--checkpoint_path", str(args.policy_oracle_checkpoint),
                  "--aqi_mode", str(args.policy_oracle_mode),
                  "--checkpoint_step", str(args.policy_oracle_step),
                  "--profile", profile,
                  "--operator_id", operator,
                  "--episodes", str(args.demos),
                  ("--filtered" if args.filtered else "--nofiltered"),
                  "--max_attempts", str(args.demo_max_attempts),
                  "--seed", str(seed),
                  "--policy_env_mm_per_unit", str(args.policy_mm_per_unit),
                  "--out", str(out)]
    else:
        pyargs = ["collect_synthetic_demos.py",
                  "--env",
                  "experiments.sim_peg_insertion.wrapper:make_synthetic_sim_env",
                  "--theta_json", str(theta_json),
                  "--operator_id", operator,
                  "--episodes", str(args.demos),
                  ("--filtered" if args.filtered else "--nofiltered"),
                  "--max_attempts", str(args.demo_max_attempts),
                  "--seed", str(seed),
                  "--out", str(out)]
    if args.dry_run:
        _yellow("[demos] DRY: python " + " ".join(pyargs))
        return True
        _green(f"[demos] collecting {operator} s{seed} "
            f"({args.demos} {'successes' if args.filtered else 'episodes'}, "
            f"source={args.demo_source}) ...")
    ddir.mkdir(parents=True, exist_ok=True)
    rc = _launch(pyargs, campaign / "logs" / f"demos_{operator}_s{seed}.log",
                 # demo collection is pure MuJoCo+numpy -> keep it off the GPU
                 extra_env={"JAX_PLATFORMS": "cpu",
                            "SIM_SYN_PEG_SHAPE": args.peg_shape,
                            **_sim_env(args)})
    if rc != 0 or not out.exists():
        _red(f"[demos] FAILED for {operator} s{seed} (rc={rc}); see the log.")
        return False
    _document_demos(args, campaign, operator, seed)
    return True


def _document_demos(args, campaign: Path, operator: str, seed: int):
    """Write README.md + manifest.json next to this operator's demos."""
    ddir = _demo_dir(campaign, operator, seed)
    th = _roster_theta(campaign, operator)
    gt = {}
    gt_path = ddir / "demos_groundtruth.json"
    if gt_path.exists():
        try:
            gt = json.load(open(gt_path))
        except (ValueError, OSError):
            gt = {}
    profile = _operator_profile(campaign, operator)
    if args.demo_source == "policy":
        repro = (f"# reproduce these demos\n./.venv/bin/python "
                 f"examples/collect_policy_synthetic_demos.py --exp_name "
                 f"{args.exp_name} --checkpoint_path "
                 f"{args.policy_oracle_checkpoint} --aqi_mode "
                 f"{args.policy_oracle_mode} --checkpoint_step "
                 f"{args.policy_oracle_step} --profile "
                 f"{profile} --operator_id {operator} "
                 f"--episodes {args.demos} "
                 f"{'--filtered' if args.filtered else '--nofiltered'} "
                 f"--seed {seed} --policy_env_mm_per_unit "
                 f"{args.policy_mm_per_unit} --out "
                 f"demos/{operator}_s{seed}/demos.pkl")
    else:
        repro = (f"# reproduce these demos\n./.venv/bin/python "
                 f"examples/collect_synthetic_demos.py --env "
                 f"experiments.sim_peg_insertion.wrapper:make_synthetic_sim_env "
                 f"--theta_json operators/{operator}.json --operator_id "
                 f"{operator} --episodes {args.demos} "
                 f"{'--filtered' if args.filtered else '--nofiltered'} "
                 f"--seed {seed} --out demos/{operator}_s{seed}/demos.pkl")
    prov.document(
        ddir, "demo", quiet=True, task=_task_dict(args),
        operator=_operator_dict(th, campaign) if th else {"operator_id": operator},
        seeds={"seed": seed},
        demos={"n": gt.get("episodes_kept", args.demos),
               "kept": gt.get("episodes_kept"), "attempts": gt.get("attempts"),
               "filtered": args.filtered, "file": "demos.pkl",
               "ground_truth": "demos_groundtruth.json"},
        reproduce=[repro])



# ===========================================================================
# stage 3: train one cell (distributed learner+actor, synchronous loop)
# ===========================================================================
# (campaign key, kind) in the EXACT order the flags are emitted. kind:
#   "str"  -> "--key v" if v is a non-empty string (csv lists, enums)
#   "val"  -> "--key v" if v is not None
#   "bool" -> "--key" / "--nokey" if v is not None
# train_aqi_rlpd.py drops any of these that do not apply to the run's mode.
_AQI_FLAG_SPEC = (
    ("trigger_terms", "str"), ("trigger_combiner", "str"),
    ("tau_A", "val"), ("query_target_rate", "val"),
    ("query_cooldown_steps", "val"), ("query_min_on_steps", "val"),
    ("random_request_rate", "val"), ("bootstrap_p", "val"),
    ("synthetic_request_or", "bool"), ("synthetic_request_hold_steps", "val"),
    ("synthetic_handover_steps", "val"),
    ("quality_filters", "str"), ("quality_disagreement_gate", "bool"),
    ("bc_loss_weight", "val"), ("per_standardize", "bool"),
    ("per_delta_init", "val"), ("per_eta", "val"), ("per_eps", "val"),
    ("quality_warmup_steps", "val"), ("per_scale_decay", "val"),
    ("per_refresh_period", "val"), ("per_is_beta", "val"),
    ("quality_score_mode", "val"), ("quality_behavioral_terms", "str"),
    ("quality_post_release_steps", "val"),
    ("quality_hindsight_discount", "val"),
    ("quality_consistency_weight", "bool"), ("quality_consistency_k", "val"),
    ("quality_consistency_sigma_ref", "val"),
    ("quality_cf_diag", "bool"), ("quality_drop_fraction", "val"),
    ("critic_ensemble_size", "val"), ("critic_subsample_size", "val"),
    ("per_member_targets", "bool"),
    ("e2hil_K", "val"), ("e2hil_q_low", "val"), ("e2hil_q_high", "val"),
    ("e2hil_q_agg", "val"),
)


def _aqi_flags(args, arm=None):
    """AQI composition flags shared by learner + actor, resolved for ``arm``
    (campaign-wide value unless the arm overrides it). Safe on every mode --
    train_aqi_rlpd drops the ones irrelevant to the mode (e.g. hilserl ignores
    all of them)."""
    f = []
    for key, kind in _AQI_FLAG_SPEC:
        v = getattr(args, key, None)
        if arm is not None:
            v = _arm_val(args, arm, key, v)
        if kind == "str":
            if v:
                f += [f"--{key}", str(v)]
        elif kind == "bool":
            if v is not None:
                f += [f"--{key}" if v else f"--no{key}"]
        elif v is not None:
            f += [f"--{key}", str(v)]
    return f


def _wandb_flags(args):
    """The learner is the only process that opens wandb. Campaign cells run
    unattended (background, no TTY), so 'online' wandb would block on the
    api-key prompt and crash the learner before it reports READY. Default
    'off' passes --debug (wandb mode='disabled'); 'offline' logs locally via
    the WANDB_MODE env var (see _wandb_env); 'online' passes nothing and needs
    WANDB_API_KEY in the environment."""
    return ["--debug"] if getattr(args, "wandb", "off") == "off" else []


def _wandb_env(args, campaign: Path | None = None):
    """Extra subprocess env for wandb. 'offline' writes local runs without a
    key or TTY; the other modes need nothing here."""
    env = ({"WANDB_MODE": "offline"}
           if getattr(args, "wandb", "off") == "offline" else {})
    if campaign is not None and not os.environ.get("WANDB_RUN_GROUP"):
        env["WANDB_RUN_GROUP"] = campaign.name
    return env


WANDB_TAG_MAX = 64   # W&B rejects longer tags


def _wandb_tag_env(args, campaign: Path, cell) -> dict:
    """W&B tags for one cell's learner: campaign, arm, base mode (named arms
    only), tier, operator, seed, plus --wandb_tags / logging.wandb_tags. They
    travel in AQI_WANDB_TAGS (read by serl_launcher.common.wandb.WandBLogger)
    instead of a CLI flag, so every training command -- and with it every
    --dry_run snapshot -- stays unchanged. Empty when wandb is off."""
    if getattr(args, "wandb", "off") == "off":
        return {}
    mode = _arm_mode(args, cell.method)
    tags = ([campaign.name, cell.method]
            + ([mode] if mode != cell.method else [])
            + [str(cell.tier), str(cell.operator), f"seed{cell.seed}"]
            + str(getattr(args, "wandb_tags", "") or "").split(","))
    out = []
    for t in tags:
        t = t.strip().replace(",", "_")[:WANDB_TAG_MAX]
        if t and t not in out:
            out.append(t)
    return {"AQI_WANDB_TAGS": ",".join(out)}


# ===========================================================================
# arms: named method variants (base aqi_mode + per-arm overrides)
# ===========================================================================
# A campaign compares ARMS. An arm is a base --aqi_mode plus overrides of the
# learning-method knobs in ARM_KEYS. A plain mode name ("hilserl", "logged",
# ...) is an arm with no overrides, so every older config means exactly what
# it did before. Variants are declared in the YAML:
#
#   methods: [logged, quality-cf, quality-beh]       # arm names, report order
#   arms:
#     quality-cf:  {mode: quality, quality_score_mode: counterfactual}
#     quality-beh: {mode: quality, quality_score_mode: behavioral}
#
# A knob of an arm resolves as
#   arms[arm][key]  >  method_overrides[arm][key]  >  campaign-wide value
# (method_overrides is the older per-mode block; same meaning, still read).
# The campaign-wide value is itself CLI > YAML > argparse default.
#
# Only learning-method knobs vary per arm. Operators, demos, budget,
# convergence rule and ruler stay campaign-wide on purpose: the arm must be
# the ONLY difference between paired cells (same operator x seed).
#
# Arm names become cell-id prefixes "<arm>_<operator>_s<seed>" that analysis
# scripts parse back, so they may not contain "_" (use "-", e.g. quality-beh).
ARM_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9.-]*$")

# Arms whose `mode` is a separate training PIPELINE instead of an aqi_mode of
# train_aqi_rlpd.py. Their learner/actor commands are built by
# _pipeline_learner_cmd / _pipeline_actor_cmd (own flags; the AQI flags are
# never passed) and the ruler is told the agent type.
PIPELINES = {"hiril": "train_hiril.py"}   # external baseline, see
                                          # docs/hiril_baseline.md


def _to_bool(v):
    if isinstance(v, str):
        if v.strip().lower() in ("1", "true", "yes", "on"):
            return True
        if v.strip().lower() in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"not a boolean: {v!r}")
    return bool(v)


def _to_csv(v):
    """List or comma string -> comma string; an empty list means 'none'."""
    return _csv(v) or "none"


ARM_KEYS = {
    # learner optimisation / warm start
    "demo_pretrain": _to_bool, "demo_pretrain_steps": int,
    "demo_pretrain_batch_size": int, "learning_rate": float,
    # critic ensemble (inert for hilserl/e2hil)
    "critic_ensemble_size": int, "critic_subsample_size": int,
    "per_member_targets": _to_bool,
    # active gate (active/aqi)
    "trigger_terms": _to_csv, "trigger_combiner": str,
    "tau_A": float, "query_target_rate": float,
    "query_cooldown_steps": int, "query_min_on_steps": int,
    "random_request_rate": float, "bootstrap_p": float,
    "synthetic_request_or": _to_bool, "synthetic_request_hold_steps": int,
    "synthetic_handover_steps": int,
    # quality module (quality/aqi)
    "quality_filters": _to_csv, "quality_disagreement_gate": _to_bool,
    "bc_loss_weight": float, "quality_score_mode": str,
    "quality_behavioral_terms": _to_csv,
    "quality_post_release_steps": int, "quality_hindsight_discount": float,
    "quality_consistency_weight": _to_bool, "quality_consistency_k": int,
    "quality_consistency_sigma_ref": float,
    "quality_cf_diag": _to_bool, "quality_drop_fraction": float,
    "per_standardize": _to_bool, "per_delta_init": float, "per_eta": float,
    "per_eps": float, "quality_warmup_steps": int, "per_scale_decay": float,
    "per_refresh_period": int, "per_is_beta": float,
    # E2HiL external baseline (e2hil)
    "e2hil_K": int, "e2hil_q_low": float, "e2hil_q_high": float,
    "e2hil_q_agg": str,
    # HiRIL external baseline (mode: hiril) -- serl_launcher/hiril/config.py
    **{f"hiril_{name}": (_to_bool if cast is bool else cast)
       for name, cast in HIRIL_FIELDS.items()},
}


# per-arm keys that every pipeline understands (demo warm start); the rest
# are either train_aqi_rlpd knobs (incl. learning_rate) or hiril_* knobs
# (train_hiril.py, which has its own actor/critic learning rates)
_LEARNER_ARM_KEYS = {"demo_pretrain", "demo_pretrain_steps",
                     "demo_pretrain_batch_size"}


def _arm_def(args, arm: str) -> dict:
    """Raw definition {"mode": ..., <key>: <value>} of an arm. A name with no
    definition is a plain mode: {"mode": arm}."""
    d = dict((getattr(args, "method_overrides", None) or {}).get(arm) or {})
    d.update((getattr(args, "arms", None) or {}).get(arm) or {})
    d.setdefault("mode", arm)
    return d


def _arm_mode(args, arm: str) -> str:
    """The --aqi_mode an arm runs."""
    return str(_arm_def(args, arm)["mode"])


def _arm_overrides(args, arm: str) -> dict:
    """The arm's own overrides (cast, without 'mode'), for docs/manifests."""
    return {k: ARM_KEYS[k](v) for k, v in _arm_def(args, arm).items()
            if k != "mode"}


def _arm_val(args, method: str, key: str, default):
    """Value of ``key`` for arm ``method``: its override if it has one,
    else the campaign-wide ``default``."""
    d = _arm_def(args, method)
    if key in d and key != "mode":
        return ARM_KEYS[key](d[key]) if key in ARM_KEYS else d[key]
    return default


def validate_arms(args, methods):
    """Fail fast (before any training) on a bad arm: invalid name, unknown
    mode, unknown knob, or a value of the wrong type."""
    for arm in methods:
        if not ARM_NAME_RE.match(arm):
            raise SystemExit(
                f"[arms] invalid arm name {arm!r}: use lowercase letters, "
                f"digits, '-' or '.', no '_' (cell ids are "
                f"<arm>_<operator>_s<seed>).")
        d = _arm_def(args, arm)
        valid_modes = tuple(AQI_MODES) + tuple(PIPELINES)
        if d["mode"] not in valid_modes:
            raise SystemExit(
                f"[arms] '{arm}' is not a mode {valid_modes}; define it "
                f"under `arms:` with a `mode:` key.")
        if arm in valid_modes and d["mode"] != arm:
            raise SystemExit(
                f"[arms] arm '{arm}' is a mode name but maps to mode "
                f"'{d['mode']}'; give the variant its own name.")
        unknown = sorted(set(d) - set(ARM_KEYS) - {"mode"})
        if unknown:
            raise SystemExit(
                f"[arms] '{arm}': unknown knob(s) {unknown}; per-arm knobs "
                f"are {sorted(ARM_KEYS)} (operators, demos, budget and ruler "
                f"are campaign-wide by design).")
        for k, v in d.items():
            if k == "mode":
                continue
            try:
                ARM_KEYS[k](v)
            except (TypeError, ValueError) as exc:
                raise SystemExit(f"[arms] '{arm}': bad value for {k}: {exc}")
        if ("quality_score_mode" in d
                and str(d["quality_score_mode"]) not in QUALITY_SCORE_MODES):
            raise SystemExit(
                f"[arms] '{arm}': quality_score_mode "
                f"{d['quality_score_mode']!r} is not one of "
                f"{QUALITY_SCORE_MODES}.")
        is_pipeline = d["mode"] in PIPELINES
        if is_pipeline:     # the pipeline's own config validates the values
            from serl_launcher.hiril.config import HiRILConfig
            kw = {n: _arm_val(args, arm, f"hiril_{n}",
                              getattr(args, f"hiril_{n}", None))
                  for n in HIRIL_FIELDS}
            try:
                HiRILConfig(**{k: v for k, v in kw.items() if v is not None})
            except (AssertionError, TypeError, ValueError) as exc:
                raise SystemExit(f"[arms] '{arm}': invalid HiRIL setting "
                                 f"({exc or 'see serl_launcher/hiril/config.py'})")
        inert = sorted(k for k in d if k != "mode" and (
            k.startswith("hiril_") != is_pipeline)
            and k not in _LEARNER_ARM_KEYS)
        if inert:
            _yellow(f"[arms] '{arm}' (mode {d['mode']}): {inert} do not "
                    f"apply to this mode and are ignored.")
    for arm in (getattr(args, "arms", None) or {}):
        if arm not in methods:
            _yellow(f"[arms] '{arm}' is defined under `arms:` but not listed "
                    f"in `methods:` -> it will not run.")


def _pipeline_flags(args, arm):
    """--hiril_<name> <value> for every HiRIL knob set campaign-wide or on
    the arm (unset knobs keep HiRILConfig's paper defaults)."""
    f = []
    for name, cast in HIRIL_FIELDS.items():
        key = f"hiril_{name}"
        v = _arm_val(args, arm, key, getattr(args, key, None))
        if v is None:
            continue
        if cast is bool:            # absl boolean flag: --x / --nox
            f += [f"--{key}" if _to_bool(v) else f"--no{key}"]
        else:
            f += [f"--{key}", str(v)]
    return f


def _pipeline_learner_cmd(args, ckpt, demo_path, method, seed, run_name):
    """Learner command of a PIPELINES arm (train_hiril.py): the shared budget
    / warm-start flags + its own knobs. It has its own optimiser settings,
    so the campaign-wide learning_rate is NOT passed."""
    dp = bool(_arm_val(args, method, "demo_pretrain", args.demo_pretrain))
    dps = int(_arm_val(args, method, "demo_pretrain_steps",
                       args.demo_pretrain_steps))
    dpb = int(_arm_val(args, method, "demo_pretrain_batch_size",
                       args.demo_pretrain_batch_size))
    return [PIPELINES[_arm_mode(args, method)], "--exp_name", args.exp_name,
            "--learner", "--demo_path", str(demo_path),
            "--checkpoint_path", str(ckpt), "--resume", "scratch",
            "--peg_shape", args.peg_shape, "--seed", str(seed),
            "--max_steps", str(args.learner_max_steps),
            "--checkpoint_keep", str(getattr(args, "checkpoint_keep", 25)),
            ("--demo_pretrain" if dp else "--nodemo_pretrain"),
            "--demo_pretrain_steps", str(dps),
            "--demo_pretrain_batch_size", str(dpb),
            "--run_name", run_name, "--arm", method,
            ] + (["--utd_ratio", str(args.utd_ratio)]
                 if float(getattr(args, "utd_ratio", 0.0) or 0.0) > 0.0 else []
                 ) + _pipeline_flags(args, method) + _wandb_flags(args)


def _rpy_box(args):
    """The sim orientation box as three floats (roll, pitch, yaw half-widths,
    rad), or None when off. Validated in main()."""
    vals = _parse_floats(str(getattr(args, "sim_rpy_box", "") or "").replace(";", ","))
    return vals or None


def _sim_env(args):
    """Environment of every process of a cell (demos, learner, actor, ruler):
    $SIM_RPY_BOX is always set, empty when the box is off, so a value
    exported in the user's shell cannot leak into a campaign."""
    box = _rpy_box(args)
    return {"SIM_RPY_BOX": ",".join(f"{v:g}" for v in box) if box else ""}


def _platform_switches(args):
    """Campaign-wide switches (patches 0014/0015) that differ from their
    defaults. They are passed to train_aqi_rlpd.py only when set, so every
    older config keeps byte-identical commands."""
    on = []
    if getattr(args, "synthetic_rotation_control", False):
        on.append("synthetic_rotation_control")
    if getattr(args, "synthetic_orientation_monitor", False):
        on.append("synthetic_orientation_monitor")
    if getattr(args, "synthetic_inconsistent", False):
        on.append("synthetic_inconsistent")
    if str(getattr(args, "converge_mode", "rolling")) != "rolling":
        on.append("converge_mode")
    if float(getattr(args, "utd_ratio", 0.0) or 0.0) > 0.0:
        on.append("utd_ratio")
    if getattr(args, "log_takeover_channels", False):
        on.append("log_takeover_channels")
    return on


def _platform_actor_flags(args):
    out = []
    if getattr(args, "synthetic_rotation_control", False):
        out += ["--synthetic_rotation_control"]
    if getattr(args, "synthetic_orientation_monitor", False):
        out += ["--synthetic_orientation_monitor"]
        for k in ("synthetic_orientation_tilt_rad", "synthetic_orientation_yaw_rad"):
            if float(getattr(args, k, 0.0) or 0.0) > 0.0:
                out += [f"--{k}", str(getattr(args, k))]
    if getattr(args, "synthetic_inconsistent", False):
        out += ["--synthetic_inconsistent",
                "--synthetic_bad_share", str(args.synthetic_bad_share),
                "--synthetic_bad_persistence", str(args.synthetic_bad_persistence)]
    if str(getattr(args, "converge_mode", "rolling")) != "rolling":
        out += ["--converge_mode", str(args.converge_mode),
                "--converge_eval_every", str(args.converge_eval_every),
                "--converge_eval_consecutive", str(args.converge_eval_consecutive)]
    if float(getattr(args, "utd_ratio", 0.0) or 0.0) > 0.0:
        out += ["--utd_ratio", str(args.utd_ratio)]
    if getattr(args, "log_takeover_channels", False):
        out += ["--log_takeover_channels"]
    return out


def _pipeline_actor_cmd(args, ckpt, method, tier, operator, theta_json, seed,
                        run_name):
    """Actor command of a PIPELINES arm: the same operator / budget /
    convergence flags as the AQI actor, without --aqi_mode and AQI knobs."""
    if "synthetic_inconsistent" in _platform_switches(args):
        raise SystemExit(
            f"[arms] '{method}': synthetic_inconsistent is not wired into the "
            f"{PIPELINES[_arm_mode(args, method)]} pipeline; the arms would "
            f"differ in more than the method. Run that arm without it.")
    actor_steps, actor_minutes = _actor_budget(args)
    cmd = [PIPELINES[_arm_mode(args, method)], "--exp_name", args.exp_name,
           "--actor", "--checkpoint_path", str(ckpt),
           "--peg_shape", args.peg_shape, "--seed", str(seed),
           "--max_steps", str(actor_steps),
           "--max_minutes", str(actor_minutes),
           "--buffer_period", str(getattr(args, "buffer_period", 0)),
           "--converge_success", str(args.converge_success),
           "--converge_window", str(args.converge_window),
           "--converge_min_steps", str(args.converge_min_steps),
           "--converge_eval_threshold", str(args.converge_eval_threshold),
           "--converge_eval_max_steps", str(args.converge_eval_max_steps),
           "--converge_eval_poses", str(args.converge_eval_poses),
           ("--converge_require_autonomous"
            if args.converge_require_autonomous
            else "--noconverge_require_autonomous"),
           "--operator_id", operator,
           "--synthetic_operator", tier,
           "--synthetic_theta", str(theta_json),
           "--synthetic_seed", str(seed),
           "--synthetic_intervention_mode",
           str(args.synthetic_intervention_mode),
           "--env_seed", str(seed),
           "--run_name", run_name,
           "--aqi_log_dir", str((ckpt.parent.parent / "aqi_logs").resolve()),
           "--arm", method] + _platform_actor_flags(args) + _pipeline_flags(args, method)
    if args.policy_oracle_checkpoint:
        cmd += ["--oracle_checkpoint", str(args.policy_oracle_checkpoint)]
    if getattr(args, "training_oracle", "auto") != "auto":
        cmd += ["--training_oracle", args.training_oracle]
    return cmd


def _learner_cmd(args, ckpt: Path, demo_path: Path, method: str, seed: int,
                 run_name: str):
    if _arm_mode(args, method) in PIPELINES:
        return _pipeline_learner_cmd(args, ckpt, demo_path, method, seed,
                                     run_name)
    dp = bool(_arm_val(args, method, "demo_pretrain", args.demo_pretrain))
    dps = int(_arm_val(args, method, "demo_pretrain_steps",
                       args.demo_pretrain_steps))
    dpb = int(_arm_val(args, method, "demo_pretrain_batch_size",
                       args.demo_pretrain_batch_size))
    lr = float(_arm_val(args, method, "learning_rate",
                        getattr(args, "learning_rate", 0) or 0))
    return ["train_aqi_rlpd.py", "--exp_name", args.exp_name, "--learner",
            "--demo_path", str(demo_path), "--checkpoint_path", str(ckpt),
            "--resume", "scratch", "--aqi_mode", _arm_mode(args, method),
            "--peg_shape", args.peg_shape, "--seed", str(seed),
            "--max_steps", str(args.learner_max_steps),
            "--checkpoint_keep", str(getattr(args, "checkpoint_keep", 25)),
        ("--demo_pretrain" if dp else "--nodemo_pretrain"),
        "--demo_pretrain_steps", str(dps),
        "--demo_pretrain_batch_size", str(dpb),
            "--run_name", run_name] + (
        ["--utd_ratio", str(args.utd_ratio)]
        if float(getattr(args, "utd_ratio", 0.0) or 0.0) > 0.0 else []) + (
        ["--learning_rate", str(lr)] if lr and lr > 0 else []
    ) + _arm_flag(args, method) + _aqi_flags(args, method) + _wandb_flags(args)


def _arm_flag(args, arm):
    """--arm <name> for a named variant (recorded in meta.json / wandb so the
    analysis scripts group by arm); omitted for a plain mode, whose arm name
    IS its mode."""
    return ["--arm", arm] if _arm_mode(args, arm) != arm else []


def _actor_cmd(args, ckpt: Path, method: str, tier: str, operator: str,
               theta_json: Path, seed: int, run_name: str):
    if _arm_mode(args, method) in PIPELINES:
        return _pipeline_actor_cmd(args, ckpt, method, tier, operator,
                                   theta_json, seed, run_name)
    actor_steps, actor_minutes = _actor_budget(args)
    cmd = ["train_aqi_rlpd.py", "--exp_name", args.exp_name, "--actor",
            "--checkpoint_path", str(ckpt), "--aqi_mode",
            _arm_mode(args, method),
            "--peg_shape", args.peg_shape, "--seed", str(seed),
        "--max_steps", str(actor_steps),
        "--max_minutes", str(actor_minutes),
        "--buffer_period", str(getattr(args, "buffer_period", 0)),
            "--converge_success", str(args.converge_success),
            "--converge_window", str(args.converge_window),
            "--converge_min_steps", str(args.converge_min_steps),
            "--converge_eval_threshold", str(args.converge_eval_threshold),
            "--converge_eval_max_steps", str(args.converge_eval_max_steps),
            "--converge_eval_poses", str(args.converge_eval_poses),
            ("--converge_require_autonomous"
             if args.converge_require_autonomous
             else "--noconverge_require_autonomous"),
            "--operator_id", operator,
            "--synthetic_operator", tier,          # non-empty -> synthetic branch
            "--synthetic_theta", str(theta_json),  # the EXACT jittered operator
            "--synthetic_seed", str(seed),
            "--synthetic_intervention_mode",
            str(args.synthetic_intervention_mode),
            "--env_seed", str(seed),               # independent resets/replicate
            "--run_name", run_name,
            "--aqi_log_dir", str((ckpt.parent.parent / "aqi_logs").resolve()),
            ] + _arm_flag(args, method) + _aqi_flags(args, method)
    cmd += _platform_actor_flags(args)
    if args.policy_oracle_checkpoint:
        cmd += ["--oracle_checkpoint", str(args.policy_oracle_checkpoint)]
    if getattr(args, "training_oracle", "auto") != "auto":
        cmd += ["--training_oracle", args.training_oracle]
    return cmd


def _ready_timeout(args, arm):
    """Seconds to wait for the learner's LEARNER READY banner. Demo
    pretraining runs BEFORE the banner, so it is budgeted at a conservative
    >= 2.5 demo-pretrain it/s -- per ARM, since an arm may opt in or out."""
    timeout = float(args.ready_timeout)
    dp = bool(_arm_val(args, arm, "demo_pretrain", args.demo_pretrain))
    dps = int(_arm_val(args, arm, "demo_pretrain_steps",
                       args.demo_pretrain_steps))
    if dp and dps > 0:
        timeout += dps / 2.5
    return timeout


def train_cell(args, campaign: Path, cell: Cell):
    ckpt = campaign / "checkpoints" / cell.id
    done = ckpt / ".done"
    if done.exists() and args.skip_trained:
        _green(f"[train] {cell.id} already trained (.done) -- skip")
        return True
    demo_path = _demo_pkl(campaign, cell.operator, cell.seed)
    if not demo_path.exists() and not args.dry_run:
        _red(f"[train] missing demos {demo_path} for {cell.id}; run the "
             f"demos stage first.")
        return False
    theta_json = campaign / "operators" / f"{cell.operator}.json"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "STOP").unlink(missing_ok=True)   # clear any stale sentinel

    lcmd = _learner_cmd(args, ckpt, demo_path, cell.method, cell.seed,
                        f"{cell.id}_learner")
    acmd = _actor_cmd(args, ckpt, cell.method, cell.tier, cell.operator,
                      theta_json, cell.seed, f"{cell.id}_actor")
    if args.dry_run:
        _yellow("[train] DRY learner: python " + " ".join(lcmd))
        _yellow("[train] DRY actor:   python " + " ".join(acmd))
        return True

    llog = campaign / "logs" / f"{cell.id}.learner.log"
    alog = campaign / "logs" / f"{cell.id}.actor.log"
    _green(f"[train] {cell.id}: starting learner ...")
    lp, lf = _launch(lcmd, llog, background=True, extra_env={
        "JAX_PLATFORMS": args.jax_platform,
        "XLA_PYTHON_CLIENT_MEM_FRACTION": args.gpu_mem_learner,
        **_wandb_env(args, campaign), **_wandb_tag_env(args, campaign, cell),
        **_sim_env(args)})
    try:
        # Demo pretraining runs BEFORE the learner prints "LEARNER READY", so
        # the ready wait must budget for it (a crash still returns early via
        # proc.poll()). Assume a conservative >=2.5 demo-pretrain it/s.
        ready_timeout = _ready_timeout(args, cell.method)
        if not _wait_for_banner(lp, llog, "LEARNER READY", ready_timeout):
            _red(f"[train] {cell.id}: learner never reported READY "
                 f"(timeout {ready_timeout:g}s or crash); see {llog.name}.")
            _terminate(lp)
            lf.close()
            return False
        actor_steps, actor_minutes = _actor_budget(args)
        if args.cell_budget_mode == "time":
            _green(f"[train] {cell.id}: learner READY -> starting actor "
                   f"(budget {actor_minutes:g} min wall-clock) ...")
        else:
            _green(f"[train] {cell.id}: learner READY -> starting actor "
                   f"(budget {actor_steps} env steps) ...")
        ap, af = _launch(acmd, alog, background=True, extra_env={
            "JAX_PLATFORMS": args.jax_platform,
            "XLA_PYTHON_CLIENT_MEM_FRACTION": args.gpu_mem_actor,
            **_sim_env(args)})
        # Wall-clock safety net: the actor enforces --max_minutes itself, but if
        # it ever hangs, terminate it a few minutes past its budget so the
        # campaign can never stall on one cell.
        # steps mode has no wall-clock budget of its own; fall back to
        # --minutes_per_cell as a pure anti-hang SAFETY net (the actor normally
        # exits cleanly at its env-step cap well before this).
        actor_timeout = (actor_minutes * 60.0 + 300.0 if actor_minutes > 0
                         else (args.minutes_per_cell * 60.0 + 300.0
                               if getattr(args, "minutes_per_cell", 0)
                               and args.minutes_per_cell > 0 else None))
        try:
            arc = ap.wait(timeout=actor_timeout)
        except subprocess.TimeoutExpired:
            _yellow(f"[train] {cell.id}: actor exceeded its time budget + grace "
                    f"-> terminating.")
            arc = _terminate(ap)
        af.close()
        if arc != 0:
            _yellow(f"[train] {cell.id}: actor exited rc={arc} "
                    f"(see {alog.name}); stopping learner anyway.")
        # graceful learner shutdown: sentinel -> finish step, save, exit
        _green(f"[train] {cell.id}: actor done -> stopping learner (STOP).")
        ckpt.mkdir(parents=True, exist_ok=True)   # scratch may have wiped it
        (ckpt / "STOP").touch()
        try:
            lrc = lp.wait(timeout=args.learner_stop_timeout)
        except subprocess.TimeoutExpired:
            _yellow(f"[train] {cell.id}: learner did not stop in "
                    f"{args.learner_stop_timeout}s -> terminating.")
            lrc = _terminate(lp)
    finally:
        lf.close()
        (ckpt / "STOP").unlink(missing_ok=True)

    if not _has_checkpoint(ckpt):
        _red(f"[train] {cell.id}: no checkpoint written; treating as FAILED.")
        return False
    conv = None
    cj = ckpt / "CONVERGED.json"
    if cj.exists():
        try:
            conv = json.load(open(cj))
        except (ValueError, OSError):
            conv = None
    done.write_text(json.dumps({
        "cell": cell.id, "method": cell.method, "operator": cell.operator,
        "tier": cell.tier, "seed": cell.seed,
        "actor_rc": arc, "learner_rc": lrc,
        "converged": conv is not None,
        "converged_step": (conv or {}).get("step"),
        "convergence": conv,
        "finished": datetime.datetime.now().isoformat(timespec="seconds")},
        indent=2))
    _document_run(args, campaign, cell, conv, cmds=(lcmd, acmd))
    if conv is not None:
        _green(f"[train] {cell.id}: done (converged at step {conv['step']}).")
    else:
        if args.cell_budget_mode == "time":
            _yellow(f"[train] {cell.id}: done (did not converge within "
                    f"{args.minutes_per_cell:g} min).")
        else:
            _yellow(f"[train] {cell.id}: done (hit the {args.steps}-step cap "
                    f"without converging).")
    return True


def _document_run(args, campaign: Path, cell, conv, cmds=None):
    """Write README.md + manifest.json next to a trained cell's checkpoints.
    ``cmds`` = (learner_cmd, actor_cmd) exactly as launched; the reproduce
    block then replays them verbatim (every per-arm knob included)."""
    ckpt = campaign / "checkpoints" / cell.id
    th = _roster_theta(campaign, cell.operator)
    term = {"budget_mode": args.cell_budget_mode}
    if args.cell_budget_mode == "steps":
        term["step_budget"] = args.steps
    else:
        term["time_budget_min"] = args.minutes_per_cell
    if conv is not None:
        term.update(converged=True, step=conv.get("step"),
                    rolling_success=conv.get("rolling_success"),
                    window=conv.get("window"), autonomous=conv.get("autonomous"))
        if conv.get("mode") == "eval":          # --converge_mode eval
            term.update(mode="eval",
                        mini_eval_threshold=conv.get("mini_eval_threshold"),
                        consecutive=conv.get("consecutive"))
    else:
        term["converged"] = False
    if cmds is None:
        cmds = (_learner_cmd(args, ckpt, _demo_pkl(campaign, cell.operator,
                                                   cell.seed),
                             cell.method, cell.seed, f"{cell.id}_learner"),
                _actor_cmd(args, ckpt, cell.method, cell.tier, cell.operator,
                           campaign / "operators" / f"{cell.operator}.json",
                           cell.seed, f"{cell.id}_actor"))
    env = "cd examples && source experiments/sim_peg_insertion/sim_env.sh && "
    reproduce = [
        f"# 1) learner (leave running)\n{env}python "
        + " ".join(shlex.quote(str(a)) for a in cmds[0]),
        f"# 2) actor (synthetic operator {cell.operator}), once the learner "
        f"prints LEARNER READY\n{env}python "
        + " ".join(shlex.quote(str(a)) for a in cmds[1])]
    prov.document(
        ckpt, "run", quiet=True, task=_task_dict(args), method=cell.method,
        operator=_operator_dict(th, campaign) if th else {"operator_id": cell.operator},
        seeds={"seed": cell.seed, "synthetic_seed": cell.seed,
               "env_seed": cell.seed},
        demos={"n": args.demos, "filtered": args.filtered,
               "path": f"../../demos/{cell.operator}_s{cell.seed}/demos.pkl"},
        termination=term, conditions=_method_conditions(cell.method, args),
        reproduce=reproduce,
        extra={"arm": {"name": cell.method,
                       "aqi_mode": _arm_mode(args, cell.method),
                       "overrides": _arm_overrides(args, cell.method)}})


# ===========================================================================
# stage 4: eval one cell against the frozen ruler
# ===========================================================================
def eval_cell(args, campaign: Path, cell: Cell):
    out = campaign / "eval_runs" / cell.id
    if (out / "cell.json").exists():
        _green(f"[eval] {cell.id} already evaluated -- skip")
        return True
    ckpt = campaign / "checkpoints" / cell.id
    if not _has_checkpoint(ckpt) and not args.dry_run:
        _red(f"[eval] {cell.id}: no checkpoint to evaluate -- skip")
        return False
    pyargs = ["eval_ruler.py", "--exp_name", args.exp_name,
              "--peg_shape", args.peg_shape,
              "--checkpoint_path", str(ckpt),
              "--grid_file", str(campaign / "eval_grid.json"),
              "--out", str(out), "--cell_id", cell.id,
              "--argmax", "--max_steps", str(args.eval_max_steps),
              "--eval_last_k", str(args.eval_last_k),
              "--select", args.eval_select,
              "--select_window_k", str(args.eval_window_k),
              "--no_realtime"]
    if _arm_mode(args, cell.method) in PIPELINES:
        pyargs += ["--agent_type", _arm_mode(args, cell.method)]
    if args.eval_select in ("best", "mean", "best_window"):
        # these metrics score ALL checkpoints; don't let the certified ckpt
        # (saved late at convergence) pre-empt the sweep.
        pyargs.append("--noprefer_certified")
    if args.dry_run:
        _yellow("[eval] DRY: python " + " ".join(pyargs))
        return True
    _green(f"[eval] {cell.id}: scoring on the fixed ruler ...")
    rc = _launch(pyargs, campaign / "logs" / f"{cell.id}.eval.log",
                 extra_env={"JAX_PLATFORMS": args.jax_platform,
                            "XLA_PYTHON_CLIENT_MEM_FRACTION": args.gpu_mem_actor,
                            "SIM_DISABLE_SPACEMOUSE": "1",
                            **_sim_env(args)})
    if rc != 0 or not (out / "cell.json").exists():
        _red(f"[eval] {cell.id}: FAILED (rc={rc}); see the log.")
        return False
    _document_eval(args, campaign, cell)
    return True


def _document_eval(args, campaign: Path, cell):
    """Write README.md + manifest.json next to a cell's ruler outcomes."""
    out = campaign / "eval_runs" / cell.id
    cj = out / "cell.json"
    ev = {"grid_sha256": None, "argmax": True}
    if cj.exists():
        try:
            c = json.load(open(cj))
            ev = {"grid_sha256": c.get("grid_sha256"),
                  "n_trials": c.get("n_trials_spec"),
                  "argmax": c.get("argmax", True),
                  "checkpoint_step": c.get("checkpoint_step"),
                  "f_max": c.get("f_max"), "outcomes": c.get("outcomes")}
        except (ValueError, OSError):
            pass
    th = _roster_theta(campaign, cell.operator)
    prov.document(
        out, "eval", quiet=True, task=_task_dict(args), method=cell.method,
        operator=_operator_dict(th, campaign) if th else {"operator_id": cell.operator},
        seeds={"seed": cell.seed}, evaluation=ev,
        reproduce=[f"./.venv/bin/python examples/eval_ruler.py --exp_name "
                   f"{args.exp_name} --checkpoint_path checkpoints/{cell.id} "
                   f"--grid_file eval_grid.json --out eval_runs/{cell.id} "
                   f"--cell_id {cell.id} --argmax"])



# ===========================================================================
# stage 5: analyse operator dependency (the thesis core)
# ===========================================================================
def _read_cell_sr(campaign: Path, cell: Cell):
    cj = campaign / "eval_runs" / cell.id / "cell.json"
    if not cj.exists():
        return None
    try:
        sr = json.load(open(cj))["outcomes"]["sr"]
    except (KeyError, ValueError, OSError):
        return None
    if sr is None or (isinstance(sr, float) and sr != sr):   # None / NaN
        return None
    return float(sr)


def compute_dependency(sr_by_method, methods, tiers, baseline):
    """Core analysis, pure so it is unit-testable on fabricated SRs.

    ``sr_by_method`` = {method: {operator: {"tier": t, "sr": {seed: sr}}}}.
    For each method: groups = per-operator lists of per-seed SR ->
    icc_oneway gives sigma2_between (operators) and sigma2_within (seeds); we
    add the tier means (skill gradient) and, versus ``baseline``, the reduction
    in ICC / between-operator variance that quantifies AQI's promise.
    """
    import numpy as np
    report = {"methods": {}, "baseline": baseline}
    for method in methods:
        ops = sr_by_method.get(method, {})
        groups = {op: [v for v in d["sr"].values()] for op, d in ops.items()
                  if d["sr"]}
        icc = S.icc_oneway(groups)
        try:
            boot = S.bootstrap_icc(groups)
            icc_ci = boot.get("icc_ci95")
        except Exception:
            icc_ci = None
        flat = [v for g in groups.values() for v in g]
        # B.3 operator spread (sigma_op, CV, range, worst-case) over the
        # per-operator MEAN SR, and C.2 permutation test "operator matters"
        # (does the between-operator variance exceed chance?) -- both computed
        # WITHIN this method, alongside the ICC decomposition.
        op_means = {op: float(np.mean(list(d["sr"].values())))
                    for op, d in ops.items() if d["sr"]}
        spread = S.operator_spread(op_means) if op_means else {}
        perm_vals, perm_labs = [], []
        for op, d in ops.items():
            for v in d["sr"].values():
                perm_vals.append(v)
                perm_labs.append(op)
        try:
            perm = (S.permutation_sigma2(perm_vals, perm_labs)
                    if len(set(perm_labs)) >= 2 else {})
        except Exception:
            perm = {}
        tier_means = {}
        for t in tiers:
            vals = [v for op, d in ops.items() if d["tier"] == t
                    for v in d["sr"].values()]
            tier_means[t] = float(np.mean(vals)) if vals else float("nan")
        report["methods"][method] = {
            "n_operators": len(groups),
            "n_cells": len(flat),
            "mean_sr": float(np.mean(flat)) if flat else float("nan"),
            "sd_sr": float(np.std(flat, ddof=1)) if len(flat) > 1
            else float("nan"),
            "sigma2_between_operator": icc["sigma2_op"],
            "sigma2_within_operator": icc["sigma2_seed"],
            "icc": icc["icc"],
            "icc_ci95": icc_ci,
            "sigma_op": spread.get("sigma_op"),
            "cv_operator": spread.get("cv"),
            "range_operator": spread.get("range"),
            "worst_operator": spread.get("worst_operator"),
            "worst_operator_sr": spread.get("worst"),
            "best_operator_sr": spread.get("best"),
            "operator_matters_p": perm.get("p_value"),
            "tier_mean_sr": tier_means,
        }
    base = report["methods"].get(baseline)
    if base:
        for method, m in report["methods"].items():
            m["delta_icc_vs_baseline"] = (
                base["icc"] - m["icc"]
                if base["icc"] == base["icc"] and m["icc"] == m["icc"]
                else float("nan"))
            m["delta_sigma2_between_vs_baseline"] = (
                base["sigma2_between_operator"] - m["sigma2_between_operator"])
    return report


def _sign_test_p(n_pos, n_neg):
    """Two-sided exact-binomial sign test over the non-tied pairs."""
    import math
    m = n_pos + n_neg
    if m == 0:
        return float("nan")
    k = min(n_pos, n_neg)
    tail = sum(math.comb(m, i) for i in range(0, k + 1)) * (0.5 ** m)
    return min(1.0, 2.0 * tail)


def compute_paired(sr_by_method, methods, baseline):
    """Paired per-cell comparison: since every method shares the SAME operator
    and seed grid, SR_method - SR_baseline for each matched (operator, seed) is
    a paired difference that cancels operator/seed variance. Reports the deltas
    and a two-sided exact-binomial SIGN TEST over the non-tied pairs."""
    import numpy as np
    out = {}
    base = sr_by_method.get(baseline, {})
    for method in methods:
        if method == baseline:
            continue
        mm = sr_by_method.get(method, {})
        pairs = []
        for op, d in mm.items():
            bsr = base.get(op, {}).get("sr", {})
            tier = d.get("tier")
            for s, sr in d.get("sr", {}).items():
                b = bsr.get(s)
                if b is None or sr is None:
                    continue
                pairs.append(dict(operator=op, tier=tier, seed=s,
                                  baseline_sr=float(b), method_sr=float(sr),
                                  delta=float(sr) - float(b)))
        deltas = [p["delta"] for p in pairs]
        n_pos = sum(1 for x in deltas if x > 0)
        n_neg = sum(1 for x in deltas if x < 0)
        n_tie = sum(1 for x in deltas if x == 0)
        sign_p = _sign_test_p(n_pos, n_neg)
        out[method] = dict(
            baseline=baseline, n_pairs=len(pairs),
            n_method_wins=n_pos, n_baseline_wins=n_neg, n_tied=n_tie,
            mean_delta=float(np.mean(deltas)) if deltas else float("nan"),
            median_delta=float(np.median(deltas)) if deltas else float("nan"),
            sign_test_p=sign_p, pairs=pairs)
    return out


def compute_convergence(steps_by_method, methods, tiers):
    """Complementary EFFORT dependency: how many env steps each cell needed to
    converge. ``steps_by_method`` = {method: {operator: {"tier", "steps":
    {seed: step_or_None}}}} where a None step means the cell trained but never
    converged (hit the cap). Per method we report the converged fraction, the
    mean/median steps-to-convergence, and the ICC of steps-to-convergence
    across operators (does *how long* training takes depend on who trained it?).
    """
    import numpy as np
    out = {}
    for method in methods:
        ops = steps_by_method.get(method, {})
        groups = {op: [s for s in d["steps"].values() if s is not None]
                  for op, d in ops.items()}
        groups = {op: v for op, v in groups.items() if v}
        all_steps = [s for d in ops.values() for s in d["steps"].values()
                     if s is not None]
        n_cells = sum(len(d["steps"]) for d in ops.values())
        n_conv = len(all_steps)
        icc = S.icc_oneway(groups)
        tier_steps = {}
        for t in tiers:
            vals = [s for op, d in ops.items() if d["tier"] == t
                    for s in d["steps"].values() if s is not None]
            tier_steps[t] = float(np.mean(vals)) if vals else float("nan")
        out[method] = {
            "n_cells": n_cells, "n_converged": n_conv,
            "converged_fraction": (n_conv / n_cells) if n_cells
            else float("nan"),
            "steps_mean": float(np.mean(all_steps)) if all_steps
            else float("nan"),
            "steps_median": float(np.median(all_steps)) if all_steps
            else float("nan"),
            "steps_sd": float(np.std(all_steps, ddof=1)) if len(all_steps) > 1
            else float("nan"),
            "sigma2_between_operator": icc["sigma2_op"],
            "sigma2_within_operator": icc["sigma2_seed"],
            "icc": icc["icc"],
            "tier_mean_steps": tier_steps,
        }
    return out


def _read_cell_interventions(campaign: Path, cell: Cell):
    """Human effort of one trained cell from the actor's per-episode takeover
    log (checkpoints/<cell>/episodes.jsonl, written in every mode): operator-
    controlled env steps / all env steps run during training. None if the
    log is absent (e.g. a cell trained before the log existed)."""
    path = campaign / "checkpoints" / cell.id / "episodes.jsonl"
    if not path.exists():
        return None
    n_int = n_steps = n_ep = n_ep_int = 0
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            k = int(rec.get("intervention_steps", 0))
            n_int += k
            n_steps += int(rec.get("length", 0))
            n_ep += 1
            n_ep_int += int(k > 0)
    if n_steps <= 0:
        return None
    return {"intervention_steps": n_int, "env_steps": n_steps,
            "episodes": n_ep, "episodes_with_takeover": n_ep_int,
            "rate": n_int / n_steps}


def compute_interventions(ivn_by_method, methods, tiers, baseline):
    """Human-effort comparison (the intervention-rate metric E2HiL reports).
    ``ivn_by_method`` = {method: {operator: {"tier", "ivn": {seed: dict}}}}
    with dicts from _read_cell_interventions. Per method: mean/median
    intervention rate over cells, mean operator steps per cell, per-tier
    rates; versus ``baseline``: paired per-(operator, seed) rate deltas with
    a sign test (negative delta = the method needed LESS human effort)."""
    import numpy as np
    out = {"methods": {}, "paired": {}, "baseline": baseline}
    for method in methods:
        ops = ivn_by_method.get(method, {})
        cells = [v for d in ops.values() for v in d["ivn"].values()]
        rates = [c["rate"] for c in cells]
        tier_rate = {}
        for t in tiers:
            vals = [v["rate"] for d in ops.values() if d["tier"] == t
                    for v in d["ivn"].values()]
            tier_rate[t] = float(np.mean(vals)) if vals else float("nan")
        out["methods"][method] = {
            "n_cells": len(cells),
            "rate_mean": float(np.mean(rates)) if rates else float("nan"),
            "rate_median": float(np.median(rates)) if rates else float("nan"),
            "intervention_steps_mean": (
                float(np.mean([c["intervention_steps"] for c in cells]))
                if cells else float("nan")),
            "env_steps_mean": (float(np.mean([c["env_steps"] for c in cells]))
                               if cells else float("nan")),
            "tier_rate": tier_rate,
        }
    base = ivn_by_method.get(baseline, {})
    for method in methods:
        if method == baseline:
            continue
        pairs = []
        for op, d in ivn_by_method.get(method, {}).items():
            b_ivn = base.get(op, {}).get("ivn", {})
            for s, v in d["ivn"].items():
                if s in b_ivn:
                    pairs.append(dict(operator=op, tier=d["tier"], seed=s,
                                      baseline_rate=b_ivn[s]["rate"],
                                      method_rate=v["rate"],
                                      delta=v["rate"] - b_ivn[s]["rate"]))
        deltas = [p["delta"] for p in pairs]
        n_lower = sum(1 for x in deltas if x < 0)
        n_higher = sum(1 for x in deltas if x > 0)
        out["paired"][method] = dict(
            baseline=baseline, n_pairs=len(pairs),
            n_method_lower=n_lower, n_baseline_lower=n_higher,
            n_tied=len(deltas) - n_lower - n_higher,
            mean_delta=float(np.mean(deltas)) if deltas else float("nan"),
            median_delta=float(np.median(deltas)) if deltas else float("nan"),
            sign_test_p=_sign_test_p(n_lower, n_higher), pairs=pairs)
    return out


def _fmt(x, pct=False):
    if x is None or (isinstance(x, float) and x != x):
        return "  n/a"
    return f"{100 * x:5.1f}%" if pct else f"{x:6.3f}"


def render_report_md(report, campaign_meta, tiers, conv=None, paired=None,
                     ivn=None):
    L = []
    L.append("# Operator-dependency report\n")
    L.append(f"- campaign: `{campaign_meta['campaign_dir']}`")
    L.append(f"- task: `{campaign_meta['exp_name']}` "
             f"(peg `{campaign_meta['peg_shape']}`)")
    L.append(f"- grid sha256: `{campaign_meta['grid_sha256'][:16]}...` "
             f"({campaign_meta['n_grid_trials']} trials)")
    opt = campaign_meta["ops_per_tier"]
    ops_txt = (", ".join(f"{t} {n}" for t, n in opt.items())
               if isinstance(opt, dict) else f"{opt}/tier x {len(tiers)} tiers")
    L.append(f"- operators: {campaign_meta['n_operators']} ({ops_txt}), "
             f"seeds/op: {campaign_meta['n_seeds']}, "
             f"cap: {campaign_meta['steps']} env steps/cell\n")
    L.append("## Operator dependency per method (final autonomous SR)\n")
    L.append("Lower ICC and lower between-operator variance = the outcome "
             "depends LESS on who trained it. The thesis claim is that AQI "
             "reduces these versus baseline HIL-SERL. ICC drop = baseline ICC "
             "minus method ICC (> 0: less operator-dependent than the "
             "baseline).\n")
    L.append("| method | mean SR | ICC | sigma2_between | sigma2_within | "
             "ICC drop vs base | operators | cells |")
    L.append("|---|---|---|---|---|---|---|---|")
    for method, m in report["methods"].items():
        tag = " (baseline)" if method == report["baseline"] else ""
        L.append(f"| `{method}`{tag} | {_fmt(m['mean_sr'], pct=True)} | "
                 f"{_fmt(m['icc'])} | {_fmt(m['sigma2_between_operator'])} | "
                 f"{_fmt(m['sigma2_within_operator'])} | "
                 f"{_fmt(m.get('delta_icc_vs_baseline'))} | "
                 f"{m['n_operators']} | {m['n_cells']} |")
    L.append("\n## Operator spread + 'operator matters' test (B.3 / C.2)\n")
    L.append("sigma_op / CV / range are over the per-operator MEAN SR within "
             "each method; p is the permutation test that the between-operator "
             "variance exceeds chance. High CV / low p = operators matter "
             "(strong dependency); AQI should flatten CV and raise p.\n")
    L.append("| method | sigma_op | CV | range | worst operator | "
             "p(operator matters) |")
    L.append("|---|---|---|---|---|---|")
    for method, m in report["methods"].items():
        wo = m.get("worst_operator") or "-"
        L.append(f"| `{method}` | {_fmt(m.get('sigma_op'))} | "
                 f"{_fmt(m.get('cv_operator'))} | "
                 f"{_fmt(m.get('range_operator'))} | "
                 f"`{wo}` ({_fmt(m.get('worst_operator_sr'), pct=True)}) | "
                 f"{_fmt(m.get('operator_matters_p'))} |")
    L.append("\n## Skill gradient (mean SR per tier)\n")
    L.append("| method | " + " | ".join(tiers) + " |")
    L.append("|---|" + "|".join(["---"] * len(tiers)) + "|")
    for method, m in report["methods"].items():
        row = " | ".join(_fmt(m["tier_mean_sr"].get(t), pct=True) for t in tiers)
        L.append(f"| `{method}` | {row} |")
    if conv:
        L.append("\n## Training effort: steps-to-convergence\n")
        L.append("How many env steps each cell needed to converge (the actor "
                 "stops on a sustained autonomous-success streak). Fewer steps "
                 "AND a lower steps-ICC = training cost depends less on who "
                 "trained it.\n")
        L.append("| method | converged | mean steps | median | sigma2_between "
                 "| sigma2_within | steps-ICC |")
        L.append("|---|---|---|---|---|---|---|")
        for method, c in conv.items():
            L.append(f"| `{method}` | {c['n_converged']}/{c['n_cells']} "
                     f"({_fmt(c['converged_fraction'], pct=True)}) | "
                     f"{c['steps_mean']:.0f} | {c['steps_median']:.0f} | "
                     f"{_fmt(c['sigma2_between_operator'])} | "
                     f"{_fmt(c['sigma2_within_operator'])} | {_fmt(c['icc'])} |")
        L.append("\n### Steps-to-convergence per tier\n")
        L.append("| method | " + " | ".join(tiers) + " |")
        L.append("|---|" + "|".join(["---"] * len(tiers)) + "|")
        for method, c in conv.items():
            row = " | ".join(
                (f"{c['tier_mean_steps'].get(t):.0f}"
                 if c['tier_mean_steps'].get(t) == c['tier_mean_steps'].get(t)
                 else "  n/a") for t in tiers)
            L.append(f"| `{method}` | {row} |")
    base = report["baseline"]
    if paired:
        L.append("\n## Paired per-cell comparison (matched operator x seed)\n")
        L.append("Every method shares the SAME operator and seed grid, so "
                 f"`SR_method - SR_{base}` per cell is a PAIRED difference that "
                 "cancels operator/seed variance (more powerful than comparing "
                 "group means at this small n). The sign test is the two-sided "
                 "exact-binomial probability of this many wins by chance.\n")
        L.append(f"| method vs `{base}` | pairs | method wins | {base} wins | "
                 "ties | mean d | median d | sign-test p |")
        L.append("|---|---|---|---|---|---|---|---|")
        for method, pr in paired.items():
            L.append(
                f"| `{method}` | {pr['n_pairs']} | {pr['n_method_wins']} | "
                f"{pr['n_baseline_wins']} | {pr['n_tied']} | "
                f"{pr['mean_delta']*100:+.1f} | {pr['median_delta']*100:+.1f} | "
                f"{_fmt(pr['sign_test_p'])} |")
        for method, pr in paired.items():
            if not pr["pairs"]:
                continue
            L.append(f"\n### `{method}` - `{base}` per cell (SR pp)\n")
            L.append("| operator | tier | seed | " + f"`{base}` | `{method}` | "
                     "delta |")
            L.append("|---|---|---|---|---|---|")
            for p in sorted(pr["pairs"], key=lambda x: (str(x["tier"]),
                                                        str(x["operator"]),
                                                        str(x["seed"]))):
                L.append(
                    f"| `{p['operator']}` | {p['tier']} | {p['seed']} | "
                    f"{_fmt(p['baseline_sr'], pct=True)} | "
                    f"{_fmt(p['method_sr'], pct=True)} | "
                    f"{p['delta']*100:+.1f} |")
    if ivn and any(m["n_cells"] for m in ivn["methods"].values()):
        L.append("\n## Human effort: intervention rate\n")
        L.append("Operator-controlled env steps / all env steps the actor ran "
                 "during training (until convergence or the cap), from each "
                 "cell's `episodes.jsonl`. Lower = less human effort. This is "
                 "the metric E2HiL reports next to success rate.\n")
        L.append("| method | cells | mean rate | median rate | mean operator "
                 "steps/cell | mean env steps/cell | " + " | ".join(tiers) + " |")
        L.append("|---|---|---|---|---|---|" + "|".join(["---"] * len(tiers))
                 + "|")
        for method, m in ivn["methods"].items():
            row = " | ".join(_fmt(m["tier_rate"].get(t), pct=True) for t in tiers)
            L.append(f"| `{method}` | {m['n_cells']} | "
                     f"{_fmt(m['rate_mean'], pct=True)} | "
                     f"{_fmt(m['rate_median'], pct=True)} | "
                     f"{m['intervention_steps_mean']:.0f} | "
                     f"{m['env_steps_mean']:.0f} | {row} |")
        if ivn["paired"]:
            L.append(f"\n| method vs `{base}` | pairs | method lower | "
                     f"{base} lower | ties | mean d rate (pp) | median d "
                     "(pp) | sign-test p |")
            L.append("|---|---|---|---|---|---|---|---|")
            for method, pr in ivn["paired"].items():
                L.append(
                    f"| `{method}` | {pr['n_pairs']} | {pr['n_method_lower']} | "
                    f"{pr['n_baseline_lower']} | {pr['n_tied']} | "
                    f"{pr['mean_delta']*100:+.1f} | "
                    f"{pr['median_delta']*100:+.1f} | "
                    f"{_fmt(pr['sign_test_p'])} |")
    L.append("\n## Headline\n")
    if base in report["methods"]:
        for method, m in report["methods"].items():
            if method == base:
                continue
            d = m.get("delta_icc_vs_baseline")      # base ICC - method ICC
            if d is None or d != d:
                continue
            # the magnitude is printed unsigned; the verb carries the direction,
            # and a change that rounds to 0.000 is reported as unchanged
            if abs(d) < 5e-4:
                change = "leaves operator-dependency ICC unchanged"
            elif d > 0:
                change = f"reduces operator-dependency ICC by {d:.3f}"
            else:
                change = f"increases operator-dependency ICC by {-d:.3f}"
            L.append(f"- `{method}` {change} vs `{base}` "
                     f"(ICC {_fmt(report['methods'][base]['icc'])} -> "
                     f"{_fmt(m['icc'])}).")
    L.append("")
    return "\n".join(L)


def _roster_tier_counts(roster, tiers):
    """Operators per tier as analysed, counted from the roster (not from
    ``--ops_per_tier``, whose default of 5 stays in place when the config gives
    per-tier counts). Returns (int if all tiers are equal else the dict,
    {tier: count})."""
    per_tier = {t: 0 for t in tiers}
    for th in roster:
        per_tier[th.tier] = per_tier.get(th.tier, 0) + 1
    counts = set(per_tier.values())
    return (counts.pop() if len(counts) == 1 else per_tier), per_tier


def analyze(args, campaign: Path, roster, methods, seeds, tiers):
    tier_of = {th.operator_id: th.tier for th in roster}
    sr_by_method = {}
    steps_by_method = {}
    ivn_by_method = {}
    n_have = 0
    for method in methods:
        sr_by_method[method] = {}
        steps_by_method[method] = {}
        ivn_by_method[method] = {}
        for th in roster:
            entry = {"tier": tier_of[th.operator_id], "sr": {}}
            steps_entry = {"tier": tier_of[th.operator_id], "steps": {}}
            ivn_entry = {"tier": tier_of[th.operator_id], "ivn": {}}
            for s in seeds:
                cell = Cell(method, th.operator_id, th.tier, s)
                sr = _read_cell_sr(campaign, cell)
                if sr is not None:
                    entry["sr"][s] = sr
                    n_have += 1
                ivn = _read_cell_interventions(campaign, cell)
                if ivn is not None:
                    ivn_entry["ivn"][s] = ivn
                done = (campaign / "checkpoints"
                        / f"{method}_{th.operator_id}_s{s}" / ".done")
                if done.exists():
                    try:
                        steps_entry["steps"][s] = json.load(
                            open(done)).get("converged_step")
                    except (ValueError, OSError):
                        steps_entry["steps"][s] = None
            sr_by_method[method][th.operator_id] = entry
            steps_by_method[method][th.operator_id] = steps_entry
            ivn_by_method[method][th.operator_id] = ivn_entry
    baseline = args.baseline if args.baseline in methods else methods[0]
    report = compute_dependency(sr_by_method, methods, tiers, baseline)
    conv = compute_convergence(steps_by_method, methods, tiers)

    meta = json.load(open(campaign / "campaign.json"))
    ops_per_tier, per_tier = _roster_tier_counts(roster, tiers)
    campaign_meta = {
        "campaign_dir": str(campaign), "exp_name": args.exp_name,
        "peg_shape": args.peg_shape, "grid_sha256": meta["grid_sha256"],
        "n_grid_trials": meta["n_grid_trials"],
        "n_operators": len(roster), "ops_per_tier": ops_per_tier,
        "operators_per_tier": per_tier,
        "n_seeds": len(seeds), "steps": args.steps,
    }
    (campaign / "report").mkdir(parents=True, exist_ok=True)
    report_out = dict(report, campaign=campaign_meta, sr_by_method=sr_by_method,
                      convergence=conv, steps_by_method=steps_by_method,
                      generated=datetime.datetime.now().isoformat(
                          timespec="seconds"))
    paired = compute_paired(sr_by_method, methods, baseline)
    report_out["paired"] = paired
    interventions = compute_interventions(ivn_by_method, methods, tiers,
                                          baseline)
    report_out["interventions"] = interventions
    json.dump(report_out, open(campaign / "report" / "dependency.json", "w"),
              indent=2)
    md = render_report_md(report, campaign_meta, tiers, conv=conv, paired=paired,
                          ivn=interventions)
    open(campaign / "report" / "dependency.md", "w").write(md)
    _green(f"[analyze] {n_have} cells scored; report -> "
           f"{campaign / 'report' / 'dependency.md'}")
    if getattr(args, "plots", True):
        try:
            import campaign_plots
            figs = campaign_plots.make_plots(campaign)
            if figs:
                _green(f"[analyze] {len(figs)} figure(s) -> "
                       f"{campaign / 'report' / 'plots'}")
        except ImportError:
            _yellow("[analyze] matplotlib not available -> skipping plots "
                    "(pip install matplotlib).")
        except Exception as exc:
            _yellow(f"[analyze] plotting failed ({exc}); report was still "
                    "written.")
    # per-episode intervention-quality (q_int) summary + plots (quality/aqi
    # cells only; no-op when the campaign has none).
    try:
        import q_summary
        qout = q_summary.make_q_report(campaign)
        if qout:
            _green(f"[analyze] q_int per-episode summary -> {qout}")
    except ImportError:
        _yellow("[analyze] matplotlib not available -> q CSVs only "
                "(pip install matplotlib for the q plots).")
    except Exception as exc:
        _yellow(f"[analyze] q-score summary failed ({exc}); report was still "
                "written.")
    # intervention-quality signal audit: AUROC of every per-takeover signal
    # vs episode success + timing-leakage checks (docs/quality_signal_audit.md).
    # No-op when no intervention segments were logged (hilserl/e2hil only).
    try:
        import quality_signal_audit
        qa_out = quality_signal_audit.make_audit(
            str(campaign), plots=getattr(args, "plots", True))
        if qa_out:
            _green(f"[analyze] quality-signal audit -> {qa_out}/summary.md")
    except Exception as exc:
        _yellow(f"[analyze] quality-signal audit failed ({exc}); report was "
                "still written.")
    print("\n" + md)
    return report


# ===========================================================================
# manifest + plan
# ===========================================================================
def write_manifest(args, campaign: Path, roster, grid, seeds):
    tiers = [t for t in args.tiers.split(",") if t]
    manifest = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "exp_name": args.exp_name, "peg_shape": args.peg_shape,
        "tiers": tiers,
        "ops_per_tier": args.ops_per_tier,
        "operators_per_tier": {t: _ops_per_tier(args, t) for t in tiers},
        "methods": args.methods.split(","),
        "seeds": list(seeds), "sample_seed": args.sample_seed,
        "jitter": args.jitter, "demos": args.demos, "filtered": args.filtered,
        "cell_budget_mode": args.cell_budget_mode,
        "steps": args.steps, "learner_max_steps": args.learner_max_steps,
        "grid_sha256": grid["sha256"], "n_grid_trials": len(grid["trials"]),
        "operators": [th.operator_id for th in roster],
        "arms": _arms_summary(args),
    }
    if getattr(args, "synthetic_rotation_control", False):
        manifest["synthetic_rotation_control"] = True   # absent = original operator
    if getattr(args, "synthetic_orientation_monitor", False):
        manifest["synthetic_orientation_monitor"] = {   # absent = original monitor
            "tilt_rad": float(args.synthetic_orientation_tilt_rad) or "shape default",
            "yaw_rad": float(args.synthetic_orientation_yaw_rad) or "shape default"}
    if _rpy_box(args):
        manifest["sim_rpy_box"] = _rpy_box(args)        # absent = unbounded orientation
    if getattr(args, "synthetic_inconsistent", False):
        manifest["synthetic_inconsistent"] = {           # absent = consistent operator
            "bad_share": float(args.synthetic_bad_share),
            "persistence": float(args.synthetic_bad_persistence)}
    if str(getattr(args, "converge_mode", "rolling")) != "rolling":
        manifest["converge_mode"] = {"mode": args.converge_mode,
                                     "every": args.converge_eval_every,
                                     "consecutive": args.converge_eval_consecutive,
                                     "threshold": args.converge_eval_threshold}
    if float(getattr(args, "utd_ratio", 0.0) or 0.0) > 0.0:
        manifest["utd_ratio"] = float(args.utd_ratio)
    if getattr(args, "log_takeover_channels", False):
        manifest["log_takeover_channels"] = True
    json.dump(manifest, open(campaign / "campaign.json", "w"), indent=2)


def _arms_summary(args):
    """{arm: {"aqi_mode": ..., "overrides": {...}}} for every listed arm."""
    return {arm: {"aqi_mode": _arm_mode(args, arm),
                  "overrides": _arm_overrides(args, arm)}
            for arm in args.methods.split(",") if arm}


def _document_campaign(args, campaign: Path, roster, grid, seeds):
    """Top-level README.md describing the whole campaign."""
    tiers = [t for t in args.tiers.split(",") if t]
    per_tier = {t: _ops_per_tier(args, t) for t in tiers}
    ops_str = (", ".join(f"{t}:{n}" for t, n in per_tier.items())
               if getattr(args, "ops_per_tier_map", None) else args.ops_per_tier)
    prov.document(
        campaign, "campaign", quiet=True, task=_task_dict(args),
        extra={"campaign": {
            "n_operators": len(roster), "ops_per_tier": ops_str,
            "operators_per_tier": per_tier,
            "tiers": tiers, "methods": args.methods.split(","),
            "n_seeds": len(seeds), "demos": args.demos, "steps": args.steps,
            "cell_budget_mode": args.cell_budget_mode,
            "converge_success": args.converge_success,
            "converge_eval_threshold": args.converge_eval_threshold,
            "minutes_per_cell": args.minutes_per_cell,
            "max_hours": args.max_hours,
            "grid_sha256": grid["sha256"], "n_grid_trials": len(grid["trials"]),
            "arms": _arms_summary(args),
        }})



def print_plan(args, roster, cells, seeds, grid):
    n_ops = len(roster)
    n_demo = n_ops * len(seeds)
    tiers = [t for t in args.tiers.split(",") if t]
    per_tier = ", ".join(f"{t}={_ops_per_tier(args, t)}" for t in tiers)
    _green("\n================ CAMPAIGN PLAN ================")
    print(f"  campaign dir : {args.campaign}")
    if args.config:
      print(f"  config       : {args.config}")
    print(f"  task         : {args.exp_name} (peg {args.peg_shape})")
    print(f"  operators    : {per_tier}  = {n_ops} total")
    if getattr(args, "unified_monitoring", ""):
        print(f"  monitoring   : UNIFIED at '{args.unified_monitoring}' "
              f"(skill varies only in EXECUTION)")
    if getattr(args, "tier_s_exec", None):
        print(f"  exec scale   : {args.tier_s_exec} "
              f"(0=novice anchor, 1=expert anchor)")
    if getattr(args, "tier_s_mon", None):
        print(f"  monitor scale: {args.tier_s_mon}")
    if getattr(args, "synthetic_rotation_control", False):
        print("  operator     : rotation control (corrects orientation too)")
    if getattr(args, "synthetic_orientation_monitor", False):
        lim = lambda v: f"{v:g} rad" if v else "shape default"
        print(f"  monitor      : orientation channel (tilt "
              f"{lim(args.synthetic_orientation_tilt_rad)}, yaw "
              f"{lim(args.synthetic_orientation_yaw_rad)})")
    if getattr(args, "synthetic_inconsistent", False):
        print(f"  operator     : inconsistent - inattentive in {args.synthetic_bad_share:g} of the "
              f"takeovers, persistence {args.synthetic_bad_persistence:g}")
    if _rpy_box(args):
        r, pt, y = _rpy_box(args)
        print(f"  env          : orientation box roll +/-{r:g}, pitch +/-{pt:g}, "
              f"yaw +/-{y:g} rad (every process: demos, training, ruler)")
    if str(getattr(args, "converge_mode", "rolling")) != "rolling":
        print(f"  convergence  : mini-eval SR >= {args.converge_eval_threshold} in "
              f"{args.converge_eval_consecutive} evaluations in a row, every "
              f"{args.converge_eval_every} env steps")
    if float(getattr(args, "utd_ratio", 0.0) or 0.0) > 0.0:
        print(f"  update ratio : fixed at {args.utd_ratio:g} learner steps per env "
              f"step (env without real-time pacing)")
    if getattr(args, "log_takeover_channels", False):
        print("  logging      : takeover channels and hand-backs per episode")
    print(f"  seeds/op     : {len(seeds)}  {list(seeds)}")
    print(f"  methods      : {args.methods}")
    for arm, a in _arms_summary(args).items():
        if a["overrides"] or a["aqi_mode"] != arm:
            print(f"    arm {arm:14s} aqi_mode={a['aqi_mode']:8s} "
                  + " ".join(f"{k}={v}" for k, v in a["overrides"].items()))
    if args.trigger_terms or args.trigger_combiner:
      print(f"  active gate  : trigger_terms="
          f"{args.trigger_terms or 'default(all)'} "
          f"combiner={args.trigger_combiner or 'default'}")
    print(f"  wandb        : {args.wandb}"
        + ("  (learner logging disabled)" if args.wandb == "off" else ""))
    if getattr(args, "wandb_tags", ""):
        print(f"  wandb tags   : {args.wandb_tags}  (+ campaign, arm, tier, "
              f"operator, seed per cell)")
    print(f"  ruler grid   : {len(grid['trials'])} trials "
        f"({grid['n_poses']} poses x {grid['reps']} reps), "
        f"sha256 {grid['sha256'][:12]}...")
    print(f"  demo jobs    : {n_demo}  (once per operator x seed, "
        f"{args.demos} {'successes' if args.filtered else 'episodes'} each)")
    if args.cell_budget_mode == "time":
      print(f"  train cells  : {len(cells)}  (method x operator x seed), "
          f"budget {args.minutes_per_cell:g} min wall-clock each")
    else:
      print(f"  train cells  : {len(cells)}  (method x operator x seed), "
          f"budget {args.steps} env steps each")
    if args.cell_budget_mode == "time" and args.minutes_per_cell > 0:
      print(f"  time budget  : {args.minutes_per_cell:g} min/cell "
          f"(<= ~{args.minutes_per_cell * len(cells) / 60:.1f} h total "
          f"if no cell converges early)")
    if args.max_hours > 0:
      print(f"  campaign cap : {args.max_hours:g} h wall-clock for the whole "
          f"run (stops launching new cells after that)")
    print(f"  eval jobs    : {len(cells)}  (fixed ruler)")
    _green("  roster:")
    for th in roster:
      print(f"    {th.operator_id:8s} tier={th.tier:12s} "
          f"sigma0={th.sigma0:5.2f} v_max={th.v_max:5.1f} "
          f"delta_in={th.delta_in:.2f} rho_fa={th.rho_fa:.2f} "
          f"p_miss={th.p_miss:.2f}")
    _green("==============================================\n")


# ===========================================================================
# main
# ===========================================================================
def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Automated synthetic-operator campaign runner "
                    "(operator-dependency study).")
    p.add_argument("--campaign", required=True,
                   help="output directory for the whole campaign.")
    p.add_argument("--config", default=None,
                   help="YAML (or JSON) task-definition file describing the "
                        "whole study (task, per-tier operator populations, "
                        "demos, methods, budgets). Command-line flags override "
                        "any field also set in the config.")
    p.add_argument("--exp_name", default="sim_peg_insertion")
    p.add_argument("--peg_shape", default="cylinder",
                   help="sim geometry; cylinder is the translation-only "
                        "supported geometry for synthetic operators.")
    # roster
    p.add_argument("--tiers", default="expert,intermediate,novice")
    p.add_argument("--ops_per_tier", type=int, default=5,
                   help="distinct (jittered) operators per tier (thesis: 5-10).")
    p.add_argument("--jitter", type=float, default=0.15,
                   help="per-operator theta jitter strength (log-normal sigma).")
    p.add_argument("--unified_monitoring", default="",
                   choices=["", "expert", "intermediate", "novice"],
                   help="pin every operator's MONITORING/intervention skill to "
                        "this tier (un-jittered) so operators differ ONLY in "
                        "EXECUTION skill; '' (default) = monitoring varies with "
                        "tier as usual.")
    p.add_argument("--sample_seed", type=int, default=0)
    # replicates + methods
    p.add_argument("--seeds", type=int, default=3,
                   help="replicate seeds per operator (>=2 for a within-operator "
                        "variance estimate).")
    p.add_argument("--seed_base", type=int, default=0)
    p.add_argument("--methods", default="hilserl,active,quality,aqi")
    p.add_argument("--baseline", default="hilserl",
                   help="method the others are compared against.")
    # AQI composition (S6.4 ablations). Which trigger terms feed the active
    # gate and how they combine; quality-module knobs. Empty/unset = the
    # experiment default. hilserl ignores all of these (they show as inert).
    p.add_argument("--trigger_terms", default="",
                   help="comma list subset of {uncertainty,failure,force,ood,"
                        "cost} feeding the active gate (empty = default all 5). "
                        "'uncertainty' alone = the minimal ensemble trigger.")
    p.add_argument("--trigger_combiner", default="",
                   choices=["", "weighted_sum", "max"],
                   help="how the selected trigger terms combine (empty=default).")
    p.add_argument("--quality_filters", default="",
                   help="comma list subset of the quality keep/drop filters "
                        "(quality/aqi arms; empty = default).")
    p.add_argument("--quality_disagreement_gate", default=None,
                   action=argparse.BooleanOptionalAction,
                   help="toggle the quality disagreement gate g(sigma_Q) "
                        "(quality/aqi arms; unset = default).")
    p.add_argument("--bc_loss_weight", type=float, default=None,
                   help="lambda_BC on the weighted imitation loss (quality/aqi "
                        "arms; unset = experiment default). 0.0 = quality "
                        "score drives ONLY the replay priority (no imitation "
                        "term).")
    p.add_argument("--per_standardize", default=None,
                   action=argparse.BooleanOptionalAction,
                   help="divide q_int by its running scale before the replay "
                        "priority so per_eta is dimension-free (quality/aqi).")
    p.add_argument("--quality_score_mode", default=None,
                   choices=["counterfactual", "behavioral", "hindsight",
                            "progress", "outcome", "uniform"],
                   help="graded intervention signal: 'counterfactual' A_hat "
                        "Q-gap (default), critic-free 'behavioral' composite "
                        "z(d_value)-z(t_int)-z(f_peak), 'hindsight' n-step "
                        "advantage to handback+K, 'progress' depth change to "
                        "handback+K, 'outcome' +/-1 episode success, or "
                        "'uniform' (no grading, control) (quality/aqi).")
    p.add_argument("--quality_behavioral_terms", default=None,
                   help="comma-separated terms of the behavioral score out of "
                        "d_value,t_int,f_peak (default all; quality/aqi with "
                        "quality_score_mode=behavioral).")
    p.add_argument("--quality_post_release_steps", type=int, default=None,
                   help="K steps after the handback in the hindsight/progress "
                        "window (default 10; every logging mode).")
    p.add_argument("--quality_hindsight_discount", type=float, default=None,
                   help="discount of the hindsight advantage (default: the "
                        "experiment's TD discount).")
    p.add_argument("--quality_consistency_weight", default=None,
                   action=argparse.BooleanOptionalAction,
                   help="multiply the BC weight by the operator-consistency "
                        "factor (quality/aqi; inert unless bc_loss_weight>0).")
    p.add_argument("--quality_consistency_k", type=int, default=None,
                   help="earlier takeover segments behind sigma_beta "
                        "(default 5; quality/aqi).")
    p.add_argument("--quality_consistency_sigma_ref", type=float,
                   default=None,
                   help="fixed sigma_ref of the consistency factor (default: "
                        "running mean of sigma_beta; quality/aqi).")
    p.add_argument("--quality_drop_fraction", type=float, default=None,
                   help="discard filters (quality_filters score_filter / "
                        "random_filter): share of takeovers dropped from the "
                        "intervention buffer (quality/aqi; default 0.4). Per arm.")
    p.add_argument("--quality_cf_diag", default=None,
                   action=argparse.BooleanOptionalAction,
                   help="log diagnostics of the counterfactual score and the "
                        "synthetic operator's simulator ground truth "
                        "(logging only; every logging mode).")
    p.add_argument("--per_delta_init", type=float, default=None,
                   help="PER |delta| init in the priority (quality/aqi).")
    p.add_argument("--per_eta", type=float, default=None,
                   help="PER weight on max(q_int,0) in the priority (quality/aqi).")
    p.add_argument("--per_eps", type=float, default=None,
                   help="PER epsilon floor in the priority (quality/aqi).")
    p.add_argument("--quality_warmup_steps", type=int, default=None,
                   help="env steps before interventions carry their quality "
                        "replay priority (immature-ensemble warm-up; "
                        "quality/aqi).")
    p.add_argument("--per_scale_decay", type=float, default=None,
                   help="EMA decay for the q-standardization scale (quality/aqi;"
                        " 0 = cumulative mean).")
    p.add_argument("--per_refresh_period", type=int, default=None,
                   help="recompute all priorities every N inserts (quality/aqi).")
    p.add_argument("--per_is_beta", type=float, default=None,
                   help="importance-sampling exponent for prioritised replay "
                        "(quality/aqi; 0 = off).")
    p.add_argument("--converge_mode", choices=["rolling", "eval"], default="rolling",
                   help="rolling = the original convergence rule; eval = the "
                        "autonomous fixed-pose mini-eval decides (needs "
                        "--converge_eval_threshold > 0). YAML: budget.converge_mode.")
    p.add_argument("--converge_eval_every", type=int, default=2000,
                   help="converge_mode eval: env steps between mini-evals.")
    p.add_argument("--converge_eval_consecutive", type=int, default=2,
                   help="converge_mode eval: passes in a row needed to stop.")
    p.add_argument("--utd_ratio", type=float, default=0.0,
                   help="> 0: fixed learner steps per env step (actor and learner "
                        "wait for each other; env without real-time pacing). "
                        "0 = the original asynchronous loop. YAML: budget.utd_ratio.")
    p.add_argument("--log_takeover_channels", action="store_true", default=False,
                   help="log the monitor channel behind each takeover and how it "
                        "ended, per episode (logging only). YAML: "
                        "logging.takeover_channels.")
    p.add_argument("--synthetic_inconsistent", action="store_true", default=False,
                   help="inconsistent synthetic operator: a hidden attentive / "
                        "inattentive condition per takeover, logged as ground "
                        "truth. YAML: aqi.synthetic_inconsistent.")
    p.add_argument("--synthetic_bad_share", type=float, default=0.4,
                   help="inconsistent operator: share of inattentive takeovers.")
    p.add_argument("--synthetic_bad_persistence", type=float, default=0.7,
                   help="inconsistent operator: probability a takeover keeps the "
                        "previous condition.")
    p.add_argument("--sim_rpy_box", default="",
                   help="'roll,pitch,yaw' half-widths (rad) of the sim env's "
                        "orientation box around the aligned home orientation, "
                        "like the real envs' rpy limits; empty = unbounded (the "
                        "original task). Set for every process of the campaign "
                        "(demos, training, ruler) via $SIM_RPY_BOX. YAML: "
                        "task.rpy_box.")
    p.add_argument("--synthetic_orientation_monitor", action="store_true",
                   default=False,
                   help="the synthetic operator's monitor also watches the tool "
                        "orientation (channel 'orientation'); needs "
                        "--synthetic_rotation_control. YAML: "
                        "aqi.synthetic_orientation_monitor.")
    p.add_argument("--synthetic_orientation_tilt_rad", type=float, default=0.0,
                   help="orientation monitor: tilt limit (rad); 0 = the shape's "
                        "default. YAML: aqi.synthetic_orientation_tilt_rad.")
    p.add_argument("--synthetic_orientation_yaw_rad", type=float, default=0.0,
                   help="orientation monitor: yaw limit (rad); 0 = the shape's "
                        "default. YAML: aqi.synthetic_orientation_yaw_rad.")
    p.add_argument("--synthetic_rotation_control", action="store_true",
                   default=False,
                   help="the geometric synthetic operator also corrects the "
                        "tool orientation (needed for the cuboid / hex peg); "
                        "off = the original translation-only operator. "
                        "YAML: aqi.synthetic_rotation_control.")
    p.add_argument("--synthetic_intervention_mode",
                   choices=["signal", "always_latch",
                            "latch_until_autonomous_success"],
                   default="signal",
                   help="Synthetic-operator intervention policy during actor "
                        "training: signal, always_latch, or "
                        "latch_until_autonomous_success.")
    # demos
    p.add_argument("--demos", type=int, default=6,
                   help="demonstrations per (operator, seed). Keep MODEST so the "
                        "online interventions -- where operator skill and AQI "
                        "act -- still matter (too many demos and every method "
                        "converges, hiding the dependency).")
    p.add_argument("--demo_source", choices=["synthetic", "policy"],
                   default="synthetic",
                   help="demo generator: synthetic=geometric oracle demos; "
                        "policy=collect demos with a frozen policy oracle.")
    p.add_argument("--policy_oracle_checkpoint", default="",
                   help="checkpoint directory for PolicyOracle demos/actor "
                        "(enables policy-oracle synthetic interventions when set).")
    p.add_argument("--policy_oracle_mode", default="hilserl",
                   choices=["hilserl", "logged", "active", "quality", "aqi",
                            "e2hil"],
                   help="aqi_mode architecture used by the policy-oracle checkpoint.")
    p.add_argument("--policy_oracle_step", type=int, default=0,
                   help="PolicyOracle checkpoint step (0=latest).")
    p.add_argument("--training_oracle", choices=["auto", "geometric", "policy"],
                   default="auto",
                   help="which oracle drives the ACTOR's intervention action: "
                        "'geometric' (sim's attached oracle), 'policy' (the "
                        "frozen --policy_oracle_checkpoint, even in the sim), "
                        "or 'auto'. 'policy' requires policy_oracle.checkpoint_path.")
    p.add_argument("--policy_mm_per_unit", type=float, default=100.0,
                   help="Policy-oracle action scaling in mm/s per env action unit.")
    p.add_argument("--filtered", action="store_true", default=True,
                   help="keep only successful demos (default).")
    p.add_argument("--nofiltered", dest="filtered", action="store_false")
    p.add_argument("--demo_max_attempts", type=int, default=300)
    # budgets
    p.add_argument("--cell_budget_mode", choices=["steps", "time"],
                   default="steps",
                   help="per-cell budget mode: 'steps' uses only --steps; "
                        "'time' uses only --minutes_per_cell.")
    p.add_argument("--steps", type=int, default=20000,
                   help="per-cell ACTOR env-step cap when "
                        "--cell_budget_mode=steps.")
    p.add_argument("--learner_max_steps", type=int, default=1000000,
                   help="learner cap; kept large so the learner stops via the "
                        "STOP sentinel when the actor finishes, not on its own.")
    p.add_argument("--buffer_period", type=int, default=0,
                   help="dump replay/demo transitions (with images, ~9GB/cell "
                        "over a full run) every N steps; 0 disables the dumps "
                        "(default) since batch cells never warm-restart.")
    p.add_argument("--checkpoint_keep", type=int, default=25,
                   help="how many periodic checkpoints each cell retains "
                        "(orbax 'keep'); smaller saves disk. best_window eval "
                        "still has plenty to select from.")
    p.add_argument("--demo_pretrain", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="enable/disable learner demo-only warm start before "
                        "online replay training.")
    p.add_argument("--demo_pretrain_steps", type=int, default=0,
                   help="optional learner demo-only warm start before waiting "
                        "for online replay data (0 disables).")
    p.add_argument("--demo_pretrain_batch_size", type=int, default=0,
                   help="batch size for demo-only pretraining (0 uses "
                        "config.batch_size).")
    p.add_argument("--minutes_per_cell", type=float, default=0.0,
                   help="fixed WALL-CLOCK training budget per cell in minutes "
                        "when --cell_budget_mode=time.")
    p.add_argument("--max_hours", type=float, default=0.0,
                   help="fixed WALL-CLOCK budget for the WHOLE campaign in hours "
                        "(0 = unlimited). Once elapsed, the runner stops "
                        "launching new training cells and proceeds to evaluate "
                        "+ analyze whatever finished (resumable -- re-run to "
                        "continue the rest).")
    p.add_argument("--eval_max_steps", type=int, default=300,
                   help="max env steps per ruler trial.")
    p.add_argument("--eval_last_k", type=int, default=1,
                   help="mean-over-last-K checkpoints on the ruler (de-noises "
                        "training oscillation); 1 = last checkpoint only.")
    p.add_argument("--eval_select",
                   choices=["pooled", "best", "mean", "best_window"],
                   default="best_window",
                   help="ruler checkpoint-selection metric: 'best_window' "
                        "(default) reports the best mean over K consecutive "
                        "checkpoints (de-noises oscillation without best-of-1 "
                        "selection bias); 'pooled' = the last-K window; 'best' "
                        "= single best checkpoint; 'mean' = mean over all.")
    p.add_argument("--eval_window_k", type=int, default=3,
                   help="window size for --eval_select=best_window.")
    p.add_argument("--learning_rate", type=float, default=0.0,
                   help="LEARNER Adam LR override for actor/critic/temperature "
                        "(0 = built-in 3e-4; 1.5e-4 halves it for stability).")
    # critic-ensemble knobs (ensemble-tradeoff / convergence study; critic_*
    # apply only to non-hilserl arms -- hilserl stays fixed at 2 critics).
    p.add_argument("--critic_ensemble_size", type=int, default=None,
                   help="number of critics K for the ensemble arms (default "
                        "config = 10; hilserl fixed at 2).")
    p.add_argument("--critic_subsample_size", type=int, default=None,
                   help="REDQ target: number M of critics subsampled for the "
                        "min-Q target (smaller = less pessimistic; 1 = single "
                        "random critic, no min). Clamped to K.")
    p.add_argument("--per_member_targets", action=argparse.BooleanOptionalAction,
                   default=None,
                   help="each ensemble member draws its own target "
                        "(decorrelated) instead of one shared REDQ target.")
    # HiRIL external baseline knobs (apply only to arms with mode hiril);
    # defaults live in serl_launcher/hiril/config.py (paper values)
    for _name, _cast in HIRIL_FIELDS.items():
        p.add_argument(f"--hiril_{_name}",
                       type=_to_bool if _cast is bool else _cast, default=None,
                       help=f"HiRIL (train_hiril.py): HiRILConfig.{_name}")
    # E2HiL external baseline knobs (apply only to the 'e2hil' arm)
    p.add_argument("--e2hil_K", type=int, default=None,
                   help="E2HiL: MC actions per state for the entropy-influence "
                        "estimate (paper default 8).")
    p.add_argument("--e2hil_q_low", type=float, default=None,
                   help="E2HiL: lower batch percentile (0-1) of |c(s)| "
                        "(default 0.05).")
    p.add_argument("--e2hil_q_high", type=float, default=None,
                   help="E2HiL: upper batch percentile (0-1) of |c(s)| "
                        "(default 0.90).")
    p.add_argument("--e2hil_q_agg", choices=["min", "mean"], default=None,
                   help="E2HiL: critic reduction in the actor objective "
                        "('min' = official code, 'mean' = hilserl's actor).")
    # convergence-based termination (what actually ends each cell's training)
    p.add_argument("--converge_success", type=float, default=0.9,
                   help="stop a cell once its rolling success rate over "
                        "--converge_window episodes reaches this (0 disables, "
                        "then --steps is the only stop).")
    p.add_argument("--converge_window", type=int, default=20,
                   help="episodes in the rolling success window.")
    p.add_argument("--converge_min_steps", type=int, default=500,
                   help="do not converge before this many env steps.")
    p.add_argument("--converge_eval_threshold", type=float, default=0.0,
                   help="optional ruler-aligned gate: require fixed-pose mini-"
                        "eval success rate >= this value before declaring "
                        "converged (0 disables).")
    p.add_argument("--converge_eval_max_steps", type=int, default=300,
                   help="max env steps per mini-eval trajectory used by "
                        "--converge_eval_threshold.")
    p.add_argument("--converge_eval_poses",
                   default="0.000,0.000;0.005,0.000;0.010,0.000;0.005,0.080;0.010,-0.080",
                   help="semicolon-separated fixed mini-eval reset poses as "
                        "dx_m,dyaw_rad pairs.")
    p.add_argument("--converge_require_autonomous", action="store_true",
                   default=True,
                   help="count only successes with NO operator takeover toward "
                        "convergence (policy solved it unaided; default).")
    p.add_argument("--converge_allow_assisted", dest="converge_require_autonomous",
                   action="store_false",
                   help="count any success toward convergence, even operator-"
                        "assisted ones.")
    # ruler grid
    p.add_argument("--eval_dx_mm", default="-16,-12,-8,-4,0,4,8,12,16",
                   help="lateral offsets (mm) swept by the ruler.")
    p.add_argument("--eval_yaw", default="0",
                   help="yaw offsets (rad); 0 only for the symmetric cylinder.")
    p.add_argument("--eval_reps", type=int, default=1)
    # gpu / process control
    p.add_argument("--jax_platform", default="cuda",
                   help="'cuda' (default) or 'cpu'.")
    p.add_argument("--gpu_mem_learner", default="0.6")
    p.add_argument("--gpu_mem_actor", default="0.2")
    p.add_argument("--gpu_free_mb", type=int, default=1500,
                   help="abort at start if other processes already use more "
                        "than this many MB of VRAM (unless --force).")
    p.add_argument("--wandb", choices=["off", "offline", "online"], default="off",
                   help="learner wandb logging. 'off' (default) disables it "
                        "(--debug); 'offline' logs locally (no key needed); "
                        "'online' needs WANDB_API_KEY in the environment. "
                        "Cells run with no TTY, so 'online' without a key "
                        "crashes the learner.")
    p.add_argument("--wandb_tags", default="",
                   help="extra W&B tags for every learner run, comma-separated "
                        "(e.g. 'D3,RQ4'; YAML logging.wandb_tags). Each run is "
                        "also tagged with its campaign, arm, base mode, tier, "
                        "operator and seed. The tags travel in the "
                        "AQI_WANDB_TAGS env var, so the training commands do "
                        "not change.")
    p.add_argument("--ready_timeout", type=float, default=600.0)
    p.add_argument("--learner_stop_timeout", type=float, default=240.0)
    # flow control
    p.add_argument("--stages", default="sample,demos,train,eval,analyze",
                   help="comma list subset of: sample,demos,train,eval,analyze.")
    p.add_argument("--plan", action="store_true",
                   help="sample + freeze the grid + print the plan, then stop.")
    p.add_argument("--dry_run", action="store_true",
                   help="print every subprocess command instead of running it.")
    p.add_argument("--force", action="store_true",
                   help="bypass the shared-GPU pre-flight check.")
    p.add_argument("--shard", default="",
                   help="'i/n': train/eval only the cells with index %% n == i "
                        "(0-based, in plan order) -- one Slurm array task per "
                        "shard. Run the demos stage once beforehand and the "
                        "analyze stage once afterwards (not per shard). A "
                        "shard never rewrites the roster/grid/manifest of a "
                        "campaign that already has them.")
    p.add_argument("--list_cells", action="store_true",
                   help="print the cell ids in plan order (one per line) and "
                        "exit; writes nothing.")
    p.add_argument("--skip_trained", action="store_true", default=True,
                   help="skip a cell that finished before (has a .done marker); "
                        "partial/crashed cells are retrained from scratch "
                        "(default).")
    p.add_argument("--retrain", dest="skip_trained", action="store_false",
                   help="retrain cells even if they finished before.")
    p.add_argument("--plots", action="store_true", default=True,
                   help="render figures into report/plots/ after analysis "
                        "(default; needs matplotlib).")
    p.add_argument("--no_plots", dest="plots", action="store_false",
                   help="skip figure rendering.")
    return p


# the env's random reset yaw (PandaPegInsertGymEnv.random_rz_range, rad)
RESET_YAW_RANGE = 0.08


def _validate_rpy_box(args):
    """The box must hold every start pose of the campaign: the random training
    reset yaw, the ruler's yaw grid and the convergence mini-eval's poses.
    Otherwise the first step would clip the start pose."""
    try:
        box = _rpy_box(args)
    except ValueError:
        raise SystemExit(f"[config] --sim_rpy_box {args.sim_rpy_box!r}: three "
                         f"numbers 'roll,pitch,yaw' (rad) expected.")
    if box is None:
        return
    if len(box) != 3 or min(box) <= 0.0 or max(box) >= 1.5:
        raise SystemExit(f"[config] --sim_rpy_box {args.sim_rpy_box!r}: three "
                         f"half-widths in (0, 1.5) rad expected (roll, pitch, yaw).")
    yaws = [RESET_YAW_RANGE] + [abs(y) for y in _parse_floats(args.eval_yaw)]
    for pose in str(args.converge_eval_poses or "").split(";"):
        parts = [x for x in pose.split(",") if x.strip()]
        if len(parts) == 2:
            yaws.append(abs(float(parts[1])))
    if max(yaws) > box[2]:
        raise SystemExit(f"[config] --sim_rpy_box yaw +/-{box[2]:g} rad is smaller "
                         f"than a start pose of this campaign ({max(yaws):g} rad: "
                         f"reset yaw, ruler yaw grid or mini-eval poses).")


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.ops_per_tier_map = None
    if args.config:
        cfg = load_config(args.config)
        explicit = _explicit_dests(parser, argv)
        args.ops_per_tier_map = apply_config(args, cfg, explicit) or None
    _valid_terms = {"uncertainty", "failure", "force", "ood", "cost"}
    if args.trigger_terms:
        bad = {t for t in args.trigger_terms.split(",") if t} - _valid_terms
        if bad:
            raise SystemExit(f"[config] unknown trigger_terms {sorted(bad)}; "
                             f"valid: {sorted(_valid_terms)}")
    if args.wandb == "online" and not os.environ.get("WANDB_API_KEY"):
        _yellow("[wandb] --wandb online but WANDB_API_KEY is not set; the "
                "learner runs with no TTY and will crash at wandb init unless "
                "a key is available (run `wandb login`, or use --wandb off/"
                "offline).")
    if args.cell_budget_mode == "steps" and args.steps <= 0:
        raise SystemExit("[config] --steps must be > 0 when "
                         "--cell_budget_mode=steps.")
    if args.cell_budget_mode == "time" and args.minutes_per_cell <= 0:
        raise SystemExit("[config] --minutes_per_cell must be > 0 when "
                         "--cell_budget_mode=time.")
    if args.converge_mode == "eval" and (
            args.converge_eval_threshold <= 0 or args.converge_eval_every <= 0
            or args.converge_eval_consecutive < 1):
        raise SystemExit("[config] --converge_mode eval needs "
                         "--converge_eval_threshold > 0, --converge_eval_every "
                         "> 0 and --converge_eval_consecutive >= 1.")
    if args.utd_ratio < 0:
        raise SystemExit("[config] --utd_ratio must be >= 0 (0 = off).")
    if args.synthetic_orientation_monitor and not args.synthetic_rotation_control:
        raise SystemExit("[config] --synthetic_orientation_monitor needs "
                         "--synthetic_rotation_control: a takeover for a twisted "
                         "tool must be able to undo the twist.")
    if (args.synthetic_orientation_tilt_rad < 0
            or args.synthetic_orientation_yaw_rad < 0):
        raise SystemExit("[config] orientation limits must be >= 0 (0 = default).")
    _validate_rpy_box(args)
    if not (0.0 <= args.synthetic_bad_share <= 1.0 and 0.0 <= args.synthetic_bad_persistence < 1.0):
        raise SystemExit("[config] --synthetic_bad_share must be in [0, 1] and "
                         "--synthetic_bad_persistence in [0, 1).")
    campaign = Path(args.campaign).resolve()
    campaign.mkdir(parents=True, exist_ok=True)
    (campaign / "logs").mkdir(exist_ok=True)

    tiers = [t for t in args.tiers.split(",") if t]
    methods = [m for m in args.methods.split(",") if m]
    validate_arms(args, methods)
    if args.baseline not in methods:
        _yellow(f"[config] baseline '{args.baseline}' is not among the "
                f"methods {methods}; the analysis will use '{methods[0]}'.")
    seeds = [args.seed_base + i for i in range(args.seeds)]
    stages = set(s for s in args.stages.split(",") if s)
    if args.plan:
        stages = {"sample"}

    # -- stage 1: roster + grid + manifest (always; cheap, deterministic) -----
    roster = sample_operators(tiers, _ops_effective(args), args.sample_seed,
                              args.jitter, args.unified_monitoring,
                              getattr(args, "tier_s_exec", None),
                              getattr(args, "tier_s_mon", None))
    if args.list_cells:
        for cell in enumerate_cells(methods, roster, seeds):
            print(cell.id)
        return 0
    shard = _parse_shard(args.shard)
    # parallel shards of one campaign must not rewrite the shared files the
    # prepare step already wrote (a reader could see a half-written file)
    shared_written = ((campaign / "campaign.json").exists()
                      and (campaign / "eval_grid.json").exists())
    write_shared = not (shard and shared_written)
    if shard and ("demos" in stages or "analyze" in stages):
        raise SystemExit(
            "[shard] run the demos stage once BEFORE the shards and the "
            "analyze stage once AFTER them (shards: --stages train,eval); "
            "parallel shards would collect the same demos concurrently.")
    # The operator roster is frozen like the ruler (handoff FINDINGS F1):
    # an existing operators/<id>.json is never re-sampled or overwritten,
    # because the demos and the trained cells of this campaign used it.
    # - To get different operators, use a fresh --campaign folder; changing
    #   jitter / sample_seed for an existing folder only prints a warning.
    # - To reuse an old campaign's operators in a new folder, copy its
    #   operators/*.json there before the first run.
    # - This never repairs a roster that an older run already rewrote; run
    #   examples/roster_audit.py (read-only) to find out.
    roster, differs = freeze_roster(roster, campaign)
    if differs:
        _yellow(f"[roster] keeping the frozen operators/*.json for "
                f"{', '.join(differs)}: the sampler now draws different values "
                "(the campaign was sampled before the fixed key order, or "
                "jitter/sample_seed changed). The frozen files are what the "
                "demos and cells used; examples/roster_audit.py checks this.")
    if write_shared:
        write_roster(roster, campaign)
    # The eval ruler is FROZEN on first write: all cells (including across
    # resumed runs) are scored on the byte-identical, hash-locked grid. A later
    # run that requests a different grid keeps the frozen one and warns -- use a
    # fresh --campaign dir to change the ruler.
    grid_path = campaign / "eval_grid.json"
    requested_grid = build_grid(_parse_floats(args.eval_dx_mm),
                                _parse_floats(args.eval_yaw), args.eval_reps,
                                args.peg_shape)
    if grid_path.exists():
        grid = json.load(open(grid_path))
        if grid.get("sha256") != _sha256_of_grid(grid):
            raise SystemExit(f"[grid] {grid_path} is corrupted (hash mismatch); "
                             "the ruler cannot be trusted.")
        if grid["sha256"] != requested_grid["sha256"]:
            _yellow("[grid] a DIFFERENT eval grid was requested, but this "
                    "campaign already froze one -- keeping the frozen ruler so "
                    "all cells stay comparable. Use a new --campaign dir to "
                    "change the ruler.")
    else:
        grid = requested_grid
        json.dump(grid, open(grid_path, "w"), indent=2)
    if write_shared:
        write_manifest(args, campaign, roster, grid, seeds)
        _document_campaign(args, campaign, roster, grid, seeds)
    cells = enumerate_cells(methods, roster, seeds)
    if shard:
        i, n = shard
        cells = [c for k, c in enumerate(cells) if k % n == i]
        _green(f"[shard] {i}/{n}: {len(cells)} cell(s): "
               + ", ".join(c.id for c in cells))
    print_plan(args, roster, cells, seeds, grid)
    if args.plan:
        _green("[plan] roster, eval_grid.json and campaign.json written. "
               "Re-run without --plan to execute.")
        return 0

    if not args.dry_run:
        gpu_preflight(args)

    ok = True
    # -- stage 2: demos (once per operator x seed) ----------------------------
    if "demos" in stages:
        _green(f"\n===== STAGE demos: {len(roster) * len(seeds)} jobs =====")
        for th in roster:
            for s in seeds:
                ok = collect_demos_job(args, campaign, th.operator_id, s) and ok

    # -- stage 3: train every cell -------------------------------------------
    if "train" in stages:
        _green(f"\n===== STAGE train: {len(cells)} cells =====")
        campaign_start = time.time()
        for i, cell in enumerate(cells, 1):
            if args.max_hours and (time.time() - campaign_start) / 3600.0 >= args.max_hours:
                _yellow(f"[budget] campaign time budget of {args.max_hours} h "
                        f"reached -> stopping training at cell {i}/{len(cells)}; "
                        f"evaluating + analyzing what finished (re-run to "
                        f"continue the rest).")
                break
            _green(f"\n----- cell {i}/{len(cells)}: {cell.id} -----")
            ok = train_cell(args, campaign, cell) and ok

    # -- stage 4: eval every cell --------------------------------------------
    if "eval" in stages:
        _green(f"\n===== STAGE eval: {len(cells)} cells =====")
        for i, cell in enumerate(cells, 1):
            eval_cell(args, campaign, cell)

    # -- stage 5: analyze -----------------------------------------------------
    if "analyze" in stages and not args.dry_run:
        _green("\n===== STAGE analyze =====")
        analyze(args, campaign, roster, methods, seeds, tiers)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
