#!/usr/bin/env python3
"""Driver of the synthetic-operator test suite (docs/synthetic_suite.md).

The suite tests the operator and platform changes of patches 0014-0016 --
rotation control (behaviour), the orientation box and the orientation channel
(signal), mini-eval convergence and a fixed update ratio (convergence) -- on
whichever machine hosts the repo. All paths are relative to the repo.

    python examples/synthetic_suite.py check     # machine + repo check  -> <root>/machine.json
    python examples/synthetic_suite.py stage0    # tests, operator test, tolerance sweep -> <root>/stage0/
    python examples/synthetic_suite.py plan      # settings, switches, status; --validate dry-runs every config
    python examples/synthetic_suite.py run smoke | pilots | confirm | <setting> ...
    python examples/synthetic_suite.py report    # comparison table -> <root>/report.md, report_cells.csv

Stages: 0 no training (CPU) -> 1 smoke (one cell) -> 2 pilots (9 settings x 3
cells) -> 3 confirmation of the best stack at thesis scale. `run` executes the
campaigns one after another (one GPU), each in its own folder <root>/<setting>,
and refuses to reuse a folder with a different config or different limits.

`--suite inconsistent_operator` drives the second suite (docs/inconsistent_operator.md):
the inconsistent operator and the quality scores. Its stage 0 tests the operator in each
condition, `audit` measures how well each score finds the inattentive corrections (Q1),
and the report pairs every arm with the keep-all arm and with the baseline.

`--suite study` drives the comprehensive study (docs/comprehensive_study.md): study A (number
of critics and asking rate, consistent experts and intermediates) and study B (quality scores
against HIL-SERL, E2HiL and HiRIL, a sometimes-careless expert), each with a smoke campaign
first. Its stage 0 runs the tests, the operator in each condition and the tolerance sweep;
`run --parallel 2` trains two cells side by side on one GPU; `plots` draws the study figures.

    python examples/synthetic_suite.py --suite study all --push       # EVERYTHING, unattended, never stops
                                                                     # (tools/run_study_unattended.sh runs it detached)
    python examples/synthetic_suite.py --suite study check            # also: W&B login when a config logs online
    python examples/synthetic_suite.py --suite study stage0
    python examples/synthetic_suite.py --suite study run smoke --parallel 2
    python examples/synthetic_suite.py --suite study run A_consistent B_quality --parallel 2
    python examples/synthetic_suite.py --suite study rulercheck        # would the shorter ruler change a smoke score?
    python examples/synthetic_suite.py --suite study audit             # score validity in B_quality
    python examples/synthetic_suite.py --suite study report
    python examples/synthetic_suite.py --suite study plots

`--suite final` drives the final study (docs/final_study.md): asking x grading on the same operators and seeds
(the ensemble, asking, grading, both = AQI), the perfect filter, random requests and HIL-SERL; every operator is
careless in 40 % of its takeovers; experts, intermediates and a stronger novice (skill 0.3); held requests. Two main
campaigns: F1_hard (3 demonstrations) first, then F2_standard (6). Unattended, results only in results/final/:

    bash tools/run_final_unattended.sh          # everything, in the background; never pushes anything

Every command except plan and export ends by refreshing results/<suite>/ (run also after each finished campaign):
reports, plots, the machine check, stage 0 and every campaign's light files (ruler outcomes, training records,
takeover logs, compacted logs) in the runs folder's layout, without checkpoints, demonstrations or per-step logs.
That folder is what goes to the analysis; `export` refreshes it by hand, `--no_export` skips it, and
`--root results/<suite> report|plots|audit` re-analyses an exported folder in place.
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import traceback
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CONFIGS = HERE / "campaigns" / "synthetic_suite"
SIM_ENV = HERE / "experiments" / "sim_peg_insertion" / "sim_env.sh"
DEFAULT_ROOT = REPO / "runs" / "synthetic_suite"

GROUPS = {
    "smoke": ["S_smoke"],
    "pilots": ["L0_original", "L1_rotation", "L2_box", "L3_channel", "L4_evalconv", "L5_full",
               "X_nobox", "X_nochannel", "X_noeval"],
    "confirm": ["C_reference", "C_full"],
}
ORDER = GROUPS["smoke"] + GROUPS["pilots"] + GROUPS["confirm"]
TESTS = ["tests/test_operator_rotation.py", "tests/test_update_ratio.py", "tests/test_takeover_channels.py",
         "tests/test_campaign_switches.py", "tests/test_orientation_box.py", "tests/test_orientation_channel.py",
         "tests/test_sim_orientation.py", "tests/test_synthetic_suite.py"]
REF_ARMS = ()                  # arms every other arm of a setting is paired with in the report

_INC = ["Q1_validity", "Q2_pilot", "Q3_consistent", "Q2_confirm", "two_arm_pilot", "two_arm_confirm"]
SUITES = {
    "synthetic_suite": dict(configs=CONFIGS, root=DEFAULT_ROOT, groups=GROUPS, order=ORDER, tests=TESTS,
                            ref_arms=REF_ARMS),
    "inconsistent_operator": dict(
        configs=HERE / "campaigns" / "inconsistent_operator", root=REPO / "runs" / "inconsistent_operator",
        groups={"q1": ["Q1_validity_orig", "Q1_validity_new"], "q2": ["Q2_pilot_orig", "Q2_pilot_new"],
                "q3": ["Q3_consistent_orig", "Q3_consistent_new"],
                "q2confirm": ["Q2_confirm_orig", "Q2_confirm_new"],
                # the two uses of the score alone: discard and re-weight
                "twoarm": ["two_arm_pilot_orig", "two_arm_pilot_new"],
                "twoarmconfirm": ["two_arm_confirm_orig", "two_arm_confirm_new"],
                # a platform's pilots in order; the confirmation only by name, after the decision
                "orig": [f"{n}_orig" for n in _INC[:3]], "new": [f"{n}_new" for n in _INC[:3]]},
        order=[f"{n}_orig" for n in _INC] + [f"{n}_new" for n in _INC],
        tests=["tests/test_inconsistent_operator.py", "tests/test_operator_rotation.py",
               "tests/test_takeover_channels.py", "tests/test_quality_scores_v2.py", "tests/test_quality_audit.py",
               "tests/test_campaign_switches.py", "tests/test_synthetic_suite.py"],
        ref_arms=("keepall", "hilserl", "drop-hind")),
    "study": dict(
        configs=HERE / "campaigns" / "study", root=REPO / "runs" / "study",
        groups={"smoke": ["S0_smoke_A", "S0_smoke_B"], "a": ["A_consistent"], "b": ["B_quality"],
                "main": ["A_consistent", "B_quality"], "audit": ["S0_smoke_B", "B_quality"]},
        order=["S0_smoke_A", "S0_smoke_B", "A_consistent", "B_quality"],
        tests=["tests/test_inconsistent_operator.py", "tests/test_operator_rotation.py",
               "tests/test_takeover_channels.py", "tests/test_quality_scores_v2.py", "tests/test_quality_audit.py",
               "tests/test_campaign_switches.py", "tests/test_campaign_hiril.py", "tests/test_campaign_e2hil.py",
               "tests/test_update_ratio.py", "tests/test_ruler_select.py", "tests/test_synthetic_suite.py",
               "tests/test_study_suite.py"],
        ref_arms=("hilserl", "ens-k8", "keepall")),
    "final": dict(
        configs=HERE / "campaigns" / "final", root=REPO / "runs" / "final",
        groups={"smoke": ["F0_smoke"], "hard": ["F1_hard"], "standard": ["F2_standard"],
                "main": ["F1_hard", "F2_standard"], "audit": ["F0_smoke", "F1_hard", "F2_standard"]},
        order=["F0_smoke", "F1_hard", "F2_standard"],
        tests=["tests/test_final_suite.py", "tests/test_inconsistent_operator.py", "tests/test_operator_rotation.py",
               "tests/test_takeover_channels.py", "tests/test_quality_scores_v2.py", "tests/test_quality_audit.py",
               "tests/test_campaign_switches.py", "tests/test_update_ratio.py", "tests/test_ruler_select.py",
               "tests/test_synthetic_suite.py"],
        ref_arms=("hilserl", "keepall"), ruler_check=False, gb_per_cell=1.6,
        operator_tiers="expert,intermediate,s=0.3"),
    # phase 2 of the extension study (docs/extension_study.md): operator screen of four candidate tasks, learning
    # pilots of the passing ones, the task decision, a smoke test of the new arms on the chosen task
    "extension": dict(
        configs=HERE / "campaigns" / "extension", root=REPO / "runs" / "extension",
        groups={"pilot": ["P1_hex", "P2_tight", "P3_wide", "P4_hexwide"],
                "smoke": ["S1_hex", "S2_tight", "S3_wide", "S4_hexwide"],
                "audit": ["S1_hex", "S2_tight", "S3_wide", "S4_hexwide"]},
        order=["P1_hex", "P2_tight", "P3_wide", "P4_hexwide", "S1_hex", "S2_tight", "S3_wide", "S4_hexwide"],
        tests=["tests/test_extension_suite.py", "tests/test_final_twins.py", "tests/test_final_suite.py",
               "tests/test_inconsistent_operator.py", "tests/test_operator_rotation.py",
               "tests/test_takeover_channels.py", "tests/test_quality_scores_v2.py", "tests/test_quality_audit.py",
               "tests/test_campaign_switches.py", "tests/test_update_ratio.py", "tests/test_ruler_select.py",
               "tests/test_synthetic_suite.py"],
        ref_arms=("hilserl", "keepall"), ruler_check=False, gb_per_cell=1.6, sweep_shapes="cuboid,hex"),
    # phase 3 of the extension study (docs/extension_phase3.md): the NaN probe, a smoke test, the main
    # campaign with the sometimes-careless operator and its careful twin, on the task phase 2 chose (hex peg from the
    # wider start range), fixed budget of 20 000 environment steps
    "phase3": dict(
        configs=HERE / "campaigns" / "phase3", root=REPO / "runs" / "phase3",
        groups={"probe": ["X0_probe"], "smoke": ["X_smoke"], "main": ["X1_careless", "X2_careful"],
                "audit": ["X_smoke", "X1_careless"]},
        order=["X0_probe", "X_smoke", "X1_careless", "X2_careful"],
        tests=["tests/test_run_guard.py", "tests/test_phase3.py", "tests/test_e2hil.py",
               "tests/test_extension_suite.py", "tests/test_final_twins.py", "tests/test_final_suite.py",
               "tests/test_inconsistent_operator.py", "tests/test_operator_rotation.py",
               "tests/test_takeover_channels.py", "tests/test_quality_scores_v2.py", "tests/test_quality_audit.py",
               "tests/test_campaign_switches.py", "tests/test_update_ratio.py", "tests/test_ruler_select.py",
               "tests/test_synthetic_suite.py"],
        ref_arms=("hilserl", "keepall"), ruler_check=False, gb_per_cell=1.0, sweep_shapes="hex"),
}
SUITE = "synthetic_suite"


def configure(name):
    """Point the module at one suite (configs, groups, default root, tests)."""
    global CONFIGS, GROUPS, ORDER, TESTS, DEFAULT_ROOT, REF_ARMS, SUITE
    if name not in SUITES:
        raise SystemExit(f"[suite] unknown suite {name!r}; suites: {', '.join(SUITES)}")
    d = SUITES[name]
    CONFIGS, GROUPS, ORDER, TESTS = d["configs"], d["groups"], d["order"], d["tests"]
    DEFAULT_ROOT, REF_ARMS, SUITE = d["root"], d["ref_arms"], name
EPISODE_LIMIT = 300            # sim_peg_insertion max_traj_length: an episode this long ran out of time
EARLY = 10000                  # the "first 10 k env steps" window of the evidence


# ===================================================================== configs
def load_yaml(path):
    try:
        import yaml
    except ImportError:                                   # same fallback as run_campaign.load_config
        sys.path.insert(0, str(HERE))
        from run_campaign import load_config
        return load_config(str(path))
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def config_path(name):
    p = CONFIGS / f"{name}.yaml"
    if not p.exists():
        raise SystemExit(f"[suite] no config {p.relative_to(REPO)}; settings: {', '.join(ORDER)}")
    return p


def switches(cfg):
    """The tested switches of one config, as the campaign runner reads them."""
    task, aqi = cfg.get("task") or {}, cfg.get("aqi") or {}
    bud, log = cfg.get("budget") or {}, cfg.get("logging") or {}
    box = task.get("rpy_box")
    return {
        "inconsistent": bool(aqi.get("synthetic_inconsistent", False)),
        "rotation": bool(aqi.get("synthetic_rotation_control", False)),
        "box": [float(v) for v in box] if box else None,
        "channel": bool(aqi.get("synthetic_orientation_monitor", False)),
        "convergence": str(bud.get("converge_mode", "rolling")),
        "utd_ratio": float(bud.get("utd_ratio", 0) or 0),
        "takeover_logging": bool(log.get("takeover_channels", False)),
        "peg_shape": task.get("peg_shape", "cylinder"),
        "methods": list(cfg.get("methods") or []),
    }


def resolve(names):
    out = []
    for n in names:
        out += GROUPS.get(n, [n])
    for n in out:
        config_path(n)
    return out


# ===================================================================== helpers
def sh(cmd, log=None, env=None, cwd=REPO, sim_env=True):
    """Run a command (list) inside the sim environment (sim_env.sh puts the repo's packages first on
    PYTHONPATH and selects EGL rendering). Returns the exit code; output goes to `log` and the console."""
    inner = " ".join(_q(c) for c in cmd)
    if sim_env and SIM_ENV.exists():
        inner = f"source {_q(SIM_ENV)} >/dev/null 2>&1; {inner}"
    full_env = dict(os.environ, **(env or {}))
    with (open(log, "a", encoding="utf-8") if log else open(os.devnull, "w")) as lf:
        p = subprocess.Popen(["bash", "-c", inner], cwd=str(cwd), env=full_env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
        for line in p.stdout:
            sys.stdout.write(line)
            lf.write(line)
        return p.wait()


def _q(s):
    s = str(s)
    return s if re.fullmatch(r"[\w@%+=:,./-]+", s) else "'" + s.replace("'", "'\"'\"'") + "'"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


# ===================================================================== check
PROBE = r'''
import json, os, sys
out = {"python": sys.version.split()[0], "packages": {}, "paths": {}}
for name in ("jax", "flax", "optax", "gymnasium", "mujoco", "scipy", "yaml", "agentlace", "wandb", "tensorflow"):
    try:
        m = __import__(name)
        out["packages"][name] = getattr(m, "__version__", "ok")
    except Exception as exc:
        out["packages"][name] = None
for name in ("franka_sim", "serl_launcher", "experiments"):
    try:
        m = __import__(name)
        out["paths"][name] = os.path.dirname(os.path.abspath(m.__file__))
    except Exception as exc:
        out["paths"][name] = "ERROR: %s" % exc
try:
    import jax
    out["jax_devices"] = [str(d) for d in jax.devices()]
except Exception as exc:
    out["jax_devices"] = "ERROR: %s" % exc
try:
    from experiments.sim_peg_insertion.wrapper import SimPegEnv
    env = SimPegEnv(fake_env=True, image_obs=True, peg_shape="cuboid", rpy_box=[0.1, 0.1, 0.4])
    obs, _ = env.reset()
    out["sim_render"] = {k: list(v.shape) for k, v in obs.get("images", {}).items()}
    out["sim_rpy_box"] = [float(v) for v in env._rpy_box]
    env.close()
except Exception as exc:
    out["sim_render"] = "ERROR: %s" % exc
print("PROBE_JSON " + json.dumps(out))
'''

FEATURES = {   # patch: (file, text that only exists with the patch)
    "0014 rotation control": ("serl_launcher/serl_launcher/aqi/operator_model.py", "rotation_control"),
    "0015 takeover channels / mini-eval / update ratio": ("examples/train_aqi_rlpd.py", '"utd_ratio"'),
    "0015 update-ratio lock": ("serl_launcher/serl_launcher/utils/update_ratio.py", "class UpdateRatio"),
    "0016 orientation box": ("franka_sim/franka_sim/envs/orientation_box.py", "def clip_to_rpy_box"),
    "0016 orientation channel": ("examples/experiments/sim_peg_insertion/wrapper.py", "def tilt_yaw_error"),
    "0016 tolerance sweep": ("examples/oracle_insertion_test.py", "def run_twist_episode"),
    "0018 inconsistent operator": ("serl_launcher/serl_launcher/aqi/operator_model.py", "class Inconsistency"),
    "0020 first-machine features": ("serl_launcher/serl_launcher/aqi/query_gate.py", "query_target_rate"),
    "0021 per-operator condition sequence": ("serl_launcher/serl_launcher/aqi/operator_model.py", "zlib.crc32"),
    "0037 shorter ruler": ("examples/ruler_select.py", "def keep_checkpoints"),
    "0048 HiRIL with the inconsistent operator": ("examples/train_hiril.py", '"synthetic_inconsistent"'),
}


def repo_features():
    out = {}
    for name, (rel, needle) in FEATURES.items():
        p = REPO / rel
        out[name] = p.exists() and needle in p.read_text(encoding="utf-8", errors="replace")
    return out


def gpu_info():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20)
        return [l.strip() for l in r.stdout.splitlines() if l.strip()] or None
    except Exception:
        return None


def git(*args):
    try:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True,
                              timeout=20).stdout.strip()
    except Exception:
        return ""


def wandb_logged_in(env=None, home=None):
    """True when W&B can log without a prompt: WANDB_API_KEY is set, or `wandb login` wrote the key to ~/.netrc.
    Campaign cells run unattended, so an 'online' learner without a key would fail or block."""
    env = os.environ if env is None else env
    if env.get("WANDB_API_KEY"):
        return True
    netrc = Path(env.get("NETRC") or Path(home or Path.home()) / ".netrc")
    try:
        return "api.wandb.ai" in netrc.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


GB_PER_CELL = 1.3        # disk per cell: up to 12 checkpoints of ~95 MB (8 critics), the certified one, logs, W&B
DISK_MARGIN_GB = 20


def planned_cells(name):
    """Cells of a config: arms x operators x seeds."""
    cfg = load_yaml(config_path(name))
    ops = sum(int(v or 0) for k, v in (cfg.get("operators") or {}).items() if k in ("expert", "intermediate", "novice"))
    seeds = int((cfg.get("replicates") or {}).get("seeds", 1) or 1)
    return len(cfg.get("methods") or []) * ops * seeds


def disk_needed_gb(names=None):
    """Free disk the study needs for its runs folder: planned cells x GB per cell (the suite's own, else GB_PER_CELL)
    + a margin."""
    per = float(SUITES[SUITE].get("gb_per_cell", GB_PER_CELL))
    return round(sum(planned_cells(n) for n in (names or ORDER)) * per + DISK_MARGIN_GB, 1)


def configs_logging_online(names=None):
    out = []
    for n in names or ORDER:
        log = load_yaml(config_path(n)).get("logging") or {}
        if str(log.get("wandb", "off")) == "online":
            out.append(n)
    return out


def cmd_check(args):
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    info = {"created": now(), "host": socket.gethostname(), "repo": str(REPO),
            "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
            "gpus": gpu_info(), "cpus": os.cpu_count(),
            "disk_free_gb": round(shutil.disk_usage(root).free / 2 ** 30, 1),
            "features": repo_features(), "sim_env_sh": SIM_ENV.exists()}
    probe = root / "probe.py"
    probe.write_text(PROBE, encoding="utf-8")
    log = root / "check.log"
    if log.exists():
        log.unlink()
    sh([sys.executable, str(probe)], log=log)
    m = re.search(r"PROBE_JSON (.*)", log.read_text(encoding="utf-8", errors="replace"))
    info["env"] = json.loads(m.group(1)) if m else "probe failed: see check.log"
    probe.unlink(missing_ok=True)
    info["machine_id"] = f"{info['host']} / {(info['gpus'] or ['no GPU'])[0].split(',')[0]}"
    problems = [f"missing: {k}" for k, ok in info["features"].items() if not ok]
    env = info["env"] if isinstance(info["env"], dict) else {}
    for k in ("jax", "flax", "optax", "gymnasium", "mujoco", "scipy", "yaml"):
        if not (env.get("packages") or {}).get(k):
            problems.append(f"package not importable: {k}")
    for k, p in (env.get("paths") or {}).items():
        if not str(p).startswith(str(REPO)):
            problems.append(f"{k} is imported from {p}, not from this repo")
    if not isinstance(env.get("sim_render"), dict) or not env.get("sim_render"):
        problems.append(f"the sim env does not render: {env.get('sim_render')}")
    if not any("gpu" in str(d).lower() or "cuda" in str(d).lower() for d in (env.get("jax_devices") or [])):
        problems.append(f"JAX sees no GPU: {env.get('jax_devices')}")
    if SUITE == "extension":        # one campaign at a time: scored checkpoints are pruned after each campaign
        per = float(SUITES[SUITE].get("gb_per_cell", GB_PER_CELL))
        need = round(max(planned_cells(n) for n in ORDER) * per + DISK_MARGIN_GB, 1)
    elif SUITE == "phase3":         # each run is pruned right after it is scored (eval.prune_keep_last)
        need = disk_needed_gb()
    else:
        need = disk_needed_gb() if SUITE in ("study", "final") else 50
    info["disk_needed_gb"] = need
    if info["disk_free_gb"] < need:
        problems.append(f"only {info['disk_free_gb']} GB free under {root}; the planned cells need about {need:g} GB "
                        f"(checkpoints of every cell stay on disk)")
    online = configs_logging_online()
    info["wandb_online_configs"] = online
    info["wandb_logged_in"] = wandb_logged_in()
    if online and not info["wandb_logged_in"]:
        problems.append(f"W&B: {', '.join(online)} log online, but no key is set: run `./.venv/bin/wandb login` once "
                        f"(or pass --wandb offline / off to `run`)")
    info["problems"] = problems
    (root / "machine.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"\n[check] machine {info['machine_id']} · commit {info['git_commit'][:8]}"
          f"{' (dirty)' if info['git_dirty'] else ''} · {info['disk_free_gb']} GB free")
    for k, ok in info["features"].items():
        print(f"  {'ok ' if ok else 'NO '} {k}")
    print("[check] " + ("READY" if not problems else "NOT READY:\n  - " + "\n  - ".join(problems)))
    print(f"[check] wrote {root / 'machine.json'}")
    return 0 if not problems else 1


# ===================================================================== stage 0
def limits_from_sweep(tolerance_rows):
    """Orientation-channel limits per peg shape from the sweep summary: the expert's tolerance (rotation
    control off) in rad. 0 = keep the shape default, used when the sweep gave no value (no success at
    0 deg) or a tolerance of 0 (already the smallest swept angle failed); both are listed in `warnings`."""
    out, warnings = {}, []
    for r in tolerance_rows:
        if r.get("tier") != "expert":
            continue
        tol = r.get("tolerance_deg")
        key = "tilt_rad" if r.get("axis") == "tilt" else "yaw_rad"
        out.setdefault(r["shape"], {"tilt_rad": 0.0, "yaw_rad": 0.0})
        out[r["shape"]][key] = round(math.radians(tol), 4) if tol else 0.0
        if not tol:
            warnings.append(f"{r['shape']} {r.get('axis')}: tolerance {tol!r} - the shape default is kept")
    return out, warnings


def cmd_stage0(args):
    root = Path(args.root) / "stage0"
    root.mkdir(parents=True, exist_ok=True)
    rc_all = 0
    if not args.skip_tests:
        log = root / "tests.log"
        log.write_text(f"# {now()}\n", encoding="utf-8")
        results = {}
        for t in TESTS:
            rc = sh([sys.executable, t], log=log)
            results[t] = rc
            rc_all |= rc
        (root / "tests.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print("[stage0] tests:", ", ".join(f"{Path(t).stem} {'ok' if rc == 0 else 'FAILED'}" for t, rc in results.items()))
    if not args.skip_operator_test and SUITE in ("inconsistent_operator", "study", "final"):
        # the inconsistent operator held in each condition: inattentive must be clearly worse
        tiers = SUITES[SUITE].get("operator_tiers")
        rc_all |= sh([sys.executable, "examples/oracle_insertion_test.py", "--shapes", "cuboid",
                      "--conditions", "attentive,inattentive", "--rotation", "off,on",
                      *(["--tiers", tiers] if tiers else []),
                      "--episodes", str(args.episodes), "--out", str(root / "operator_conditions.json")],
                     log=root / "operator_conditions.log", env={"JAX_PLATFORMS": "cpu"})
    elif not args.skip_operator_test and SUITE not in ("extension", "phase3"):  # these screen operators themselves
        rc_all |= sh([sys.executable, "examples/oracle_insertion_test.py", "--shapes", "cuboid,cylinder",
                      "--episodes", str(args.episodes), "--out", str(root / "operator_test.json")],
                     log=root / "operator_test.log", env={"JAX_PLATFORMS": "cpu"})
    if not args.skip_sweep and SUITE in ("synthetic_suite", "study", "final", "extension", "phase3"):
        shapes = SUITES[SUITE].get("sweep_shapes", "cuboid,cylinder")
        rc = sh([sys.executable, "examples/oracle_insertion_test.py", "--sweep", "tilt,yaw", "--shapes", shapes,
                 "--tiers", "expert,novice", "--episodes", str(args.sweep_episodes),
                 "--out", str(root / "tolerance.json")], log=root / "tolerance.log", env={"JAX_PLATFORMS": "cpu"})
        rc_all |= rc
        if rc == 0 and (root / "tolerance.json").exists():
            tol = json.loads((root / "tolerance.json").read_text(encoding="utf-8"))
            lim, warn = limits_from_sweep(tol.get("tolerance", []))
            (root / "limits.json").write_text(json.dumps({"created": now(), "rule": "expert, rotation control off, "
                                                          "largest angle with success >= 80 % of the success at 0 deg",
                                                          "limits": lim, "warnings": warn}, indent=2), encoding="utf-8")
            print("[stage0] orientation limits (rad, 0 = shape default):", json.dumps(lim))
            for w in warn:
                print("[stage0] WARNING:", w)
    print("[stage0] " + ("done" if rc_all == 0 else "finished with failures - see the logs in " + str(root)))
    return rc_all


def read_limits(root, shape):
    """Orientation limits of stage 0: this suite's; the inconsistent-operator suite falls back on the
    synthetic suite's of the same repo (it measures none itself)."""
    paths = [Path(root) / "stage0" / "limits.json"]
    if SUITE == "inconsistent_operator":
        paths.append(REPO / "runs" / "synthetic_suite" / "stage0" / "limits.json")
    for p in paths:
        if p.exists():
            return (json.loads(p.read_text(encoding="utf-8")).get("limits") or {}).get(shape)
    return None


# ===================================================================== plan / status
def cell_status(camp):
    ck = Path(camp) / "checkpoints"
    if not ck.exists():
        return 0, 0
    cells = [d for d in ck.iterdir() if d.is_dir()]
    return sum((d / ".done").exists() for d in cells), len(cells)


def cmd_plan(args):
    root = Path(args.root)
    print(f"{'setting':19s} {'inc':4s} {'rot':4s} {'box':4s} {'chan':5s} {'conv':8s} {'utd':4s} {'peg':9s} {'arms':28s} status")
    for n in resolve(args.names or ORDER):
        sw = switches(load_yaml(config_path(n)))
        done, seen = cell_status(root / n)
        rep = (root / n / "report" / "dependency.json").exists()
        status = "not started" if not seen else f"{done}/{seen} cells done" + (", report" if rep else "")
        print(f"{n:19s} {'x' if sw['inconsistent'] else '-':4s} {'x' if sw['rotation'] else '-':4s} "
              f"{'x' if sw['box'] else '-':4s} {'x' if sw['channel'] else '-':5s} "
              f"{sw['convergence']:8s} {sw['utd_ratio']:<4g} {sw['peg_shape']:9s} {','.join(sw['methods'])[:28]:28s} {status}")
    if args.validate:
        bad = 0
        for n in resolve(args.names or ORDER):
            tmp = root / "_dryrun" / n
            shutil.rmtree(tmp, ignore_errors=True)
            rc = subprocess.run([sys.executable, str(HERE / "run_campaign.py"), "--config", str(config_path(n)),
                                 "--campaign", str(tmp), "--dry_run"], cwd=str(REPO), capture_output=True,
                                text=True).returncode
            print(f"[validate] {n}: {'ok' if rc == 0 else 'FAILED (rc %d)' % rc}")
            bad += rc != 0
        shutil.rmtree(root / "_dryrun", ignore_errors=True)
        return 1 if bad else 0
    return 0


# ===================================================================== run
def campaign_cmd(args, name, root):
    cfg_p = config_path(name)
    sw = switches(load_yaml(cfg_p))
    cmd = [sys.executable, str(HERE / "run_campaign.py"), "--config", str(cfg_p), "--campaign", str(Path(root) / name)]
    limits = None
    if sw["channel"]:
        limits = read_limits(root, sw["peg_shape"])
        for k in ("tilt_rad", "yaw_rad"):
            if limits and float(limits.get(k) or 0) > 0:
                cmd += [f"--synthetic_orientation_{k}", f"{float(limits[k]):g}"]
    if args.wandb:
        cmd += ["--wandb", args.wandb]
    cmd += list(args.extra or [])
    return cmd, sw, limits


def fingerprint(name, limits):
    return {"config": str(config_path(name).relative_to(REPO)), "config_sha256": sha256(config_path(name)),
            "limits": limits}


def score_mismatch(root, name):
    """Q2/Q3 (inconsistent_operator): the quality arms must score with the best online score of the same
    platform's Q1 audit. Returns what is wrong, or None."""
    if name.startswith("Q1_"):
        return None
    aud = Path(root) / f"Q1_validity_{name.rsplit('_', 1)[-1]}" / "report" / "inconsistent_audit.json"
    if not aud.exists():
        return (f"no Q1 audit yet ({aud}). Run its Q1 campaign and `audit` first, then set quality_score_mode "
                f"of the quality arms in {config_path(name).relative_to(REPO)}.")
    best = (json.loads(aud.read_text(encoding="utf-8")).get("best_online_score") or {}).get("quality_score_mode")
    arms = load_yaml(config_path(name)).get("arms") or {}
    modes = {a.get("quality_score_mode") for a in arms.values() if isinstance(a, dict) and a.get("mode") == "quality"}
    if best and modes != {best}:
        return (f"its quality arms score with {', '.join(sorted(str(m) for m in modes))}, but the best online score "
                f"of the Q1 audit is {best}. Edit quality_score_mode in {config_path(name).relative_to(REPO)}.")
    return None


def parallel_cmds(cmd, n, port_base=6100, mem_learner=0.6, mem_actor=0.2):
    """One campaign as n side-by-side shards on one GPU, in three steps (as the cluster runs it):
    roster and demonstrations once, then n processes that each train and score every n-th cell, then the
    analysis. Each shard gets its own learner/actor ports (AQI_TRAINER_PORT p -> p, p + 1) and 1/n of the
    GPU-memory caps, and skips the shared-GPU pre-flight check (the other shard is expected there)."""
    prep = list(cmd) + ["--stages", "sample,demos"]
    shards = []
    for i in range(n):
        c = list(cmd) + ["--stages", "train,eval", "--shard", f"{i}/{n}", "--force",
                         "--gpu_mem_learner", f"{mem_learner / n:.2f}", "--gpu_mem_actor", f"{mem_actor / n:.2f}"]
        shards.append((c, {"AQI_TRAINER_PORT": str(int(port_base) + 10 * i)}))
    ana = list(cmd) + ["--stages", "analyze"]
    return prep, shards, ana


def run_parallel(cmd, n, name, root, port_base=6100):
    """Run a campaign as n shards side by side (parallel_cmds); returns the first non-zero exit code."""
    prep, shards, ana = parallel_cmds(cmd, n, port_base)
    log = Path(root) / f"{name}.out"
    rc = sh(prep, log=log, sim_env=False)
    if rc != 0:
        print(f"[run] {name}: roster / demonstrations failed (rc {rc})")
        return rc
    procs = []
    for i, (c, env) in enumerate(shards):
        lf = open(Path(root) / f"{name}.shard{i}.out", "a", encoding="utf-8")
        p = subprocess.Popen(["bash", "-c", " ".join(_q(x) for x in c)], cwd=str(REPO), env=dict(os.environ, **env),
                             stdout=lf, stderr=subprocess.STDOUT, text=True, errors="replace")
        procs.append((p, lf))
        print(f"[run] {name}: shard {i}/{n} started (pid {p.pid}, port {env['AQI_TRAINER_PORT']}, log {lf.name})")
    rcs = []
    for p, lf in procs:
        rcs.append(p.wait())
        lf.close()
    print(f"[run] {name}: shards finished with " + ", ".join(f"rc {r}" for r in rcs))
    rc = next((r for r in rcs if r != 0), 0)
    rc_a = sh(ana, log=log, sim_env=False)
    return rc or rc_a


def cmd_run(args):
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    names = resolve(args.names)
    for n in names:
        cmd, sw, limits = campaign_cmd(args, n, root)
        if SUITE == "inconsistent_operator" and n.endswith("_orig") and not args.allow_orig:
            print(f"[run] {n}: the original platform cannot show the inconsistent operator: without rotation "
                  f"control the attentive operator rescues a twisted cuboid in only 13-17 % of the episodes, so "
                  f"the conditions barely differ (docs/inconsistent_operator.md). Pass --allow_orig to run it anyway.")
            return 2
        if n.startswith("C_") and not any(m.startswith("active") for m in sw["methods"]) and not args.without_active:
            print(f"[run] {n}: the best asking arm is not in `methods` yet (see the config's comment). Add it, or pass "
                  f"--without_active to run baseline and ensemble extended only.")
            return 2
        msg = score_mismatch(root, n) if SUITE == "inconsistent_operator" and not args.score_as_is else None
        if msg:
            print(f"[run] {n}: {msg} Pass --score_as_is to run it anyway.")
            return 2
        if sw["channel"] and limits is None and not args.default_limits:
            print(f"[run] {n}: no stage-0 limits ({root / 'stage0' / 'limits.json'}). Run `stage0` first, or pass "
                  f"--default_limits to use the shape defaults.")
            return 2
        camp = root / n
        fp_path = camp / "suite_setting.json"
        fp = fingerprint(n, limits)
        if fp_path.exists():
            old = json.loads(fp_path.read_text(encoding="utf-8"))
            if {k: old.get(k) for k in fp} != fp:
                print(f"[run] {n}: {camp} was started with another config or other limits:\n  then {old}\n  now  {fp}\n"
                      f"Never mix them in one folder: use another --root, or move the old folder away.")
                return 2
        par = int(getattr(args, "parallel", 1) or 1)
        print(f"\n[run] {n}: " + " ".join(_q(c) for c in cmd))
        if par > 1:
            prep, shards, ana = parallel_cmds(cmd, par, getattr(args, "port_base", 6100))
            for c, env in shards:
                print(f"[run]   shard: AQI_TRAINER_PORT={env['AQI_TRAINER_PORT']} " + " ".join(_q(x) for x in c[-10:]))
        if args.dry:
            continue
        camp.mkdir(parents=True, exist_ok=True)
        if not fp_path.exists():
            fp_path.write_text(json.dumps(dict(fp, started=now(), machine=socket.gethostname()), indent=2),
                               encoding="utf-8")
        t0 = now()
        if par > 1:
            rc = run_parallel(cmd, par, n, root, getattr(args, "port_base", 6100))
        else:
            rc = sh(cmd, log=root / f"{n}.out", sim_env=False)   # run_campaign sources sim_env.sh itself
        with open(root / "suite_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"setting": n, "start": t0, "end": now(), "rc": rc, "parallel": par,
                                "machine": socket.gethostname()}) + "\n")
        if auto_export(args):
            export_results(args, quiet=True)      # results/ follows every finished campaign
        if rc != 0 and not args.keep_going:
            print(f"[run] {n} failed (rc {rc}); stopping. Re-run the same command to resume (finished cells are kept).")
            return rc
    return 0


# ===================================================================== audit (Q1)
def cmd_audit(args):
    """Score validity on the inconsistent operator: examples/inconsistent_operator_audit.py per setting."""
    root = Path(args.root)
    rc_all = 0
    for n in resolve(args.names or GROUPS.get("audit") or GROUPS.get("q1", [])):
        camp = root / n
        if not (camp / "aqi_logs").exists():
            print(f"[audit] {n}: no aqi_logs yet under {camp}")
            rc_all |= 1
            continue
        cmd = [sys.executable, str(HERE / "inconsistent_operator_audit.py"), "--campaign", str(camp)]
        if SUITE in ("study", "final", "extension", "phase3"):  # the scores are computed in the quality arms only
            cmd += ["--modes", "quality"]
        rc_all |= sh(cmd, log=root / f"{n}.audit.log", sim_env=False)
        if SUITE in ("final", "extension", "phase3"):  # again where the robot asks: it picks the onset, no timing confound
            rc_all |= sh([sys.executable, str(HERE / "inconsistent_operator_audit.py"), "--campaign", str(camp),
                          "--modes", "aqi", "--out", str(camp / "report" / "audit_robot_asks")],
                         log=root / f"{n}.audit_robot_asks.log", sim_env=False)
    return rc_all


# ===================================================================== report
def _jsonl(path):
    out = []
    if Path(path).exists():
        for line in open(path, encoding="utf-8", errors="replace"):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def actor_minutes(log):
    """Elapsed wall time of the actor from its last progress-bar update (tqdm '[h:mm:ss<' or '[mm:ss<')."""
    if not Path(log).exists():
        return None
    hits = re.findall(r"\[(\d+(?::\d+){1,2})<", Path(log).read_text(encoding="utf-8", errors="replace"))
    if not hits:
        return None
    parts = [int(x) for x in hits[-1].split(":")]
    sec = parts[-1] + 60 * parts[-2] + (3600 * parts[-3] if len(parts) == 3 else 0)
    return round(sec / 60.0, 1)


def cell_metrics(camp, cell_dir):
    camp, cell_dir = Path(camp), Path(cell_dir)
    cid = cell_dir.name
    done = json.loads((cell_dir / ".done").read_text()) if (cell_dir / ".done").exists() else {}
    eps = _jsonl(cell_dir / "episodes.jsonl")
    early = [e for e in eps if int(e.get("step", 0)) <= EARLY]
    unaided = [int(e["step"]) for e in eps if e.get("success") and int(e.get("intervention_count", 0)) == 0]
    ch, hb = {}, {}
    for e in early:
        for k, v in (e.get("takeover_channels") or {}).items():
            ch[k] = ch.get(k, 0) + v
        for k, v in (e.get("handbacks") or {}).items():
            hb[k] = hb.get(k, 0) + v
    n_tk, n_hb = sum(ch.values()), sum(hb.values())
    steps = sum(int(e.get("length", 0)) for e in eps)
    ivn = sum(int(e.get("intervention_steps", 0)) for e in eps)
    evals = _jsonl(cell_dir / "convergence_evals.jsonl")
    sr = last_sr = None
    cj = camp / "eval_runs" / cid / "cell.json"
    if cj.exists():
        try:
            d = json.loads(cj.read_text())
            sr = d["outcomes"]["sr"]
            # the policy that training stopped on; the ruler SR is the best window of three checkpoints
            last_sr = (d.get("sr_per_checkpoint") or [{}])[-1].get("sr")
        except (KeyError, ValueError):
            sr = None
    m = re.match(r"(.+)_([a-z]+\d+)_s(\d+)$", cid)
    return {
        "cell": cid, "arm": done.get("method") or (m.group(1) if m else ""),
        "operator": done.get("operator") or (m.group(2) if m else ""), "tier": done.get("tier", ""),
        "seed": done.get("seed", m.group(3) if m else ""), "finished": bool(done),
        "first_unaided_success": min(unaided) if unaided else None,
        "converged": bool(done.get("converged")), "converged_step": done.get("converged_step"),
        "episodes": len(eps), "env_steps": steps,
        "operator_share": round(ivn / steps, 3) if steps else None,
        "early_timeouts": round(sum(int(e.get("length", 0)) >= EPISODE_LIMIT for e in early) / len(early), 3) if early else None,
        "early_success": round(sum(bool(e.get("success")) for e in early) / len(early), 3) if early else None,
        "early_takeover_episodes": round(sum(int(e.get("intervention_count", 0)) > 0 for e in early) / len(early), 3)
        if early else None,
        "early_orientation_takeovers": round(ch.get("orientation", 0) / n_tk, 3) if n_tk else None,
        "early_time_cap_handbacks": round(hb.get("time_cap", 0) / n_hb, 3) if n_hb else None,
        "early_channels": ch, "mini_evals": len(evals), "last_mini_eval_sr": evals[-1]["sr"] if evals else None,
        "ruler_sr": sr, "ruler_last_sr": last_sr,
        "actor_minutes": actor_minutes(camp / "logs" / f"{cid}.actor.log"),
    }


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def _median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = len(xs) // 2
    return xs[k] if len(xs) % 2 else (xs[k - 1] + xs[k]) / 2


def campaign_machine(camp):
    for p in (Path(camp) / "suite_setting.json", Path(camp) / "manifest.json"):
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            h = d.get("machine") or (d.get("provenance") or {}).get("host")
            if h:
                return h
    return None


def summarize(name, camp):
    cells = [cell_metrics(camp, d) for d in sorted((Path(camp) / "checkpoints").iterdir()) if d.is_dir()] \
        if (Path(camp) / "checkpoints").exists() else []
    sw = switches(load_yaml(config_path(name)))
    by_arm = {}
    for c in cells:
        by_arm.setdefault(c["arm"], []).append(c)
    rows = []
    for arm, cs in by_arm.items():
        fus = [c["first_unaided_success"] for c in cs]
        per_op = {}
        for c in cs:
            per_op.setdefault(c["operator"], []).append(c["ruler_sr"])
        rows.append({
            "setting": name, "arm": arm, "cells": len(cs), "finished": sum(c["finished"] for c in cs),
            "rotation": sw["rotation"], "box": bool(sw["box"]), "channel": sw["channel"],
            "convergence": sw["convergence"], "utd_ratio": sw["utd_ratio"],
            "first_unaided_success_median": _median(fus),
            "never_unaided": sum(x is None for x in fus),
            "converged": f"{sum(c['converged'] for c in cs)}/{len(cs)}",
            "converged_step_mean": _mean([c["converged_step"] for c in cs]),
            "early_timeouts": _mean([c["early_timeouts"] for c in cs]),
            "early_success": _mean([c["early_success"] for c in cs]),
            "operator_share": _mean([c["operator_share"] for c in cs]),
            "early_orientation_takeovers": _mean([c["early_orientation_takeovers"] for c in cs]),
            "early_time_cap_handbacks": _mean([c["early_time_cap_handbacks"] for c in cs]),
            "ruler_sr": _mean([c["ruler_sr"] for c in cs]),
            "ruler_last_sr": _mean([c["ruler_last_sr"] for c in cs]),
            "actor_minutes": _mean([c["actor_minutes"] for c in cs]),
            "ruler_sr_by_operator": {op: _mean(v) for op, v in per_op.items()},   # over the seeds
        })
    return rows, cells, campaign_machine(camp)


def sign_test_p(wins, losses):
    """Two-sided exact sign test over the non-tied pairs."""
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired(cells, setting, reference):
    """Ruler SR of `setting` against `reference`, cell by cell (same arm, operator and seed)."""
    ref = {c["cell"]: c for c in cells if c["setting"] == reference and c["ruler_sr"] is not None}
    out = {}
    for c in cells:
        if c["setting"] != setting or c["ruler_sr"] is None or c["cell"] not in ref:
            continue
        out.setdefault(c["arm"], []).append(c["ruler_sr"] - ref[c["cell"]]["ruler_sr"])
    rows = []
    for arm, d in sorted(out.items()):
        w, l_ = sum(x > 1e-9 for x in d), sum(x < -1e-9 for x in d)
        rows.append({"setting": setting, "reference": reference, "arm": arm, "pairs": len(d),
                     "mean_delta_sr": round(sum(d) / len(d), 3), "wins": w, "losses": l_,
                     "ties": len(d) - w - l_, "sign_test_p": sign_test_p(w, l_)})
    return rows


def paired_arms(cells, setting, ref_arm):
    """Ruler SR of every arm of `setting` against its `ref_arm`, pair by pair (same operator and seed)."""
    key = lambda c: (c["operator"], str(c["seed"]))
    ref = {key(c): c for c in cells if c["setting"] == setting and c["arm"] == ref_arm and c["ruler_sr"] is not None}
    if not ref:
        return []
    out = {}
    for c in cells:
        if c["setting"] != setting or c["arm"] == ref_arm or c["ruler_sr"] is None or key(c) not in ref:
            continue
        out.setdefault(c["arm"], []).append(c["ruler_sr"] - ref[key(c)]["ruler_sr"])
    rows = []
    for arm, d in sorted(out.items()):
        w, l_ = sum(x > 1e-9 for x in d), sum(x < -1e-9 for x in d)
        rows.append({"setting": setting, "reference": ref_arm, "arm": arm, "pairs": len(d),
                     "mean_delta_sr": round(sum(d) / len(d), 3), "wins": w, "losses": l_,
                     "ties": len(d) - w - l_, "sign_test_p": sign_test_p(w, l_)})
    return rows


def _fmt(v):
    if v is None:
        return "–"
    if isinstance(v, bool):
        return "x" if v else "-"
    if isinstance(v, float):
        return f"{v:.2f}" if abs(v) < 10 else f"{v:.0f}"
    return str(v)


def cmd_report(args):
    root = Path(args.root)
    allrows, allcells, machines = [], [], {}
    for n in resolve(args.names or ORDER):
        camp = root / n
        if not (camp / "checkpoints").exists():
            continue
        rows, cells, mach = summarize(n, camp)
        allrows += rows
        allcells += [dict(c, setting=n) for c in cells]
        machines[n] = mach
    if not allrows:
        print(f"[report] nothing trained yet under {root}")
        return 1
    cols = ["setting", "arm", "finished", "rotation", "box", "channel", "convergence", "utd_ratio",
            "first_unaided_success_median", "never_unaided", "converged", "converged_step_mean",
            "early_timeouts", "early_success", "operator_share", "early_orientation_takeovers",
            "early_time_cap_handbacks", "ruler_sr", "ruler_last_sr", "actor_minutes"]
    head = ["setting", "arm", "done", "rot", "box", "chan", "conv", "utd", "first unaided (median)",
            "never unaided", "conv.", "conv. step", "timeouts ≤10k", "success ≤10k", "operator share",
            "orient. takeovers", "time-cap hand-backs", "ruler SR", "last ckpt SR", "actor min"]
    L = [f"# Synthetic-operator test suite – report", "", f"Generated {now()} from `{root}`.", ""]
    distinct = sorted({m for m in machines.values() if m})
    if len(distinct) > 1:
        L += [f"**Warning: the campaigns ran on different machines ({', '.join(distinct)}).** Updates per env step "
              f"differ between machines (finding F3); compare only settings from the same machine, or settings that "
              f"all use the fixed update ratio.", ""]
    L += ["Read **first unaided success** (env steps to the first success without any takeover) and the **ruler SR** "
          "first: both mean the same under every convergence rule. Convergence steps compare only between settings "
          "with the same rule (rolling vs mini-eval). Early measures cover env steps ≤ 10 000; the takeover channels "
          "need the logging, which every suite config enables.", "",
          "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in allrows:
        L.append("| " + " | ".join(_fmt(r[c]) for c in cols) + " |")
    ref = next((r for r in allrows if r["setting"] == "L0_original"), None)
    if ref:
        L += ["", "## Against the original operator (L0)", "",
              "| setting | Δ first unaided success | Δ timeouts ≤10k | Δ ruler SR |", "|---|---|---|---|"]
        for r in allrows:
            if r["setting"] == "L0_original" or r["arm"] != ref["arm"]:
                continue
            d = lambda k: (None if r[k] is None or ref[k] is None else r[k] - ref[k])
            L.append(f"| {r['setting']} | {_fmt(d('first_unaided_success_median'))} | {_fmt(d('early_timeouts'))} | "
                     f"{_fmt(d('ruler_sr'))} |")
    prs = []
    for r in sorted({r["setting"] for r in allrows}):
        refname = "C_reference" if r.startswith("C_") else "L0_original"
        if r != refname:
            prs += paired(allcells, r, refname)
    arm_pairs = []
    for n in sorted({r["setting"] for r in allrows}):
        for i, ref in enumerate(REF_ARMS):       # each pair once: not against an earlier reference
            arm_pairs += [p for p in paired_arms(allcells, n, ref) if p["arm"] not in REF_ARMS[:i]]
    if arm_pairs:
        L += ["", "## Paired ruler SR within a campaign (same operator and seed)", "",
              "| setting | arm | against | pairs | mean Δ SR | wins / losses / ties | sign test p |",
              "|---|---|---|---|---|---|---|"]
        for p in arm_pairs:
            L.append(f"| {p['setting']} | {p['arm']} | {p['reference']} | {p['pairs']} | {_fmt(p['mean_delta_sr'])} | "
                     f"{p['wins']} / {p['losses']} / {p['ties']} | {_fmt(p['sign_test_p'])} |")
    if prs:
        L += ["", "## Paired ruler SR (same arm, operator and seed)", "",
              "| setting | reference | arm | pairs | mean Δ SR | wins / losses / ties | sign test p |",
              "|---|---|---|---|---|---|---|"]
        for p in prs:
            L.append(f"| {p['setting']} | {p['reference']} | {p['arm']} | {p['pairs']} | {_fmt(p['mean_delta_sr'])} | "
                     f"{p['wins']} / {p['losses']} / {p['ties']} | {_fmt(p['sign_test_p'])} |")
    L += ["", "## Ruler SR per operator", ""]
    for r in allrows:
        L.append(f"- {r['setting']} / {r['arm']}: " + ", ".join(f"{k} {_fmt(v)}" for k, v in sorted(r["ruler_sr_by_operator"].items())))
    L += ["", "One cell per tier (pilots) gives the direction and size of an effect, not its significance."]
    (root / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    keys = ["setting"] + [k for k in allcells[0] if k not in ("setting", "early_channels")] + ["early_channels"]
    with open(root / "report_cells.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for c in allcells:
            w.writerow({k: (json.dumps(c[k]) if isinstance(c.get(k), dict) else c.get(k)) for k in keys})
    print("\n".join(L[:12 + len(allrows)]))
    print(f"\n[report] wrote {root / 'report.md'} and {root / 'report_cells.csv'}")
    return 0


# ===================================================================== plots
STUDY_SCORES = [   # (label in inconsistent_audit.json, short name) for the score-validity figure
    ("B = z(dV)-z(t)-z(F)", "behavioural"), ("q_int (logged)", "score as logged"), ("-t_int", "length"),
    ("d_value", "value gain"), ("-f_peak", "peak force"), ("a_hind (hindsight)", "hindsight"),
    ("a_hat_mean", "counterfactual"), ("outcome (episode success)", "outcome"),
    ("-jerk_mean", "smoothness (circular)")]


def _arm_K(arm):
    m = re.fullmatch(r"ens-k(\d+)", arm)
    return 2 if arm == "hilserl" else (int(m.group(1)) if m else None)


def plot_campaign(name, cells, methods, icc, audit, out, plt):
    """The study figures of one campaign from its cell metrics (summarize), the per-arm ICC of its report and its
    score audit. Returns the written paths. Arms are drawn in the config's order; a dot is one cell."""
    made = []
    by = {a: [c for c in cells if c["arm"] == a] for a in methods}
    by = {a: cs for a, cs in by.items() if cs}
    if not by:
        return made
    arms = list(by)

    def dots(ax, key, scale=1.0, label=""):
        for i, a in enumerate(arms):
            v = [c[key] * scale for c in by[a] if c.get(key) is not None]
            if v:
                ax.bar(i, sum(v) / len(v), color="#9fc5e8", width=0.6, zorder=1)
                ax.scatter([i + 0.12 * ((k % 5) - 2) / 2 for k in range(len(v))], v, s=14, color="#1f4e79", zorder=2)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels(arms, rotation=35, ha="right", fontsize=9)
        ax.set_title(label, fontsize=11)
        ax.grid(axis="y", alpha=0.3)

    fig, axs = plt.subplots(1, 4, figsize=(17, 4.2))
    dots(axs[0], "ruler_sr", 100, "ruler success (%)")
    dots(axs[1], "env_steps", 1e-3, "env steps when training stopped (thousands)")
    dots(axs[2], "first_unaided_success", 1e-3, "first success without help (thousands of steps)")
    dots(axs[3], "operator_share", 100, "operator share of the steps (%)")
    fig.suptitle(f"{name}: every arm (bar = mean, dot = one cell)", fontsize=12)
    fig.tight_layout()
    p = out / f"{name}_arms.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(p)

    ks = sorted({_arm_K(a) for a in arms if _arm_K(a) is not None})
    if len(ks) >= 2:
        fig, axs = plt.subplots(1, 4, figsize=(17, 4.2))
        for ax, (key, scale, lab) in zip(axs, (("ruler_sr", 100, "ruler success (%)"),
                                                 ("env_steps", 1e-3, "env steps when training stopped (thousands)"),
                                                 ("operator_share", 100, "operator share (%)"),
                                                 ("actor_minutes", 1, "actor minutes per cell"))):
            for a in arms:
                k = _arm_K(a)
                if k is None:
                    continue
                v = [c[key] * scale for c in by[a] if c.get(key) is not None]
                if v:
                    ax.scatter([k] * len(v), v, s=16, color="#1f4e79", alpha=0.6)
                    ax.scatter([k], [sum(v) / len(v)], s=90, marker="_", color="#c00000", linewidths=2.5)
            ax.set_xscale("log", base=2)
            ax.set_xticks(ks)
            ax.set_xticklabels([("2 (HIL-SERL)" if k == 2 else str(k)) for k in ks])
            ax.set_xlabel("number of critics K")
            ax.set_title(lab, fontsize=11)
            ax.grid(alpha=0.3)
        fig.suptitle(f"{name}: number of critics (red = mean)", fontsize=12)
        fig.tight_layout()
        p = out / f"{name}_critics.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        made.append(p)

    ask = [a for a in arms if a.startswith(("asks-", "random-")) or a in ("ens-k8", "hilserl")]
    if any(a.startswith("asks-") for a in ask):
        fig, ax = plt.subplots(figsize=(7.5, 5))
        for a in ask:
            sr = [c["ruler_sr"] * 100 for c in by[a] if c.get("ruler_sr") is not None]
            sh_ = [c["operator_share"] * 100 for c in by[a] if c.get("operator_share") is not None]
            if not sr or not sh_:
                continue
            x, y = sum(sh_) / len(sh_), sum(sr) / len(sr)
            col = "#c00000" if a.startswith("asks-") else ("#7f7f7f" if a.startswith("random-") else "#1f4e79")
            ax.scatter([x], [y], s=70, color=col, zorder=3)
            lab = a + (f"  ICC {icc[a]:.2f}" if icc.get(a) is not None else "")
            ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)
        ax.set_xlabel("operator share of the steps (%)  = human effort")
        ax.set_ylabel("ruler success (%)")
        ax.set_title(f"{name}: success against human effort (ICC = share of the spread from the operator)",
                     fontsize=10)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = out / f"{name}_asking.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        made.append(p)

    sc = (audit or {}).get("scores") or {}
    have = [(k, s) for k, s in STUDY_SCORES if k in sc and (sc[k].get("auroc_within") or {}).get("all") is not None]
    if have:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        vals = [sc[k]["auroc_within"]["all"] for k, _s in have]
        ax.bar(range(len(have)), vals, color=["#1f4e79" if v >= 0.7 else "#9fc5e8" for v in vals])
        ax.axhline(0.5, color="#7f7f7f", ls="--", lw=1)
        ax.axhline(0.7, color="#c00000", ls="--", lw=1)
        ax.text(len(have) - 0.5, 0.705, "bar 0.70", color="#c00000", ha="right", fontsize=9)
        ax.text(len(have) - 0.5, 0.505, "chance 0.50", color="#7f7f7f", ha="right", fontsize=9)
        ax.set_xticks(range(len(have)))
        ax.set_xticklabels([s for _k, s in have], rotation=30, ha="right", fontsize=9)
        ax.set_ylim(0.3, 1.0)
        ax.set_ylabel("AUROC within run, phase and onset")
        ax.set_title(f"{name}: does a score find the careless takeovers? "
                     f"({audit.get('n_segments', '?')} takeovers, {audit.get('n_runs', '?')} runs)", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        p = out / f"{name}_scores.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        made.append(p)
    return made


def cmd_plots(args):
    root = Path(args.root)
    out = root / "plots"
    out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    made = []
    for n in resolve(args.names or ORDER):
        camp = root / n
        if not (camp / "checkpoints").exists():
            continue
        _rows, cells, _m = summarize(n, camp)
        methods = list(load_yaml(config_path(n)).get("methods") or [])
        dep_p, aud_p = camp / "report" / "dependency.json", camp / "report" / "inconsistent_audit.json"
        dep = json.loads(dep_p.read_text(encoding="utf-8")) if dep_p.exists() else {}
        icc = {a: (m or {}).get("icc") for a, m in (dep.get("methods") or {}).items()}
        audit = json.loads(aud_p.read_text(encoding="utf-8")) if aud_p.exists() else None
        made += plot_campaign(n, cells, methods, icc, audit, out, plt)
    for p in made:
        print(f"[plots] wrote {p}")
    if not made:
        print(f"[plots] nothing trained yet under {root}")
    return 0 if made else 1


# ===================================================================== export to results/
EXPORT_SKIP_DIRS = ("checkpoint_", "certified", "demos", "wandb", "__pycache__", "buffer", "_dryrun", "ocdbt.process_")
EXPORT_SKIP_FILES = (".pkl", ".npz", ".npy", ".msgpack", ".ckpt", ".pt", ".pth", ".tgz", ".zip")
EXPORT_MAX_BYTES = 25 * 2 ** 20          # anything bigger is listed in the README, not copied
LOG_HEAD, LOG_TAIL = 60, 1500             # lines kept of a long log after the progress bars are compacted


def default_results(root):
    """results/<suite> for the default runs/<suite>, else <root>/results (an exported folder stays where it is)."""
    root = Path(root).resolve()
    return REPO / "results" / SUITE if root == Path(DEFAULT_ROOT).resolve() else root.parent / "results" / root.name


def compact_log(text):
    """A log as a reader needs it: every progress-bar line reduced to its last update (tqdm redraws with \\r), and a
    long log cut to its head and tail, which hold the command, the settings and how it ended."""
    lines = [ln.split("\r")[-1] for ln in text.replace("\r\n", "\n").split("\n")]
    lines = [ln for ln in lines if ln.strip() or not lines]
    if len(lines) > LOG_HEAD + LOG_TAIL:
        cut = len(lines) - LOG_HEAD - LOG_TAIL
        lines = lines[:LOG_HEAD] + [f"[... {cut} lines left out by the export ...]"] + lines[-LOG_TAIL:]
    return "\n".join(lines) + "\n"


def _skip_dir(name):
    return any(name.startswith(p) for p in EXPORT_SKIP_DIRS)


def export_tree(src, dst, with_steps=False):
    """Mirror `src` into `dst` without the heavy files: checkpoints, demonstrations, W&B folders, per-step logs
    (steps.jsonl, unless with_steps) and anything over EXPORT_MAX_BYTES; logs are compacted. The layout stays the
    same, so report / plots / audit run on the exported folder. Copies only what changed. Returns
    (files written, bytes kept, [(skipped path, bytes)])."""
    src, dst = Path(src), Path(dst)
    written, kept, skipped = 0, 0, []
    for dirpath, dirnames, filenames in os.walk(src):
        rel = Path(dirpath).relative_to(src)
        for d in list(dirnames):
            if _skip_dir(d):
                dirnames.remove(d)
                p = Path(dirpath) / d
                size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                skipped.append((str(rel / d) + "/", size))
        for fn in filenames:
            s = Path(dirpath) / fn
            t = dst / rel / fn
            size = s.stat().st_size
            heavy = fn.endswith(EXPORT_SKIP_FILES) or (fn == "steps.jsonl" and not with_steps) or size > EXPORT_MAX_BYTES
            if heavy and not fn.endswith((".log", ".out")):
                skipped.append((str(rel / fn), size))
                continue
            if t.exists() and t.stat().st_mtime >= s.stat().st_mtime:
                kept += t.stat().st_size
                continue
            t.parent.mkdir(parents=True, exist_ok=True)
            if fn.endswith((".log", ".out")):
                # newline="" keeps tqdm's \r redraws (universal newlines would turn each into a line)
                with open(s, encoding="utf-8", errors="replace", newline="") as f:
                    text = f.read()
                with open(t, "w", encoding="utf-8", newline="\n") as f:
                    f.write(compact_log(text))
                os.utime(t, (s.stat().st_atime, s.stat().st_mtime))
            else:
                shutil.copy2(s, t)
            written += 1
            kept += t.stat().st_size
    return written, kept, skipped


def export_results(args, quiet=False):
    """Collect what the analysis needs into one folder (default results/<suite>): the suite's reports, plots,
    machine check and stage 0, and every campaign's reports, ruler outcomes, per-cell training records, takeover logs
    and compacted logs. No checkpoints, demonstrations or per-step logs. Writes README.md and export.json there."""
    root = Path(args.root)
    out = Path(getattr(args, "out", None) or getattr(args, "results", None) or default_results(root))
    if not root.exists():
        if not quiet:
            print(f"[export] nothing under {root} yet")
        return 1
    out.mkdir(parents=True, exist_ok=True)
    with_steps = bool(getattr(args, "with_steps", False))
    written, kept, skipped = export_tree(root, out, with_steps=with_steps)
    camps = []
    for n in ORDER:
        camp = root / n
        if not camp.exists():
            continue
        done, seen = cell_status(camp)
        rep = camp / "report"
        camps.append({"campaign": n, "cells_done": done, "cells_seen": seen,
                      "report": (rep / "dependency.json").exists(), "audit": (rep / "inconsistent_audit.json").exists(),
                      "ruler_check": (rep / "ruler_reselect.json").exists()})
    mach = {}
    if (root / "machine.json").exists():
        try:
            mach = json.loads((root / "machine.json").read_text(encoding="utf-8"))
        except ValueError:
            mach = {}
    skipped_bytes = sum(b for _p, b in skipped)
    info = {"exported": now(), "suite": SUITE, "source": str(root), "host": socket.gethostname(),
            "machine_id": mach.get("machine_id"), "git_commit": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain")), "campaigns": camps, "files_written": written,
            "bytes_kept": kept, "skipped_items": len(skipped), "bytes_skipped": skipped_bytes,
            "with_steps": with_steps}
    (out / "export.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    L = [f"# Results of the {SUITE} suite", "",
         f"Exported {info['exported']} from `{root}` on {info['host']} ({info['machine_id'] or 'machine check not run'}), "
         f"commit `{info['git_commit'][:8]}`{' (with local changes)' if info['git_dirty'] else ''}.", "",
         "Everything the analysis needs, in the same layout as the runs folder: reports and plots, ruler outcomes "
         "(`eval_runs/<cell>/cell.json`, `trials.jsonl`), training records (`checkpoints/<cell>/episodes.jsonl`, "
         "`convergence_evals.jsonl`, `.done`), takeover logs (`aqi_logs/<run>/segments.jsonl`, `episodes.jsonl`, "
         "`meta.json`) and compacted logs. Left out: checkpoints, demonstrations, W&B folders"
         + ("" if with_steps else ", per-step logs (`steps.jsonl`)") + f" ({len(skipped)} items, "
         f"{skipped_bytes / 2 ** 30:.1f} GB).", "",
         "Re-analyse it anywhere with the same driver, e.g. "
         f"`python examples/synthetic_suite.py --suite {SUITE} --root {out.name if out.parent.name == 'results' else out} report`.",
         "", "| campaign | cells done | report | score audit | ruler check |", "|---|---|---|---|---|"]
    for c in camps:
        L.append(f"| {c['campaign']} | {c['cells_done']}/{c['cells_seen']} | {'x' if c['report'] else '-'} | "
                 f"{'x' if c['audit'] else '-'} | {'x' if c['ruler_check'] else '-'} |")
    (out / "README.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    if not quiet:
        print(f"[export] {out}: {written} files written, {kept / 2 ** 20:.1f} MB in the folder; left out "
              f"{len(skipped)} heavy items ({skipped_bytes / 2 ** 30:.1f} GB)")
    return 0


def auto_export(args):
    """Refresh the results folder after a command, unless --no_export, a dry run, or the root is itself an export
    (re-analysing a results folder writes into it, not into a copy of it)."""
    return not (getattr(args, "no_export", False) or getattr(args, "dry", False)
                or (Path(args.root) / "export.json").exists())


def cmd_export(args):
    return export_results(args)


def cmd_rulercheck(args):
    """Would the shorter ruler change a score? ruler_reselect.py on campaigns scored with the full ruler (the smoke
    campaigns), written to <campaign>/report/ruler_reselect.{txt,json}, so it reaches the export."""
    root = Path(args.root)
    rc_all = 0
    for n in resolve(args.names or GROUPS.get("smoke", [])):
        camp = root / n
        if not (camp / "eval_runs").exists():
            print(f"[rulercheck] {n}: nothing scored yet under {camp}")
            rc_all |= 1
            continue
        (camp / "report").mkdir(parents=True, exist_ok=True)
        (camp / "report" / "ruler_reselect.txt").unlink(missing_ok=True)
        rc_all |= sh([sys.executable, str(HERE / "ruler_reselect.py"), "--campaign", str(camp), "--skip_pretrain",
                      "--newest", "8", "--json", str(camp / "report" / "ruler_reselect.json")],
                     log=camp / "report" / "ruler_reselect.txt", sim_env=False)
    return rc_all


# ===================================================================== unattended: everything in one go
SINGLE_CELL_STEPS_PER_S = 6.2     # one 8-critic cell alone on the earlier RTX 5090 machine: 6.1-6.5 env steps/s
PARALLEL_MIN_GAIN = 1.15          # side by side must give at least 15 % more steps/s in all than one cell alone
K8_ARMS = ("asks-", "random-", "ens-k8", "keepall", "beh-", "oracle-", "perfect-", "aqi-")
_EXPORT_LOCK = threading.Lock()


def _ns(args, **kw):
    """Arguments for one step of `all` (the sub-steps never export themselves; `all` does, under a lock)."""
    base = dict(root=args.root, results=getattr(args, "results", None), no_export=True, dry=False)
    base.update(kw)
    return types.SimpleNamespace(**base)


class RunStatus:
    """What `all` did, decided and noticed: <root>/RUN_STATUS.md and .json, rewritten after every step (and so in
    the results folder)."""

    def __init__(self, root):
        self.root = Path(root)
        self.started, self.state = now(), "running"
        self.steps, self.decisions, self.warnings = [], {}, []

    def write(self):
        d = {"started": self.started, "updated": now(), "state": self.state, "host": socket.gethostname(),
             "git_commit": git("rev-parse", "HEAD"), "steps": self.steps, "decisions": self.decisions,
             "warnings": self.warnings}
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "RUN_STATUS.json").write_text(json.dumps(d, indent=2), encoding="utf-8")
        L = [f"# Unattended run of the {SUITE} suite: {self.state}", "",
             f"Started {self.started} on {d['host']} (commit `{d['git_commit'][:8]}`), updated {d['updated']}.", "",
             "## Steps", "", "| step | start | end | exit code | note |", "|---|---|---|---|---|"]
        L += [f"| {s['step']} | {s['start']} | {s['end']} | {s['rc']} | {s.get('note', '')} |" for s in self.steps]
        L += ["", "## Decisions taken without stopping", ""]
        L += [f"- **{k}:** {v}" for k, v in self.decisions.items()] or ["- none yet"]
        L += ["", "## Warnings", ""] + ([f"- {w}" for w in self.warnings] or ["- none"])
        (self.root / "RUN_STATUS.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def _step(st, name, fn, a, note=""):
    """Run one step; record its exit code; never stop (a crash becomes exit code 99 and a warning)."""
    t0 = now()
    print(f"\n[all] ===== {name} ({t0}) =====", flush=True)
    try:
        rc = fn(a)
    except KeyboardInterrupt:
        raise
    except BaseException as exc:                       # SystemExit included: the run goes on
        traceback.print_exc()
        rc = exc.code if isinstance(exc, SystemExit) and isinstance(exc.code, int) and exc.code else 99
        note = (note + "; " if note else "") + f"stopped: {type(exc).__name__}: {str(exc)[:200]}"
    rc = int(rc or 0)
    st.steps.append({"step": name, "start": t0, "end": now(), "rc": rc, "note": note})
    if rc:
        st.warnings.append(f"{name}: exit code {rc}" + (f" ({note})" if note else ""))
    st.write()
    return rc


def ruler_changes(root, names):
    """(cells compared, cells whose score the shorter ruler changes) from <campaign>/report/ruler_reselect.json."""
    n_cmp = n_chg = 0
    for n in names:
        p = Path(root) / n / "report" / "ruler_reselect.json"
        if not p.exists():
            continue
        for _c, r in json.loads(p.read_text(encoding="utf-8")).items():
            if r.get("old") is None or r.get("new") is None:
                continue
            n_cmp += 1
            n_chg += abs(float(r["new"]) - float(r["old"])) > 1e-9
    return n_cmp, n_chg


def parallel_decision(root, names, requested):
    """How many cells side by side for the main campaigns, from the smoke campaigns that ran `requested` side by side:
    keep them if their 8-critic cells' env steps per second, times `requested`, beat one cell alone
    (SINGLE_CELL_STEPS_PER_S) by PARALLEL_MIN_GAIN; else one at a time. Returns (n, reason)."""
    if requested <= 1:
        return 1, "one cell at a time was requested"
    rates, done, seen = [], 0, 0
    for n in names:
        camp = Path(root) / n
        if not (camp / "checkpoints").exists():
            continue
        _rows, cells, _m = summarize(n, camp)
        for c in cells:
            seen += 1
            done += bool(c["finished"])
            if c["arm"].startswith(K8_ARMS) and c.get("env_steps") and c.get("actor_minutes"):
                rates.append(c["env_steps"] / (c["actor_minutes"] * 60.0))
    if seen == 0 or done < max(1, (seen + 1) // 2):
        return 1, f"only {done} of {seen} smoke cells finished side by side: one at a time is safer"
    if not rates:
        return requested, f"no 8-critic smoke cell had a rate; keeping {requested} side by side"
    r = sum(rates) / len(rates)
    total, need = r * requested, SINGLE_CELL_STEPS_PER_S * PARALLEL_MIN_GAIN
    if total < need:
        return 1, (f"{requested} side by side gave {r:.1f} env steps/s per cell, {total:.1f} in all, below "
                   f"{need:.1f} (one alone about {SINGLE_CELL_STEPS_PER_S:g})")
    return requested, (f"{requested} side by side gave {r:.1f} env steps/s per cell, {total:.1f} in all "
                       f"(one alone about {SINGLE_CELL_STEPS_PER_S:g})")


def push_results(results_dir, branch="study-results", remote_url=None, log=print):
    """Commit the results folder in its own small git repository (<results>/.git, so the code checkout is untouched)
    and force-push it to `branch` of the code repository's remote. Never prompts (GIT_TERMINAL_PROMPT=0): without
    push credentials on the machine it logs the failure and returns non-zero."""
    d = Path(results_dir)
    url = remote_url or git("remote", "get-url", "origin")
    if not url:
        log("[push] no `origin` remote: skipped")
        return 1
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")

    def g(*a):
        return subprocess.run(["git", "-C", str(d), *a], capture_output=True, text=True, env=env, timeout=900)

    if not (d / ".git").exists():
        g("init", "-q")
        g("checkout", "-q", "-b", branch)
    if not g("config", "user.email").stdout.strip():
        g("config", "user.email", "aqi-study@localhost")
        g("config", "user.name", "AQI study (unattended run)")
    g("add", "-A")
    g("commit", "-q", "-m", f"{SUITE} results {now()}")
    p = g("push", "-q", "--force", url, f"HEAD:refs/heads/{branch}")
    if p.returncode != 0:
        log(f"[push] to {url} ({branch}) failed: {(p.stderr or p.stdout).strip()[-300:]}")
        return p.returncode
    log(f"[push] results -> {url} branch {branch}")
    return 0


def crashed_cells(camp):
    """Cells of a campaign whose actor crashed (.crashed) and that have not finished since (.done)."""
    ck = Path(camp) / "checkpoints"
    if not ck.exists():
        return []
    return sorted(d.name for d in ck.iterdir() if (d / ".crashed").exists() and not (d / ".done").exists())


def disk_decision(root, name, st):
    """Extra run_campaign flags for one campaign's disk: the configs keep the newest checkpoints plus every 2 000-th
    learner step (budget curves); if the free disk is below what that campaign needs, it keeps only the newest
    (budget curves then come from the mini-evaluations). Decided per campaign, recorded; never stops."""
    if "checkpoint_keep_every" not in ((load_yaml(config_path(name)).get("budget") or {})):
        return []
    free = round(shutil.disk_usage(root).free / 2 ** 30, 1)
    need = disk_needed_gb([name])
    key = f"disk {name}"
    if free >= need:
        st.decisions[key] = f"{free} GB free, about {need:g} GB needed: the early checkpoints stay"
        return []
    st.decisions[key] = (f"only {free} GB free, about {need:g} GB needed: newest checkpoints only "
                         f"(--checkpoint_keep_every 0)")
    st.warnings.append(st.decisions[key])
    return ["--checkpoint_keep_every", "0"]


def _refresh(args, st, push):
    """Export (and push) under a lock: called after each step and by the periodic thread."""
    with _EXPORT_LOCK:
        try:
            st.write()
            export_results(_ns(args, with_steps=False), quiet=True)
            if push:
                out = Path(getattr(args, "results", None) or default_results(Path(args.root)))
                if push_results(out, branch=getattr(args, "branch", "study-results"), log=lambda m: print(m, flush=True)):
                    st.decisions["results push"] = ("failing: no push credentials on this machine? The results stay in "
                                                    "results/ there")
        except Exception as exc:                       # never stop the run for an export
            print(f"[all] export/push failed: {exc}", flush=True)


def cmd_all(args):
    """The whole suite in one go, without stopping: check, stage 0, smoke, ruler check, smoke audit and report, then
    the main campaigns, audit, report, plots. Every gate of the runbook becomes a decision recorded in RUN_STATUS.md;
    failures are recorded and the run goes on. The results folder is refreshed (and pushed, with --push) after every
    step and every --push_every_h hours."""
    if SUITE == "extension":            # its own flow: screen, pilots, decision, smoke (never pushes)
        return cmd_extension_all(args)
    if SUITE == "phase3":               # its own flow: probe, smoke, main campaign, careful twin (never pushes)
        return cmd_phase3_all(args)
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    st = RunStatus(root)
    st.decisions["mode"] = "unattended: no step waits for a person; failures are recorded and the run goes on"
    push = bool(getattr(args, "push", False))
    stop = threading.Event()

    def beat():
        while not stop.wait(max(0.1, float(getattr(args, "push_every_h", 6.0))) * 3600.0):
            _refresh(args, st, push)

    threading.Thread(target=beat, daemon=True).start()
    smoke = GROUPS.get("smoke", [])
    main_names = GROUPS.get("main", [n for n in ORDER if n not in smoke])
    # 1. machine check; W&B online only with a key, else offline (kept on the machine)
    _step(st, "check", cmd_check, _ns(args))
    mode = getattr(args, "wandb", None) or ("online" if wandb_logged_in() else "offline")
    st.decisions["W&B"] = mode + ("" if mode == "online" or getattr(args, "wandb", None)
                                  else " (no key on this machine: the runs log locally only)")
    _refresh(args, st, push)
    # 2. stage 0
    _step(st, "stage0", cmd_stage0, _ns(args, skip_tests=False, skip_operator_test=False, skip_sweep=False,
                                       episodes=30, sweep_episodes=10))
    shape = switches(load_yaml(config_path(main_names[0])))["peg_shape"] if main_names else "cuboid"
    default_limits = read_limits(root, shape) is None
    st.decisions["orientation limits"] = ("shape defaults (stage 0 measured none)" if default_limits
                                          else f"stage 0: {read_limits(root, shape)}")
    _refresh(args, st, push)
    req = max(1, int(getattr(args, "parallel", 2)))
    run_kw = dict(wandb=mode, dry=False, keep_going=True, default_limits=default_limits, without_active=True,
                  allow_orig=False, score_as_is=True, port_base=int(getattr(args, "port_base", 6100)))
    # 3. smoke, side by side as requested
    if smoke:
        _step(st, "smoke", cmd_run, _ns(args, names=smoke, parallel=req, extra=[], **run_kw))
        _refresh(args, st, push)
        if SUITES[SUITE].get("ruler_check", True):
            _step(st, "ruler check", cmd_rulercheck, _ns(args, names=smoke))
        audit_smoke = [n for n in GROUPS.get("audit", []) if n in smoke]
        if audit_smoke:
            _step(st, "smoke audit", cmd_audit, _ns(args, names=audit_smoke))
        _step(st, "smoke report", cmd_report, _ns(args, names=smoke))
    # 4. decisions for the main campaigns
    n_cmp, n_chg = ruler_changes(root, smoke)
    extra = []
    if not SUITES[SUITE].get("ruler_check", True):
        n_cmp = n_chg = 0
        st.decisions["ruler"] = "as configured (this suite scores every kept checkpoint; no shorter-ruler check)"
    elif n_chg:
        extra = ["--no-eval_skip_pretrain", "--eval_newest", "0"]
        st.decisions["ruler"] = (f"FULL ruler for the main campaigns: the shorter ruler changed {n_chg} of {n_cmp} smoke "
                                 f"scores")
    else:
        st.decisions["ruler"] = (f"shorter ruler as configured: it changed none of {n_cmp} smoke scores" if n_cmp else
                                 "shorter ruler as configured: no smoke score to compare (0 of 57 changed on the "
                                 "earlier second machine)")
    par, why = parallel_decision(root, smoke, req)
    st.decisions["cells side by side"] = f"{par}: {why}"
    _refresh(args, st, push)
    # 5. the main campaigns, one after another, each side by side as decided
    for n in main_names:
        extra_n = list(extra) + disk_decision(root, n, st)
        _step(st, f"run {n}", cmd_run, _ns(args, names=[n], parallel=par, extra=extra_n, **run_kw),
              note=f"{par} side by side" + ("; full ruler" if "--eval_newest" in extra else "")
              + ("; newest checkpoints only (disk)" if "--checkpoint_keep_every" in extra_n else ""))
        _refresh(args, st, push)
        crashed = crashed_cells(root / n)
        if crashed:                     # once: run_campaign retrains exactly the cells without .done
            st.decisions[f"retry {n}"] = f"{len(crashed)} crashed cell(s) retrained once: {', '.join(crashed)}"
            _step(st, f"retry {n}", cmd_run, _ns(args, names=[n], parallel=par, extra=extra_n, **run_kw))
            left = crashed_cells(root / n)
            if left:
                st.warnings.append(f"{n}: still crashed after one retry, counted as failed: {', '.join(left)}")
            _refresh(args, st, push)
    # 6. analysis
    audit_main = [n for n in GROUPS.get("audit", []) if n in main_names]
    if audit_main:
        _step(st, "audit", cmd_audit, _ns(args, names=audit_main))
    _step(st, "report", cmd_report, _ns(args, names=[]))
    _step(st, "plots", cmd_plots, _ns(args, names=[]))
    stop.set()
    st.state = "finished" + (" with warnings" if st.warnings else "")
    _refresh(args, st, push)
    print(f"\n[all] {st.state}; see {root / 'RUN_STATUS.md'} and the results folder", flush=True)
    return 1 if st.warnings else 0


# ===================================================================== extension study (phase 2)
# docs/extension_study.md. The rules below are fixed before any extension run; `screen` and `decide` apply them to
# the measurements and write the decisions (<root>/screen/screen_decision.json, <root>/decision.json) that the
# pilots, the smoke test and the main study read. `prune` frees the disk of scored runs in the new roots only.
EXT_CANDIDATES = ["hex", "tight", "wide", "hexwide"]
EXT_PILOT = {c: f"P{i + 1}_{c}" for i, c in enumerate(EXT_CANDIDATES)}
EXT_SMOKE = {c: f"S{i + 1}_{c}" for i, c in enumerate(EXT_CANDIDATES)}
SCREEN_TIERS = ["expert", "intermediate", "s=0.3", "s=0.15", "s=0"]
NOVICE_SKILLS = (0.3, 0.15, 0.0)          # candidate novice skills; on a tie the earlier one wins
SCREEN_EPISODES = 30
SCREEN_RULE = {"expert_solo_min": 0.90,           # the careful expert does the task alone ...
               "expert_rescue_min": 0.80,         # ... and rescues a failing policy: corrections can help
               "careless_expert_solo_max": 0.60,  # a careless correction is clearly worse
               "novice_solo_lo": 0.40, "novice_solo_hi": 0.70,  # the operators differ: a novice skill in this band
               "novice_target": 0.55}             # closest to this if several skills are in the band
PILOT_RULE = {"budget_env_steps": 8000,           # keepall with the careful intermediate after 8 000 env steps ...
              "lo": 0.30, "hi": 0.70,             # ... in this band: learnable but not at the ceiling ...
              "final_min": 0.80,                  # ... and >= 80 % evaluation success by the cap
              "fallback_final_min": 0.50, "fallback_cap": 40000,  # none: hardest one still >= 50 %, cap 40 000
              "budget_target": 0.50,              # budget B: where keepall-intermediate is closest to 50 %
              "budgets": list(range(2000, 20001, 2000))}
PRUNE_ROOTS = ("final_twins", "extension", "phase3")   # runs/<name>: the only default roots whose checkpoints are pruned
PRUNE_MARKER = ".prune_scored_ok"                 # any other root needs this file to allow pruning


def _skill_label(s):
    return f"s={float(s):g}"


def ext_task_env(name):
    """(environment, peg shape) of the operator screen for one candidate: the pilot config's task knobs, exactly as
    run_campaign.py sets them for every process of a campaign."""
    task = load_yaml(config_path(name)).get("task") or {}
    shape = str(task.get("peg_shape", "cuboid"))
    # no orientation box, like the final study's operator test (stage 0), whatever the shell exports
    env = {"JAX_PLATFORMS": "cpu", "SIM_SYN_PEG_SHAPE": shape, "SIM_RPY_BOX": ""}
    for key, var in (("success_lateral", "SIM_SUCCESS_LATERAL"), ("random_xy", "SIM_RANDOM_XY"),
                     ("random_rz", "SIM_RANDOM_RZ")):
        env[var] = f"{float(task[key]):g}" if task.get(key) is not None else ""
    return env, shape


def _screen_rate(rows, tier, scenario, condition):
    for r in rows:
        if (r.get("tier") == tier and r.get("scenario") == scenario and r.get("condition") == condition
                and r.get("rotation", "on") == "on"):
            return float(r["success"])
    return None


def screen_verdict(rows, rule=None):
    """The screen rule for one candidate's operator-test rows (rotation control on, as in training)."""
    rule = rule or SCREEN_RULE
    es = _screen_rate(rows, "expert", "solo", "attentive")
    er = _screen_rate(rows, "expert", "rescue", "attentive")
    ec = _screen_rate(rows, "expert", "solo", "inattentive")
    nov = {s: _screen_rate(rows, _skill_label(s), "solo", "attentive") for s in NOVICE_SKILLS}
    reasons = []
    if es is None or es < rule["expert_solo_min"]:
        reasons.append(f"careful expert alone {es} < {rule['expert_solo_min']}")
    if er is None or er < rule["expert_rescue_min"]:
        reasons.append(f"careful expert rescue {er} < {rule['expert_rescue_min']}")
    if ec is None or ec > rule["careless_expert_solo_max"]:
        reasons.append(f"careless expert alone {ec} > {rule['careless_expert_solo_max']}")
    band = [s for s in NOVICE_SKILLS
            if nov[s] is not None and rule["novice_solo_lo"] <= nov[s] <= rule["novice_solo_hi"]]
    # distance rounded so that equal distances (rates are k / episodes) tie exactly; the earlier skill wins a tie
    skill = (min(band, key=lambda s: (round(abs(nov[s] - rule["novice_target"]), 9), NOVICE_SKILLS.index(s)))
             if band else None)
    if skill is None:
        reasons.append(f"no novice skill {list(NOVICE_SKILLS)} succeeds alone in "
                       f"{rule['novice_solo_lo']:g}-{rule['novice_solo_hi']:g}")
    return {"passes": not reasons, "novice_skill": skill, "expert_solo": es, "expert_rescue": er,
            "careless_expert_solo": ec, "novice_solo": {_skill_label(s): v for s, v in nov.items()},
            "reasons": reasons}


def cmd_screen(args):
    """Operator screen of the candidate tasks (CPU only): examples/oracle_insertion_test.py per candidate with its
    task knobs, then the rule. Resumable: a candidate with a result file is not re-run."""
    root = Path(args.root) / "screen"
    root.mkdir(parents=True, exist_ok=True)
    rc_all, verdicts = 0, {}
    for c in EXT_CANDIDATES:
        out = root / f"{c}.json"
        env, shape = ext_task_env(EXT_PILOT[c])
        if not out.exists():
            rc_all |= sh([sys.executable, "examples/oracle_insertion_test.py", "--shapes", shape,
                          "--tiers", ",".join(SCREEN_TIERS), "--rotation", "on", "--scenarios", "solo,rescue",
                          "--conditions", "attentive,inattentive",
                          "--episodes", str(int(getattr(args, "episodes", SCREEN_EPISODES) or SCREEN_EPISODES)),
                          "--out", str(out)], log=root / f"{c}.log", env=env)
        try:
            rows = json.loads(out.read_text(encoding="utf-8")).get("rows", []) if out.exists() else []
        except ValueError:
            rows = []
        v = screen_verdict(rows)
        if not rows:
            v["reasons"].insert(0, "the operator test produced no result (see the log)")
        v.update(pilot=EXT_PILOT[c], smoke=EXT_SMOKE[c], peg_shape=shape,
                 knobs={k: val for k, val in env.items()
                        if k in ("SIM_SUCCESS_LATERAL", "SIM_RANDOM_XY", "SIM_RANDOM_RZ")})
        verdicts[c] = v
    dec = {"created": now(), "rule": SCREEN_RULE, "tiers": SCREEN_TIERS, "candidates": verdicts,
           "passing": [c for c in EXT_CANDIDATES if verdicts[c]["passes"]]}
    (root / "screen_decision.json").write_text(json.dumps(dec, indent=2), encoding="utf-8")
    L = ["# Operator screen (extension study, phase 2)", "",
         f"{now()}. Rotation control on, {getattr(args, 'episodes', SCREEN_EPISODES)} episodes per row. Rule: "
         + ", ".join(f"{k} {v}" for k, v in SCREEN_RULE.items()) + ".", "",
         "| candidate | careful expert alone | rescue | careless expert alone | novice alone (s=0.3 / 0.15 / 0) "
         "| novice skill | passes | why not |", "|---|---|---|---|---|---|---|---|"]
    for c, v in verdicts.items():
        ns = " / ".join(_fmt(v["novice_solo"][_skill_label(s)]) for s in NOVICE_SKILLS)
        L.append(f"| {c} | {_fmt(v['expert_solo'])} | {_fmt(v['expert_rescue'])} | {_fmt(v['careless_expert_solo'])} "
                 f"| {ns} | {_fmt(v['novice_skill'])} | {'yes' if v['passes'] else 'no'} | {'; '.join(v['reasons'])} |")
    L += ["", f"Passing: {', '.join(dec['passing']) or 'none'}."]
    (root / "SCREEN.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("[screen] passing:", ", ".join(dec["passing"]) or "none")
    return rc_all


def _pretrain_steps(man):
    """Warm-start updates of a run (subtracted to get environment steps), from its manifest."""
    if not man:
        return 0
    ov = (man.get("arm") or {}).get("overrides") or {}
    if ov.get("demo_pretrain"):
        return int(ov.get("demo_pretrain_steps", 5000))
    rep = " ".join(str(x) for x in (man.get("reproduce") or []))
    if re.search(r"--demo_pretrain(\s|$)", rep):
        m = re.search(r"--demo_pretrain_steps (\d+)", rep)
        return int(m.group(1)) if m else 5000
    return 0


def cell_curve(camp, cid):
    """([(environment step, ruler success)] per scored checkpoint, evaluation success = best window) of one cell,
    or None while it has no ruler result."""
    camp = Path(camp)
    cj = camp / "eval_runs" / cid / "cell.json"
    if not cj.exists():
        return None
    try:
        d = json.loads(cj.read_text(encoding="utf-8"))
    except ValueError:
        return None
    cps = sorted(d.get("sr_per_checkpoint") or [], key=lambda x: x["step"])
    if not cps:
        return None
    man = {}
    for p in (camp / "checkpoints" / cid / "manifest.json", camp / "eval_runs" / cid / "manifest.json"):
        if p.exists():
            try:
                man = json.loads(p.read_text(encoding="utf-8"))
                break
            except ValueError:
                pass
    pre = _pretrain_steps(man)
    final = d.get("best_window_sr")
    if final is None:
        final = (d.get("outcomes") or {}).get("sr", 0.0)
    return [(int(c["step"]) - pre, float(c["sr"])) for c in cps], float(final)


def budget_success(curve, env_step):
    """Ruler success of the newest checkpoint at or before env_step (0 before the first); a stopped run keeps its
    last score."""
    best = 0.0
    for s, v in curve:
        if s <= env_step:
            best = v
    return best


def pilot_cells(camp, arm="keepall"):
    """{'intermediate': [(curve, final)], 'novice': [...]} of one arm of a pilot."""
    out = {"intermediate": [], "novice": []}
    ev = Path(camp) / "eval_runs"
    if not ev.exists():
        return out
    for d in sorted(ev.iterdir()):
        m = re.match(rf"{re.escape(arm)}_(int|nov)\d+_s\d+$", d.name)
        if not m:
            continue
        cc = cell_curve(camp, d.name)
        if cc:
            out["intermediate" if m.group(1) == "int" else "novice"].append(cc)
    return out


def pilot_verdicts(root, rule=None):
    """The pilot rule per candidate whose pilot has scored keepall cells."""
    rule = rule or PILOT_RULE
    b0, res = rule["budget_env_steps"], {}
    for c in EXT_CANDIDATES:
        camp = Path(root) / EXT_PILOT[c]
        if not (camp / "eval_runs").exists():
            continue
        cells = pilot_cells(camp)
        ints, novs = cells["intermediate"], cells["novice"]
        if not ints:
            res[c] = {"ran": True, "qualifies": False, "reasons": ["no scored keepall cell of the intermediate"]}
            continue
        a = sum(budget_success(cv, b0) for cv, _ in ints) / len(ints)
        f = sum(fin for _, fin in ints) / len(ints)
        an = sum(budget_success(cv, b0) for cv, _ in novs) / len(novs) if novs else None
        q = rule["lo"] <= a <= rule["hi"] and f >= rule["final_min"]
        res[c] = {"ran": True, "intermediate_at_budget": round(a, 4), "intermediate_final": round(f, 4),
                  "novice_at_budget": None if an is None else round(an, 4),
                  "gap": None if an is None else round(a - an, 4), "qualifies": q,
                  "n_intermediate": len(ints), "n_novice": len(novs)}
    return res


def choose_task(verdicts, rule=None):
    """(candidate or None, how, cap or None) by the pilot rule."""
    rule = rule or PILOT_RULE
    qual = [c for c in EXT_CANDIDATES if verdicts.get(c, {}).get("qualifies")]
    if qual:
        best = max(qual, key=lambda c: (verdicts[c]["gap"] if verdicts[c]["gap"] is not None else -9.0,
                                        -verdicts[c]["intermediate_at_budget"], -EXT_CANDIDATES.index(c)))
        return best, "qualifies (largest intermediate - novice gap)", None
    fb = [c for c in EXT_CANDIDATES if verdicts.get(c, {}).get("ran")
          and (verdicts[c].get("intermediate_final") or 0.0) >= rule["fallback_final_min"]]
    if fb:
        best = min(fb, key=lambda c: (verdicts[c]["intermediate_at_budget"], EXT_CANDIDATES.index(c)))
        return best, "fallback (none qualifies: the hardest one still learnable)", rule["fallback_cap"]
    return None, "no candidate is learnable: the main study stays on the standard task", None


def choose_budget(camp, rule=None):
    """(B, keepall-intermediate success at B): the budget closest to the target, the earlier one on a tie."""
    rule = rule or PILOT_RULE
    ints = pilot_cells(camp)["intermediate"]
    if not ints:
        return None, None
    best = None
    for b in rule["budgets"]:
        m = sum(budget_success(cv, b) for cv, _ in ints) / len(ints)
        if best is None or round(abs(m - rule["budget_target"]), 9) < round(abs(best[1] - rule["budget_target"]), 9):
            best = (b, m)
    return best[0], round(best[1], 4)


def cmd_decide(args):
    """The main-study task from the pilots (rule fixed in PILOT_RULE): <root>/decision.json and DECISION.md."""
    root = Path(args.root)
    sd = root / "screen" / "screen_decision.json"
    screen = json.loads(sd.read_text(encoding="utf-8")) if sd.exists() else {"candidates": {}, "passing": []}
    ver = pilot_verdicts(root)
    chosen, how, cap = choose_task(ver)
    b, at_b = choose_budget(root / EXT_PILOT[chosen]) if chosen else (None, None)
    dec = {"created": now(), "rule": PILOT_RULE, "screen_passing": screen.get("passing", []), "pilots": ver,
           "chosen": chosen, "how": how, "pilot": EXT_PILOT.get(chosen), "smoke": EXT_SMOKE.get(chosen),
           "novice_skill": ((screen.get("candidates") or {}).get(chosen) or {}).get("novice_skill") if chosen else None,
           "task": ((screen.get("candidates") or {}).get(chosen) or {}).get("knobs") if chosen else None,
           "peg_shape": ((screen.get("candidates") or {}).get(chosen) or {}).get("peg_shape") if chosen else None,
           "budget_B_env_steps": b, "keepall_intermediate_at_B": at_b, "steps_cap": cap or 30000}
    (root / "decision.json").write_text(json.dumps(dec, indent=2), encoding="utf-8")
    L = ["# Decision: the task of the main study (extension study, phase 2)", "",
         f"{now()}. Rule: " + ", ".join(f"{k} {v}" for k, v in PILOT_RULE.items() if k != "budgets") + ".", "",
         "| candidate | keepall intermediate at 8 000 env steps | final | novice at 8 000 | gap | qualifies |",
         "|---|---|---|---|---|---|"]
    for c, v in ver.items():
        L.append(f"| {c} | {_fmt(v.get('intermediate_at_budget'))} | {_fmt(v.get('intermediate_final'))} | "
                 f"{_fmt(v.get('novice_at_budget'))} | {_fmt(v.get('gap'))} | {'yes' if v.get('qualifies') else 'no'} |")
    L += ["", f"**Chosen:** {chosen or 'none'} ({how}); novice skill {_fmt(dec['novice_skill'])}; budget B = "
              f"{_fmt(b)} environment steps (keepall intermediate {_fmt(at_b)} there); step cap {dec['steps_cap']}."]
    (root / "DECISION.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[decide] chosen: {chosen or 'none'} ({how}); B = {b}; novice skill {dec['novice_skill']}")
    return 0


def prunable_root(root):
    r = Path(root).resolve()
    return ((r.parent == (REPO / "runs").resolve() and r.name in PRUNE_ROOTS)
            or (r / PRUNE_MARKER).exists())


def prune_campaign(camp, log=print, keep_last=1):
    """Free the disk of scored cells: delete the checkpoint_<step> folders that the ruler result no longer needs,
    keeping each cell's best window, its last scored and its newest `keep_last` checkpoints (phase 3: the three behind
    success at B). Cells without .done or without a ruler result, and every other file or folder, stay. Returns
    (cells pruned, folders deleted, bytes freed)."""
    camp = Path(camp)
    ck = camp / "checkpoints"
    n_cells = n_dirs = freed = 0
    if not ck.exists():
        return 0, 0, 0
    for cell in sorted(d for d in ck.iterdir() if d.is_dir()):
        cj = camp / "eval_runs" / cell.name / "cell.json"
        if not (cell / ".done").exists() or not cj.exists():
            continue
        try:
            d = json.loads(cj.read_text(encoding="utf-8"))
        except ValueError:
            continue
        steps = []
        for sub in cell.iterdir():
            m = re.fullmatch(r"checkpoint_(\d+)", sub.name)
            if m and sub.is_dir():
                steps.append(int(m.group(1)))
        if not steps:
            continue
        keep = {int(s) for s in (d.get("best_window_steps") or [])} | set(sorted(steps)[-max(int(keep_last), 1):])
        if d.get("checkpoint_step") is not None:
            keep.add(int(d["checkpoint_step"]))
        gone = 0
        for s in steps:
            if s in keep:
                continue
            p = cell / f"checkpoint_{s}"
            freed += sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            shutil.rmtree(p, ignore_errors=True)
            gone += 1
        if gone:
            n_cells += 1
            n_dirs += gone
    if log and n_dirs:
        log(f"[prune] {camp.name}: {n_dirs} checkpoint folders of {n_cells} scored cells deleted, "
            f"{freed / 2 ** 30:.1f} GB freed (best window, last and newest kept)")
    return n_cells, n_dirs, freed


def cmd_prune(args):
    root = Path(args.root)
    if not prunable_root(root):
        print(f"[prune] refusing: {root} is not one of runs/{{{','.join(PRUNE_ROOTS)}}} and has no {PRUNE_MARKER}")
        return 2
    for n in resolve(args.names or ORDER):
        if (root / n).exists():
            prune_campaign(root / n)
    return 0


def cmd_extension_all(args):
    """Phase 2 in one go, unattended: check, stage 0 (tests, cuboid + hex orientation limits), the operator screen,
    the learning pilots of the passing candidates (each with its screened novice skill), the decision, the smoke test
    on the chosen task, audit, report, plots. Steps already done (stage-0 limits, screen) are not repeated. Never
    stops, never pushes; scored checkpoints are pruned after each campaign."""
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    st = RunStatus(root)
    st.decisions["mode"] = "unattended: no step waits for a person; failures are recorded and the run goes on"
    stop = threading.Event()

    def beat():
        while not stop.wait(max(0.1, float(getattr(args, "push_every_h", 6.0))) * 3600.0):
            _refresh(args, st, False)

    threading.Thread(target=beat, daemon=True).start()
    _step(st, "check", cmd_check, _ns(args))
    mode = getattr(args, "wandb", None) or ("online" if wandb_logged_in() else "offline")
    st.decisions["W&B"] = mode
    lim = root / "stage0" / "limits.json"
    have = set((json.loads(lim.read_text(encoding="utf-8")).get("limits") or {}) if lim.exists() else {})
    if {"cuboid", "hex"} <= have:
        st.decisions["stage 0"] = "orientation limits for cuboid and hex measured before; not repeated"
    else:
        _step(st, "stage0", cmd_stage0, _ns(args, skip_tests=False, skip_operator_test=True, skip_sweep=False,
                                           episodes=30, sweep_episodes=10))
    sd = root / "screen" / "screen_decision.json"
    if sd.exists() and all((root / "screen" / f"{c}.json").exists() for c in EXT_CANDIDATES):
        st.decisions["screen"] = "done before; not repeated"
    else:
        _step(st, "screen", cmd_screen, _ns(args, episodes=SCREEN_EPISODES))
    screen = json.loads(sd.read_text(encoding="utf-8")) if sd.exists() else {"passing": [], "candidates": {}}
    passing = list(screen.get("passing") or [])
    st.decisions["screen result"] = (", ".join(f"{c} (novice s = {screen['candidates'][c]['novice_skill']:g})"
                                               for c in passing) or "no candidate passes")
    _refresh(args, st, False)
    run_kw = dict(wandb=mode, dry=False, keep_going=True, without_active=True, allow_orig=False, score_as_is=True,
                  port_base=int(getattr(args, "port_base", 6100)))
    par = max(1, int(getattr(args, "parallel", 2) or 2))

    def campaign(name, extra, note):
        shape = switches(load_yaml(config_path(name)))["peg_shape"]
        dl = read_limits(root, shape) is None
        if dl:
            st.decisions[f"limits {name}"] = f"no stage-0 limits for {shape}: shape defaults"
        extra = list(extra) + disk_decision(root, name, st)
        _step(st, f"run {name}", cmd_run, _ns(args, names=[name], parallel=par, extra=extra, default_limits=dl,
                                              **run_kw), note=note)
        crashed = crashed_cells(root / name)
        if crashed:
            st.decisions[f"retry {name}"] = f"{len(crashed)} crashed cell(s) retrained once: {', '.join(crashed)}"
            _step(st, f"retry {name}", cmd_run, _ns(args, names=[name], parallel=par, extra=extra,
                                                    default_limits=dl, **run_kw))
            left = crashed_cells(root / name)
            if left:
                st.warnings.append(f"{name}: still crashed after one retry, counted as failed: {', '.join(left)}")
        if prunable_root(root):
            _n, _d, freed = prune_campaign(root / name)
            if _d:
                st.decisions[f"prune {name}"] = f"{_d} scored checkpoints deleted, {freed / 2 ** 30:.1f} GB freed"
        _refresh(args, st, False)

    for c in passing:
        s = float(screen["candidates"][c]["novice_skill"])
        campaign(EXT_PILOT[c], ["--tier_s_exec", f"novice={s:g}", "--tier_s_mon", f"novice={s:g}"],
                 f"pilot, novice skill {s:g}")
    _step(st, "decide", cmd_decide, _ns(args))
    dp = root / "decision.json"
    dec = json.loads(dp.read_text(encoding="utf-8")) if dp.exists() else {}
    st.decisions["task of the main study"] = (f"{dec.get('chosen')} ({dec.get('how')}); novice skill "
                                              f"{dec.get('novice_skill')}; budget B {dec.get('budget_B_env_steps')} "
                                              f"env steps; cap {dec.get('steps_cap')}" if dec.get("chosen")
                                              else str(dec.get("how", "no decision")))
    if dec.get("chosen"):
        smoke = EXT_SMOKE[dec["chosen"]]
        campaign(smoke, [], "smoke test of the new arms on the chosen task")
        _step(st, "audit", cmd_audit, _ns(args, names=[smoke]))
    _step(st, "report", cmd_report, _ns(args, names=[]))
    _step(st, "plots", cmd_plots, _ns(args, names=[]))
    stop.set()
    st.state = "finished" + (" with warnings" if st.warnings else "")
    _refresh(args, st, False)
    print(f"\n[all] {st.state}; see {root / 'RUN_STATUS.md'}, {root / 'DECISION.md'} and the results folder", flush=True)
    return 1 if st.warnings else 0


# ===================================================================== phase 3 of the extension study
P3_SOURCE = REPO / "runs" / "extension"        # phase 2's root: stage-0 limits, the decision, P1_hex's inputs
P3_PROBE = ("P1_hex", "X0_probe", ["operators/int00.json", "demos/int00_s0", "eval_grid.json"])
P3_TWIN = ("X1_careless", "X2_careful", ["operators", "demos", "eval_grid.json"])
P3_EXPECT = {"chosen": "hexwide"}              # the task the configs are written for (phase 2's decision)
P3_KEEP_LAST = 3                               # checkpoints kept after scoring: the three behind success at B


def tree_hashes(base, items):
    """{relative path: sha256} of every file under the given items (files or folders) of `base`."""
    base, out = Path(base), {}
    for it in items:
        p = base / it
        files = [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.is_file()) if p.is_dir() else []
        for f in files:
            out[f.relative_to(base).as_posix()] = sha256(f)
    return out


def copy_inputs(src, dst, items):
    """Copy a campaign's inputs (operators, demonstrations, ruler) into another campaign folder once, then check them
    by hash on every start. Returns (ok, message): a missing source or a differing copy is never trained on."""
    src, dst = Path(src), Path(dst)
    missing = [it for it in items if not (src / it).exists()]
    if missing:
        return False, f"inputs missing in {src}: {', '.join(missing)}"
    for it in items:
        if not (dst / it).exists():
            (dst / it).parent.mkdir(parents=True, exist_ok=True)
            if (src / it).is_dir():
                shutil.copytree(src / it, dst / it)
            else:
                shutil.copy2(src / it, dst / it)
    a, b = tree_hashes(src, items), tree_hashes(dst, items)
    if a != b:
        diff = sorted(set(a) ^ set(b) | {k for k in set(a) & set(b) if a[k] != b[k]})
        return False, f"copied inputs differ from {src}: {', '.join(diff[:6])}"
    (dst / "inputs_from.json").write_text(json.dumps({"source": str(src), "items": items, "files": len(a),
                                                      "checked": now(), "sha256": a}, indent=2), encoding="utf-8")
    return True, f"{len(a)} input files from {src.name}, identical by hash"


def probe_report(camp):
    """PROBE.md / probe.json of the NaN probe: per run, did the learner diverge, where, which metrics first."""
    camp = Path(camp)
    rows = []
    for d in sorted((camp / "checkpoints").iterdir()) if (camp / "checkpoints").exists() else []:
        if not d.is_dir():
            continue
        div = None
        for p in [d / "DIVERGED.json"] + sorted((camp / "diverged").glob(f"{d.name}.*.json")):
            if p.exists():
                try:
                    div = json.loads(p.read_text(encoding="utf-8"))
                    break
                except ValueError:
                    pass
        met = _jsonl(d / "learner_metrics.jsonl")
        last = met[-1] if met else {}
        rows.append({"run": d.name, "diverged": div is not None, "step": (div or {}).get("step"),
                     "phase": (div or {}).get("phase"), "nonfinite": (div or {}).get("nonfinite", []),
                     "last_finite_step": (div or {}).get("last_finite_step"),
                     "last_finite": (div or {}).get("last_finite_metrics"),
                     "metrics_rows": len(met), "last_logged_step": last.get("step"),
                     "env_received": (div or {}).get("env_received", last.get("env_received"))})
    rep = any(r["diverged"] for r in rows)
    (camp / "probe.json").write_text(json.dumps({"created": now(), "reproduced": rep, "runs": rows}, indent=2),
                                     encoding="utf-8")
    L = ["# NaN probe (phase 3 of the extension study)", "",
         f"{now()}. P1_hex's keepall run with int00, seed 0, twice, 3 000 environment steps, learner metrics and the "
         "divergence check every 50 learner steps.", "",
         "| run | diverged | learner step | phase | first non-finite metrics | last finite step | env steps received |",
         "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['run']} | {'yes' if r['diverged'] else 'no'} | {_fmt(r['step'])} | {r['phase'] or '-'} | "
                 f"{', '.join(r['nonfinite'][:6]) or '-'} | {_fmt(r['last_finite_step'])} | {_fmt(r['env_received'])} |")
    L += ["", "**Reading.** " + ("The divergence reproduced. The first non-finite metrics point to the term that breaks "
                                 "first; the last finite values are in probe.json. A fix (for example gradient "
                                 "clipping in every arm) is a design change: decide it before trusting runs that "
                                 "diverge again. The main campaign runs regardless, with the divergence guard (one "
                                 "retry per run)." if rep else
                                 "Neither run diverged: the divergence did not reproduce on the same inputs, so it "
                                 "depends on run-to-run noise (GPU non-determinism, timing). The guard handles it."),
          ""]
    for r in rows:
        if r["diverged"] and r["last_finite"]:
            keys = [k for k in r["last_finite"] if any(t in k for t in ("loss", "temperature", "entropy", "q", "bc"))]
            L.append(f"Last finite values of {r['run']} (step {r['last_finite_step']}): " +
                     ", ".join(f"{k} {r['last_finite'][k]}" for k in keys[:12]))
    (camp / "PROBE.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return rep


def smoke_checks(camp):
    """[(check, ok, detail)] of the phase-3 smoke: every new code path left its trace."""
    camp = Path(camp)
    ck = camp / "checkpoints"
    cells = sorted(d.name for d in ck.iterdir() if d.is_dir()) if ck.exists() else []
    out = []
    done = [c for c in cells if (ck / c / ".done").exists()]
    out.append(("every run finished", len(done) == planned_cells(camp.name) and len(done) > 0,
                f"{len(done)} of {planned_cells(camp.name)}"))
    scored = [c for c in cells if (camp / "eval_runs" / c / "cell.json").exists()]
    out.append(("every run scored on the ruler", len(scored) == len(done) and len(done) > 0, f"{len(scored)}"))
    env_ok = []
    for c in done:
        rows = _jsonl(ck / c / "env_checkpoints.jsonl")
        if rows and rows[-1].get("final") and len(rows) >= 3:
            env_ok.append(c)
    out.append(("environment-step checkpoints with a final one", len(env_ok) == len(done) and len(done) > 0,
                f"{len(env_ok)} of {len(done)}"))
    obs = [c for c in done if (ck / c / "convergence_evals.jsonl").exists()]
    out.append(("mini-evaluations logged (observe mode)", len(obs) > 0, f"{len(obs)} runs; would-be stops: "
                f"{sum((ck / c / 'WOULD_STOP.json').exists() for c in done)}"))
    careless = [c for c in done if any("careless_takeovers" in e for e in _jsonl(ck / c / "episodes.jsonl"))]
    out.append(("careless counts in episodes.jsonl, every arm", len(careless) == len(done) and len(done) > 0,
                f"{len(careless)} of {len(done)}"))
    met = [c for c in done if (ck / c / "learner_metrics.jsonl").exists()]
    out.append(("learner metrics logged", len(met) == len(done) and len(done) > 0, f"{len(met)} of {len(done)}"))
    e2 = [c for c in done if c.startswith("e2hil")]
    e2ok = [c for c in e2 if any(k.split("/")[-1] == "e2hil_careless_keep_frac"
                                 for r in _jsonl(ck / c / "learner_metrics.jsonl") for k in (r.get("metrics") or {}))]
    out.append(("E2HiL keep fractions of careless takeover states", len(e2ok) == len(e2) and len(e2) > 0,
                f"{len(e2ok)} of {len(e2)}"))
    div = sorted(p.name for p in (camp / "diverged").glob("*.json")) if (camp / "diverged").exists() else []
    out.append(("no divergence", not div, ", ".join(div) or "none"))
    return out


def cmd_phase3_report(args):
    """examples/phase3_analysis.py on the phase-3 root -> <results>/PHASE3_REPORT.md (+ csv, json)."""
    out = Path(getattr(args, "results", None) or default_results(Path(args.root)))
    return sh([sys.executable, str(HERE / "phase3_analysis.py"), "--root", str(args.root), "--out", str(out)],
              log=Path(args.root) / "phase3_report.log", sim_env=False)


def cmd_phase3_all(args):
    """Phase 3 in one go, unattended: check; stage 0 (tests; the hex orientation limits copied from phase 2); the NaN
    probe on P1_hex's inputs; the smoke test; the main campaign X1_careless; its careful twin X2_careful on X1's
    inputs (copied, checked by hash); audit, the phase-3 analysis, report, plots. A run that crashed or diverged is
    retrained once; scored checkpoints are pruned to the best window and the last three. Never stops, never pushes."""
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    st = RunStatus(root)
    st.decisions["mode"] = "unattended: no step waits for a person; failures are recorded and the run goes on"
    stop = threading.Event()

    def beat():
        while not stop.wait(max(0.1, float(getattr(args, "push_every_h", 6.0))) * 3600.0):
            _refresh(args, st, False)

    threading.Thread(target=beat, daemon=True).start()
    _step(st, "check", cmd_check, _ns(args))
    mode = getattr(args, "wandb", None) or ("online" if wandb_logged_in() else "offline")
    st.decisions["W&B"] = mode
    # ---- the task phase 2 chose (the configs are written for it)
    dp = P3_SOURCE / "decision.json"
    dec = json.loads(dp.read_text(encoding="utf-8")) if dp.exists() else {}
    if dec.get("chosen") != P3_EXPECT["chosen"]:
        st.warnings.append(f"phase 2's decision ({dp}) says {dec.get('chosen')}; the configs are written for "
                           f"{P3_EXPECT['chosen']}: running them as written")
    st.decisions["task"] = ("hex peg from the wider start range (phase 2's decision); experts and intermediates "
                            "only, careful (X2) and sometimes careless (X1)")
    # ---- stage 0: tests here; the hex orientation limits are phase 2's (the pilots and the smoke trained with them)
    lim_src, lim = P3_SOURCE / "stage0" / "limits.json", root / "stage0" / "limits.json"
    if lim_src.exists() and not lim.exists():
        lim.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(lim_src, lim)
    have = set((json.loads(lim.read_text(encoding="utf-8")).get("limits") or {}) if lim.exists() else {})
    if "hex" in have:
        st.decisions["stage 0"] = f"hex orientation limits copied from {lim_src} (phase 2); tests run here"
        _step(st, "stage0", cmd_stage0, _ns(args, skip_tests=False, skip_operator_test=True, skip_sweep=True,
                                           episodes=30, sweep_episodes=10))
    else:
        st.decisions["stage 0"] = "no phase-2 limits for the hex peg: measured here"
        _step(st, "stage0", cmd_stage0, _ns(args, skip_tests=False, skip_operator_test=True, skip_sweep=False,
                                           episodes=30, sweep_episodes=10))
    _refresh(args, st, False)
    run_kw = dict(wandb=mode, dry=False, keep_going=True, without_active=True, allow_orig=False, score_as_is=True,
                  port_base=int(getattr(args, "port_base", 6100)))
    par = max(1, int(getattr(args, "parallel", 2) or 2))

    def campaign(name, note, retry=True):
        shape = switches(load_yaml(config_path(name)))["peg_shape"]
        dl = read_limits(root, shape) is None
        if dl:
            st.decisions[f"limits {name}"] = f"no stage-0 limits for {shape}: shape defaults"
        _step(st, f"run {name}", cmd_run, _ns(args, names=[name], parallel=par, extra=[], default_limits=dl,
                                              **run_kw), note=note)
        crashed = crashed_cells(root / name)
        if crashed and retry:
            st.decisions[f"retry {name}"] = (f"{len(crashed)} crashed or diverged run(s) retrained once from scratch: "
                                             f"{', '.join(crashed)}")
            _step(st, f"retry {name}", cmd_run, _ns(args, names=[name], parallel=par, extra=[],
                                                    default_limits=dl, **run_kw))
            left = crashed_cells(root / name)
            if left:
                st.warnings.append(f"{name}: still crashed or diverged after one retry, technical failures (their "
                                   f"pairs are left out): {', '.join(left)}")
        if prunable_root(root):
            _n, _d, freed = prune_campaign(root / name, keep_last=P3_KEEP_LAST)
            if _d:
                st.decisions[f"prune {name}"] = f"{_d} scored checkpoints deleted, {freed / 2 ** 30:.1f} GB freed"
        _refresh(args, st, False)

    # ---- 1. the NaN probe: P1_hex's diverged run again, twice, on its own inputs
    src, dst, items = P3_PROBE
    ok, msg = copy_inputs(P3_SOURCE / src, root / dst, items)
    st.decisions["probe inputs"] = msg
    if ok:
        campaign(dst, "NaN probe: P1_hex keepall int00 s0, twice, 3 000 steps", retry=False)
        rep = probe_report(root / dst)
        st.decisions["probe"] = ("the divergence REPRODUCED (see X0_probe/PROBE.md); the main campaign runs with "
                                 "the guard" if rep else "no divergence in either run (see X0_probe/PROBE.md)")
        if rep:
            st.warnings.append("NaN probe: the divergence reproduced; decide on a fix (X0_probe/PROBE.md)")
    else:
        st.warnings.append(f"probe skipped: {msg}")
    # ---- 2. smoke: every new code path once
    campaign("X_smoke", "smoke test of the phase-3 code paths, 4 000 steps")
    checks = smoke_checks(root / "X_smoke")
    (root / "X_smoke" / "SMOKE.md").write_text(
        "# Smoke test (phase 3)\n\n| check | ok | detail |\n|---|---|---|\n"
        + "".join(f"| {c} | {'yes' if ok_ else 'NO'} | {d} |\n" for c, ok_, d in checks), encoding="utf-8")
    bad = [c for c, ok_, _d in checks if not ok_]
    st.decisions["smoke"] = "all checks passed" if not bad else "failed checks: " + "; ".join(bad)
    if bad:
        st.warnings.append("smoke: " + "; ".join(bad) + " (X_smoke/SMOKE.md); the main campaign runs anyway")
    # ---- 3. the main campaign, then its careful twin on the same inputs
    campaign("X1_careless", "main campaign: sometimes-careless operator, 12 arms, fixed budget 20 000 steps")
    src, dst, items = P3_TWIN
    ok, msg = copy_inputs(root / src, root / dst, items)
    st.decisions["twin inputs"] = msg
    if ok:
        campaign(dst, "careful twin of X1_careless: 6 arms, the same operators, demonstrations and ruler")
    else:
        st.warnings.append(f"X2_careful skipped: {msg}")
    # ---- 4. analysis
    _step(st, "audit", cmd_audit, _ns(args, names=[]))
    _step(st, "phase3 report", cmd_phase3_report, _ns(args))
    _step(st, "report", cmd_report, _ns(args, names=[]))
    _step(st, "plots", cmd_plots, _ns(args, names=[]))
    stop.set()
    st.state = "finished" + (" with warnings" if st.warnings else "")
    _refresh(args, st, False)
    print(f"\n[all] {st.state}; see {root / 'RUN_STATUS.md'} and the results folder", flush=True)
    return 1 if st.warnings else 0


# ===================================================================== main
def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", default="synthetic_suite", choices=sorted(SUITES),
                   help="which suite: synthetic_suite (default), inconsistent_operator, study or final")
    p.add_argument("--root", default=None, help="suite folder (default runs/<suite>)")
    p.add_argument("--results", default=None,
                   help="export folder (default results/<suite> in the repo); refreshed after every command")
    p.add_argument("--no_export", action="store_true", help="do not refresh the results folder")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="machine + repo check")
    s0 = sub.add_parser("stage0", help="tests, operator test, tolerance sweep (no training, CPU)")
    s0.add_argument("--episodes", type=int, default=30, help="operator test: episodes per setting")
    s0.add_argument("--sweep_episodes", type=int, default=10, help="tolerance sweep: episodes per angle")
    for k in ("tests", "operator_test", "sweep"):
        s0.add_argument(f"--skip_{k}", action="store_true")
    pl = sub.add_parser("plan", help="settings, switches and status")
    pl.add_argument("names", nargs="*", help="settings or groups (smoke, pilots, confirm); default all")
    pl.add_argument("--validate", action="store_true", help="dry-run every config through run_campaign.py")
    rn = sub.add_parser("run", help="run campaigns one after another")
    rn.add_argument("names", nargs="+", help="settings or groups: smoke, pilots, confirm, or setting names")
    rn.add_argument("--wandb", choices=["off", "offline", "online"], default=None)
    rn.add_argument("--dry", action="store_true", help="print the commands only")
    rn.add_argument("--keep_going", action="store_true", help="continue with the next setting after a failure")
    rn.add_argument("--default_limits", action="store_true", help="allow the channel without stage-0 limits")
    rn.add_argument("--without_active", action="store_true", help="confirmation without the asking arm")
    rn.add_argument("--allow_orig", action="store_true",
                    help="inconsistent_operator: run a setting of the original platform")
    rn.add_argument("--score_as_is", action="store_true",
                    help="inconsistent_operator: run Q2/Q3 without matching the score to the Q1 audit")
    rn.add_argument("--parallel", type=int, default=1,
                    help="train N cells side by side on one GPU (shards with their own ports; default 1)")
    rn.add_argument("--port_base", type=int, default=6100, help="--parallel: first learner port (+10 per shard)")
    rn.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="further run_campaign.py flags")
    au = sub.add_parser("audit", help="inconsistent_operator: score validity of the Q1 campaigns")
    au.add_argument("names", nargs="*", help="settings (default the Q1 group)")
    rp = sub.add_parser("report", help="comparison table of the trained settings")
    rp.add_argument("names", nargs="*", help="settings or groups; default all")
    pt = sub.add_parser("plots", help="figures per campaign -> <root>/plots/")
    pt.add_argument("names", nargs="*", help="settings or groups; default all")
    rk = sub.add_parser("rulercheck", help="would the shorter ruler change a score? (smoke campaigns)")
    rk.add_argument("names", nargs="*", help="settings or groups; default the smoke group")
    al = sub.add_parser("all", help="everything in one go, unattended: check, stage 0, smoke, the main campaigns, "
                                    "audit, report, plots; never stops")
    al.add_argument("--parallel", type=int, default=2, help="cells side by side in the smoke (kept for the main "
                                                            "campaigns only if it is faster)")
    al.add_argument("--port_base", type=int, default=6100)
    al.add_argument("--wandb", choices=["off", "offline", "online"], default=None,
                    help="default: online if a W&B key is set, else offline")
    al.add_argument("--push", action="store_true",
                    help="push the results folder to the branch --branch of the origin remote after every step")
    al.add_argument("--branch", default="study-results")
    al.add_argument("--push_every_h", type=float, default=6.0, help="also refresh/push every N hours")
    sc = sub.add_parser("screen", help="extension: operator screen of the candidate tasks and the screen rule (CPU)")
    sc.add_argument("--episodes", type=int, default=SCREEN_EPISODES)
    sub.add_parser("decide", help="extension: the main-study task from the pilots (rule fixed in PILOT_RULE)")
    sub.add_parser("phase3report", help="phase3: per-run measures and the fixed contrasts (examples/phase3_analysis.py)")
    pr = sub.add_parser("prune", help="delete checkpoints that scored runs no longer need (runs/final_twins, "
                                      "runs/extension, or a root with .prune_scored_ok)")
    pr.add_argument("names", nargs="*", help="settings or groups; default all")
    ex = sub.add_parser("export", help="collect reports, plots and light per-cell files into the results folder")
    ex.add_argument("--out", default=None, help="target folder (default --results, else results/<suite>)")
    ex.add_argument("--with_steps", action="store_true", help="also copy the per-step logs (steps.jsonl, large)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    configure(args.suite)
    if args.root is None:
        args.root = str(DEFAULT_ROOT)
    rc = {"check": cmd_check, "stage0": cmd_stage0, "plan": cmd_plan, "run": cmd_run, "audit": cmd_audit,
          "report": cmd_report, "plots": cmd_plots, "rulercheck": cmd_rulercheck, "export": cmd_export,
          "all": cmd_all, "screen": cmd_screen, "decide": cmd_decide, "prune": cmd_prune,
          "phase3report": cmd_phase3_report}[args.cmd](args)
    if args.cmd in ("check", "stage0", "run", "audit", "report", "plots", "rulercheck", "screen",
                    "decide") and auto_export(args):
        export_results(args)
    return rc


if __name__ == "__main__":
    sys.exit(main())
