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
    if info["disk_free_gb"] < 50:
        problems.append(f"only {info['disk_free_gb']} GB free under {root}")
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
    if not args.skip_operator_test and SUITE == "inconsistent_operator":
        # the inconsistent operator held in each condition: inattentive must be clearly worse
        rc_all |= sh([sys.executable, "examples/oracle_insertion_test.py", "--shapes", "cuboid",
                      "--conditions", "attentive,inattentive", "--rotation", "off,on",
                      "--episodes", str(args.episodes), "--out", str(root / "operator_conditions.json")],
                     log=root / "operator_conditions.log", env={"JAX_PLATFORMS": "cpu"})
    elif not args.skip_operator_test:
        rc_all |= sh([sys.executable, "examples/oracle_insertion_test.py", "--shapes", "cuboid,cylinder",
                      "--episodes", str(args.episodes), "--out", str(root / "operator_test.json")],
                     log=root / "operator_test.log", env={"JAX_PLATFORMS": "cpu"})
    if not args.skip_sweep and SUITE == "synthetic_suite":
        rc = sh([sys.executable, "examples/oracle_insertion_test.py", "--sweep", "tilt,yaw", "--shapes", "cuboid,cylinder",
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
    """Orientation limits of stage 0: this suite's, else the synthetic suite's (same machine)."""
    for p in (Path(root) / "stage0" / "limits.json", REPO / "runs" / "synthetic_suite" / "stage0" / "limits.json"):
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


def cmd_run(args):
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    names = resolve(args.names)
    for n in names:
        cmd, sw, limits = campaign_cmd(args, n, root)
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
        print(f"\n[run] {n}: " + " ".join(_q(c) for c in cmd))
        if args.dry:
            continue
        camp.mkdir(parents=True, exist_ok=True)
        if not fp_path.exists():
            fp_path.write_text(json.dumps(dict(fp, started=now(), machine=socket.gethostname()), indent=2),
                               encoding="utf-8")
        t0 = now()
        rc = sh(cmd, log=root / f"{n}.out", sim_env=False)       # run_campaign sources sim_env.sh itself
        with open(root / "suite_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"setting": n, "start": t0, "end": now(), "rc": rc,
                                "machine": socket.gethostname()}) + "\n")
        if rc != 0 and not args.keep_going:
            print(f"[run] {n} failed (rc {rc}); stopping. Re-run the same command to resume (finished cells are kept).")
            return rc
    return 0


# ===================================================================== audit (Q1)
def cmd_audit(args):
    """Score validity on the inconsistent operator: examples/inconsistent_operator_audit.py per setting."""
    root = Path(args.root)
    rc_all = 0
    for n in resolve(args.names or GROUPS.get("q1", [])):
        camp = root / n
        if not (camp / "aqi_logs").exists():
            print(f"[audit] {n}: no aqi_logs yet under {camp}")
            rc_all |= 1
            continue
        rc_all |= sh([sys.executable, str(HERE / "inconsistent_operator_audit.py"), "--campaign", str(camp)],
                     log=root / f"{n}.audit.log", sim_env=False)
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
    sr = None
    cj = camp / "eval_runs" / cid / "cell.json"
    if cj.exists():
        try:
            sr = json.loads(cj.read_text())["outcomes"]["sr"]
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
        "ruler_sr": sr, "actor_minutes": actor_minutes(camp / "logs" / f"{cid}.actor.log"),
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
            "actor_minutes": _mean([c["actor_minutes"] for c in cs]),
            "ruler_sr_by_operator": {c["operator"]: c["ruler_sr"] for c in cs},
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
            "early_time_cap_handbacks", "ruler_sr", "actor_minutes"]
    head = ["setting", "arm", "done", "rot", "box", "chan", "conv", "utd", "first unaided (median)",
            "never unaided", "conv.", "conv. step", "timeouts ≤10k", "success ≤10k", "operator share",
            "orient. takeovers", "time-cap hand-backs", "ruler SR", "actor min"]
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


# ===================================================================== main
def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", default="synthetic_suite", choices=sorted(SUITES),
                   help="which suite: synthetic_suite (default) or inconsistent_operator")
    p.add_argument("--root", default=None, help="suite folder (default runs/<suite>)")
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
    rn.add_argument("--score_as_is", action="store_true",
                    help="inconsistent_operator: run Q2/Q3 without matching the score to the Q1 audit")
    rn.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="further run_campaign.py flags")
    au = sub.add_parser("audit", help="inconsistent_operator: score validity of the Q1 campaigns")
    au.add_argument("names", nargs="*", help="settings (default the Q1 group)")
    rp = sub.add_parser("report", help="comparison table of the trained settings")
    rp.add_argument("names", nargs="*", help="settings or groups; default all")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    configure(args.suite)
    if args.root is None:
        args.root = str(DEFAULT_ROOT)
    return {"check": cmd_check, "stage0": cmd_stage0, "plan": cmd_plan, "run": cmd_run,
            "audit": cmd_audit, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
