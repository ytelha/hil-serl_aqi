#!/usr/bin/env python3
"""The fixed evaluation protocol - "the ruler" (Experimental Design A.2).

Evaluates ONE frozen policy (one experimental cell) on the pre-generated
initial-state grid: N autonomous trials in a fixed order, policy frozen,
no human at the controls, gate disabled, interventions impossible.

Per trial it logs success, time-to-success, peak/mean contact force and
the force-violation flag; per cell it writes the B.1 aggregate (success
rate with 95% Wilson interval, mean time-to-success, mean peak force,
violation rate).

The eval grid must be generated ONCE per campaign with
    python examples/experiment_pipeline.py init ...
which writes eval_grid.json (25 poses x repetitions in a fixed order)
and records its SHA-256 in the campaign manifest; this script refuses a
grid whose hash does not match its own content, so every cell in a
campaign is measured with the identical ruler.

Start-pose control (in preference order):
  1. any wrapper in the chain exposing set_eval_reset_pose(dx_m, dyaw_rad)
     - the generic hook a simulation env can implement;
  2. a FrankaEnv-style base env (resetpos / _RESET_POSE / randomreset):
     randomization is disabled and the exact grid offset is applied to
     the reset pose, restoring the original state afterwards;
  3. neither found: the env's own reset is used, the intended pose is
     still logged, and pose_control="env_random" marks the limitation.

Usage (one cell):
    python examples/eval_ruler.py --exp_name=ram_insertion \
        --checkpoint_path=runs/logged_alice_s0 \
        --grid_file=campaign/eval_grid.json \
        --out=campaign/eval_runs/logged_alice_s0
"""

import datetime
import glob
import hashlib
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from absl import app, flags
from flax.training import checkpoints

from serl_launcher.utils.launcher import (
    make_sac_pixel_agent,
    make_sac_pixel_agent_hybrid_single_arm,
    make_sac_pixel_agent_hybrid_dual_arm,
)
from serl_launcher.wrappers.chunking import ChunkingWrapper  # noqa: F401
from serl_launcher.aqi.runtime import make_proprio_slicer, _state_vec
from serl_launcher.aqi import stats as aqi_stats

from experiments.mappings import CONFIG_MAPPING
import ruler_select

FLAGS = flags.FLAGS


def _resolve_agent_type(checkpoint_path):
    """(agent_type, metadata) from --agent_type / agent_type.json."""
    meta = {}
    path = os.path.join(os.path.abspath(checkpoint_path or ""),
                        "agent_type.json")
    if os.path.exists(path):
        try:
            meta = json.load(open(path))
        except (ValueError, OSError):
            meta = {}
    if FLAGS.agent_type != "auto":
        return FLAGS.agent_type, meta
    return meta.get("agent_type", "sac"), meta
flags.DEFINE_string("exp_name", None, "Experiment name (task).")
flags.DEFINE_enum("peg_shape", None, ["cylinder", "hex", "cuboid"],
                  "Insertion geometry for the eval env. If unset, falls back to "
                  "the grid file's 'peg_shape' field, then the config default. "
                  "MUST match the shape the checkpoint was trained on -- the "
                  "config default is 'cylinder', so a cuboid/hex policy scored "
                  "without this flag is silently evaluated on the wrong peg.")
flags.DEFINE_string("checkpoint_path", None, "Trained cell's checkpoint dir.")
flags.DEFINE_enum("agent_type", "auto", ["auto", "sac", "hiril"],
                  "Architecture to restore the checkpoint into. 'auto' reads "
                  "<checkpoint_path>/agent_type.json (written by "
                  "train_hiril.py) and falls back to 'sac' (every "
                  "train_aqi_rlpd.py mode).")
flags.DEFINE_integer("eval_checkpoint_step", 0,
                     "Checkpoint step; 0 = latest (the frozen policy).")
flags.DEFINE_integer("eval_last_k", 1,
                     "If >1, score the last K checkpoints on the grid and "
                     "report the pooled mean SR (de-noises training "
                     "oscillation). 1 = single checkpoint (eval_checkpoint_step "
                     "or latest).")
flags.DEFINE_string("grid_file", None, "eval_grid.json from the campaign.")
flags.DEFINE_string("out", None, "Output directory for this cell's eval.")
flags.DEFINE_string("cell_id", None, "Cell identifier (default: out basename).")
flags.DEFINE_float("control_hz", 10.0, "Control rate for step->seconds.")
flags.DEFINE_float("f_max", None,
                   "Force-violation threshold [N]; default: the task's "
                   "AQIConfig.force_hard.")
flags.DEFINE_boolean("argmax", True,
                     "Deterministic (mode) actions for the frozen ruler; "
                     "set false to mirror the stochastic baseline eval.")
flags.DEFINE_integer("seed", 0, "RNG seed (used only when --argmax=false).")
flags.DEFINE_boolean("save_video", False, "Save evaluation videos.")
flags.DEFINE_boolean("no_realtime", False,
                     "Disable the env's 10 Hz real-time pacing during eval "
                     "(identical dynamics, just no wall-clock sleep) so "
                     "offline checkpoint sweeps run at full compute speed.")
flags.DEFINE_boolean("prefer_certified", True,
                     "If the cell has a <checkpoint_path>/certified checkpoint "
                     "(the EXACT weights that passed the convergence mini-eval), "
                     "evaluate THAT instead of the last-K learner checkpoints "
                     "-- avoids the actor-certified vs saved-checkpoint drift on "
                     "an oscillating policy.")
flags.DEFINE_enum("select", "pooled", ["pooled", "best", "mean", "best_window"],
                 "How to summarize the cell SR across checkpoints: 'pooled' "
                 "(default) = the eval_last_k window pooled; 'best' = score ALL "
                 "checkpoints and report the single BEST one (peak SR the run "
                 "produced -- fixes the last-K window landing in an oscillation "
                 "trough); 'mean' = score ALL and report the mean over all "
                 "checkpoints; 'best_window' = score ALL and report the best "
                 "mean over --select_window_k CONSECUTIVE checkpoints (rewards "
                 "SUSTAINED competence, removing both the last-K trough artifact "
                 "and the best-of-1 selection bias). Non-pooled modes override "
                 "eval_last_k.")
flags.DEFINE_integer("select_window_k", 3,
                     "Window size (consecutive checkpoints) for "
                     "--select=best_window.")
# Shortenings of the all-checkpoint sweep (ruler_select.py); all off by default.
flags.DEFINE_integer("select_min_step", 0,
                     "With --select best/mean/best_window: score only checkpoints "
                     "saved after this learner step (run_campaign passes the arm's "
                     "demo pre-training steps with eval.skip_pretrain). 0 = all.")
flags.DEFINE_integer("select_stride", 1,
                     "With --select best/mean/best_window: score every n-th "
                     "checkpoint, counted back from the newest. 1 = all. Changes "
                     "the best-window score (docs/eval_ruler.md).")
flags.DEFINE_integer("select_newest", 0,
                     "With --select best/mean/best_window: score only the newest N "
                     "checkpoints (after the other two). 0 = all.")
flags.DEFINE_integer("max_steps", 0,
                     "Per-trial step cap; 0 = env's own horizon.")


def _sha256_of_grid(grid: dict) -> str:
    payload = {k: grid[k] for k in sorted(grid) if k != "sha256"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _find_pose_controller(env):
    """Return (kind, target) for start-pose control, walking the chain."""
    e = env
    while e is not None:
        if hasattr(e, "set_eval_reset_pose"):
            return "hook", e
        e = getattr(e, "env", None)
    base = env.unwrapped
    if all(hasattr(base, a) for a in ("resetpos", "_RESET_POSE",
                                      "randomreset")):
        return "franka", base
    return "env_random", None


class _FrankaPoseDriver:
    """Applies exact (dx, dyaw) grid offsets through the same mechanics
    FrankaEnv.reset() uses for its own randomization, deterministically;
    restores the env's original reset behaviour on close()."""

    def __init__(self, base, lateral_axis: int = 0):
        self.base = base
        self.axis = lateral_axis
        self._orig_randomreset = base.randomreset
        self._orig_resetpos = np.array(base.resetpos, copy=True)
        try:  # the exact converter the env itself uses
            from franka_env.utils.rotations import euler_2_quat
            self._e2q = euler_2_quat
        except ImportError:
            from scipy.spatial.transform import Rotation
            self._e2q = lambda e: Rotation.from_euler("xyz", e).as_quat()

    def apply(self, dx_m: float, dyaw_rad: float):
        self.base.randomreset = False
        pose = self._orig_resetpos.copy()
        pose[self.axis] += dx_m
        euler = np.array(self.base._RESET_POSE[3:], copy=True)
        euler[-1] += dyaw_rad
        pose[3:] = self._e2q(euler)
        self.base.resetpos = pose

    def close(self):
        self.base.randomreset = self._orig_randomreset
        self.base.resetpos = self._orig_resetpos


def _score_grid(agent, env, trials_spec, kind, target, driver,
                f_sl, tau_sl, f_max, rng, tf, cstep):
    """Run every grid trial once for the currently-restored ``agent`` and
    append each trial record to ``tf``; return (trials, rng)."""
    trials = []
    for i, spec in enumerate(trials_spec):
        dx_m = spec["dx_mm"] / 1000.0
        dyaw = spec["dyaw_rad"]
        if kind == "hook":
            target.set_eval_reset_pose(dx_m, dyaw)
        elif kind == "franka":
            driver.apply(dx_m, dyaw)

        obs, _ = env.reset()
        done, steps, intervened = False, 0, False
        f_peak, tau_peak, f_sum = 0.0, 0.0, 0.0
        reward = 0.0
        while not done:
            rng, key = jax.random.split(rng)
            actions = agent.sample_actions(
                observations=jax.device_put(obs),
                argmax=FLAGS.argmax, seed=key)
            actions = np.asarray(jax.device_get(actions))
            obs, reward, done, truncated, info = env.step(actions)
            if "intervene_action" in info:
                intervened = True  # ruler contaminated; flagged
            state = _state_vec(obs)
            if f_sl:
                fn = float(np.linalg.norm(state[slice(*f_sl)]))
                f_peak = max(f_peak, fn)
                f_sum += fn
            if tau_sl:
                tau_peak = max(tau_peak, float(
                    np.linalg.norm(state[slice(*tau_sl)])))
            steps += 1
            if FLAGS.max_steps and steps >= FLAGS.max_steps:
                done = True

        success = bool(reward > 0)
        rec = dict(
            trial=i, checkpoint_step=cstep, dx_mm=spec["dx_mm"], dyaw_rad=dyaw,
            rep=spec.get("rep", 0), success=success, steps=steps,
            time_s=(steps / FLAGS.control_hz) if success else None,
            f_peak=f_peak, f_mean=f_sum / max(steps, 1),
            tau_peak=tau_peak,
            violation=bool(f_peak > f_max),
            intervened=intervened, pose_control=kind,
        )
        trials.append(rec)
        tf.write(json.dumps(rec) + "\n")
        tf.flush()
        print(f"[ruler] ckpt {cstep} trial {i + 1}/{len(trials_spec)} "
              f"dx={spec['dx_mm']:+.0f}mm yaw={dyaw:+.2f}rad -> "
              f"{'success' if success else 'fail'} "
              f"({steps} steps, F_peak={f_peak:.1f}N)"
              + (" [INTERVENED - contaminated]" if intervened else ""))
    return trials, rng


def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    assert FLAGS.grid_file and FLAGS.checkpoint_path and FLAGS.out

    with open(FLAGS.grid_file) as f:
        grid = json.load(f)
    sha = _sha256_of_grid(grid)
    if grid.get("sha256") and grid["sha256"] != sha:
        raise SystemExit("eval_grid.json content does not match its "
                         "recorded sha256 - the ruler has been modified.")
    trials_spec = grid["trials"]  # fixed order: [{dx_mm, dyaw_rad, rep}, ...]

    config = CONFIG_MAPPING[FLAGS.exp_name]()
    # The eval env MUST use the peg the policy trained on. eval_ruler has no
    # training flags, so without this the config default ('cylinder') is used
    # and a cuboid/hex policy is silently scored on the wrong geometry.
    _peg = FLAGS.peg_shape or grid.get("peg_shape")
    if _peg is not None:
        setattr(config, "peg_shape", _peg)
    print(f"[ruler] eval peg_shape = {getattr(config, 'peg_shape', None)}")
    aqi_cfg = config.get_aqi_config(mode="logged")
    f_max = FLAGS.f_max if FLAGS.f_max is not None else aqi_cfg.force_hard

    env = config.get_environment(fake_env=False,
                                 save_video=FLAGS.save_video,
                                 classifier=True)
    _box = getattr(getattr(env, "unwrapped", env), "_rpy_box", None)
    if _box is not None:              # $SIM_RPY_BOX (sim orientation box)
        print(f"[ruler] orientation box: roll +/-{_box[0]:g}, pitch +/-{_box[1]:g}, "
              f"yaw +/-{_box[2]:g} rad")
    if FLAGS.no_realtime:
        # offline eval: drop the 10 Hz operator pacing (dynamics unchanged).
        try:
            env.unwrapped.realtime = False
        except Exception:
            pass

    agent_kwargs = dict(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=config.image_keys,
        encoder_type=config.encoder_type,
        discount=config.discount,
    )
    agent_type, agent_meta = _resolve_agent_type(FLAGS.checkpoint_path)
    if agent_type == "hiril":
        # HiRIL baseline (train_hiril.py): TD3 actor + BIQN critic; rebuilt
        # from the exact HiRILConfig the learner saved. argmax = pi(s).
        from serl_launcher.hiril.config import HiRILConfig
        from serl_launcher.utils.launcher import make_hiril_pixel_agent
        agent = make_hiril_pixel_agent(
            **agent_kwargs,
            hiril=HiRILConfig.from_dict(agent_meta.get("hiril", {})))
        print("[ruler] agent_type = hiril")
    elif config.setup_mode in ("single-arm-fixed-gripper",
                               "dual-arm-fixed-gripper"):
        agent = make_sac_pixel_agent(**agent_kwargs)
    elif config.setup_mode == "single-arm-learned-gripper":
        agent = make_sac_pixel_agent_hybrid_single_arm(**agent_kwargs)
    elif config.setup_mode == "dual-arm-learned-gripper":
        agent = make_sac_pixel_agent_hybrid_dual_arm(**agent_kwargs)
    else:
        raise NotImplementedError(config.setup_mode)

    # ---- checkpoints to score -----------------------------------------------
    # Mean-over-last-K scores the frozen grid on each of the last K saved
    # checkpoints and POOLS the trials, so the reported SR is the expected
    # performance over the late-training window rather than one (possibly
    # lucky/unlucky) snapshot -- essential while the policy still oscillates.
    # K=1 keeps the original single-checkpoint behaviour.
    abspath = os.path.abspath(FLAGS.checkpoint_path)
    # Prefer the actor-saved CERTIFIED checkpoint (the exact weights that passed
    # the convergence mini-eval) over the last-K learner checkpoints: on an
    # oscillating policy the post-STOP learner checkpoint can be a different
    # (worse) phase than the certified one.
    certified_dir = os.path.join(abspath, "certified")
    use_certified = bool(
        FLAGS.prefer_certified and os.path.isdir(certified_dir)
        and glob.glob(os.path.join(certified_dir, "checkpoint_*")))
    ckpt_source = certified_dir if use_certified else abspath
    all_steps = sorted(
        int(os.path.basename(p).rsplit("_", 1)[-1])
        for p in glob.glob(os.path.join(ckpt_source, "checkpoint_*"))
        if os.path.basename(p).rsplit("_", 1)[-1].isdigit())
    _filt = (FLAGS.select_min_step, FLAGS.select_stride, FLAGS.select_newest)
    ck_filter = None
    if use_certified:
        eval_steps = [all_steps[-1]] if all_steps else [None]
        print(f"[ruler] using CERTIFIED checkpoint {eval_steps[-1]} "
              f"({certified_dir})")
    elif (FLAGS.select in ("best", "mean", "best_window") and all_steps
          and ruler_select.is_active(*_filt)):
        eval_steps = ruler_select.keep_checkpoints(all_steps, *_filt)
        ck_filter = dict(ruler_select.describe(*_filt),
                         n_saved=len(all_steps), n_scored=len(eval_steps))
        print(f"[ruler] select={FLAGS.select}: scoring {len(eval_steps)} of "
              f"{len(all_steps)} checkpoints (after step {ck_filter['min_step']}, "
              f"every {ck_filter['stride']}, newest {ck_filter['newest'] or 'all'}): "
              f"{eval_steps}")
    elif FLAGS.select in ("best", "mean", "best_window") and all_steps:
        eval_steps = all_steps
        print(f"[ruler] select={FLAGS.select}: scoring ALL "
              f"{len(all_steps)} checkpoints")
    elif FLAGS.eval_last_k and FLAGS.eval_last_k > 1 and all_steps:
        eval_steps = all_steps[-FLAGS.eval_last_k:]
    elif FLAGS.eval_checkpoint_step:
        eval_steps = [FLAGS.eval_checkpoint_step]
    else:
        eval_steps = [all_steps[-1]] if all_steps else [None]
    if ck_filter is None and ruler_select.is_active(*_filt):
        print("[ruler] WARNING: --select_min_step / --select_stride / "
              "--select_newest apply only to the all-checkpoint sweep (--select "
              "best/mean/best_window, no certified checkpoint): ignored.")

    slices = make_proprio_slicer(env)
    f_sl = slices.get(aqi_cfg.force_obs_key)
    tau_sl = slices.get(aqi_cfg.torque_obs_key)

    kind, target = _find_pose_controller(env)
    driver = _FrankaPoseDriver(target) if kind == "franka" else None
    if kind == "env_random":
        print("[ruler] WARNING: no pose hook found; using the env's own "
              "reset randomization. Requested poses are still logged. "
              "Implement set_eval_reset_pose(dx_m, dyaw_rad) on the task "
              "env for exact grid control.")

    os.makedirs(FLAGS.out, exist_ok=True)
    cell_id = FLAGS.cell_id or os.path.basename(FLAGS.out.rstrip("/"))
    trials_path = os.path.join(FLAGS.out, "trials.jsonl")
    rng = jax.random.PRNGKey(FLAGS.seed)
    trials = []
    sr_per_checkpoint = []

    try:
        with open(trials_path, "w") as tf:
            for cstep in eval_steps:
                ckpt = checkpoints.restore_checkpoint(
                    ckpt_source, agent.state, step=cstep)
                agent = agent.replace(state=ckpt)
                ck_trials, rng = _score_grid(
                    agent, env, trials_spec, kind, target, driver,
                    f_sl, tau_sl, f_max, rng, tf, cstep)
                trials.extend(ck_trials)
                ck_sr = (float(np.mean([1.0 if t["success"] else 0.0
                                        for t in ck_trials]))
                         if ck_trials else float("nan"))
                sr_per_checkpoint.append(
                    {"step": cstep, "sr": ck_sr, "n_trials": len(ck_trials)})
    finally:
        if driver is not None:
            driver.close()

    ckpt_step = (eval_steps[-1] if eval_steps and eval_steps[-1] is not None
                 else int(os.path.basename(
                     checkpoints.latest_checkpoint(ckpt_source))[11:]))
    # checkpoint-selection summary: best single checkpoint + mean over all,
    # so the reported SR need not depend on where the last-K window landed.
    best_step, best_sr, mean_all_sr = None, None, None
    best_window_sr, best_window_steps = None, None
    _valid = [p for p in sr_per_checkpoint if p["sr"] == p["sr"]]
    if _valid:
        _bp = max(_valid, key=lambda p: p["sr"])
        best_step, best_sr = _bp["step"], float(_bp["sr"])
        mean_all_sr = float(np.mean([p["sr"] for p in _valid]))
        # best-stable-window: the k consecutive checkpoints with the highest
        # mean SR (sustained competence, not a 1-checkpoint spike).
        _vs = sorted(_valid, key=lambda p: p["step"])
        _w = min(max(int(FLAGS.select_window_k), 1), len(_vs))
        _bwm = -1.0
        for _i in range(0, len(_vs) - _w + 1):
            _win = _vs[_i:_i + _w]
            _m = float(np.mean([p["sr"] for p in _win]))
            if _m > _bwm:
                _bwm = _m
                best_window_sr = _m
                best_window_steps = [p["step"] for p in _win]
    if FLAGS.select == "best" and best_step is not None:
        _sel = [t for t in trials if t.get("checkpoint_step") == best_step]
        agg = aqi_stats.cell_outcomes(_sel, f_max=f_max)
        print(f"[ruler] select=best -> checkpoint {best_step} SR={best_sr:.2f}")
    elif FLAGS.select == "best_window" and best_window_steps:
        _ws = set(best_window_steps)
        _sel = [t for t in trials if t.get("checkpoint_step") in _ws]
        agg = aqi_stats.cell_outcomes(_sel, f_max=f_max)
        print(f"[ruler] select=best_window -> steps {best_window_steps} "
              f"SR={best_window_sr:.2f}")
    else:
        agg = aqi_stats.cell_outcomes(trials, f_max=f_max)  # pooled over K x N
    cell = dict(
        cell_id=cell_id, exp_name=FLAGS.exp_name,
        checkpoint_path=abspath,
        checkpoint_step=ckpt_step,
        used_certified=use_certified,
        select=FLAGS.select,
        best_step=best_step,
        best_sr=best_sr,
        mean_all_sr=mean_all_sr,
        best_window_sr=best_window_sr,
        best_window_steps=best_window_steps,
        eval_last_k=len(eval_steps),
        eval_checkpoint_steps=[s for s in eval_steps if s is not None],
        sr_per_checkpoint=sr_per_checkpoint,
        grid_sha256=sha, n_trials_spec=len(trials_spec),
        control_hz=FLAGS.control_hz, f_max=f_max, argmax=FLAGS.argmax,
        pose_control=kind,
        datetime=datetime.datetime.now().isoformat(timespec="seconds"),
        outcomes=agg,
    )
    if ck_filter is not None:          # absent = every saved checkpoint scored
        cell["checkpoint_filter"] = ck_filter
    with open(os.path.join(FLAGS.out, "cell.json"), "w") as f:
        json.dump(cell, f, indent=2)
    lo, hi = agg["sr_wilson95"]
    print(f"[ruler] cell {cell_id}: SR={agg['sr']:.2f} "
          f"Wilson95=[{lo:.2f},{hi:.2f}]  T={agg['time_to_success_mean']:.1f}s"
          f"  F_peak={agg['f_peak_mean']:.1f}N  VR={agg['violation_rate']:.2f}"
          + (f"  ({agg['n_contaminated']} contaminated trials excluded)"
             if agg["n_contaminated"] else ""))

    # self-describing eval provenance next to cell.json (best-effort). Try to
    # recover the training method + operator from the checkpoint's run manifest.
    try:
        import provenance as _prov
        method, operator = None, None
        run_mf = os.path.join(os.path.abspath(FLAGS.checkpoint_path),
                              "manifest.json")
        if os.path.exists(run_mf):
            rm = json.load(open(run_mf))
            method, operator = rm.get("method"), rm.get("operator")
        _prov.document(
            FLAGS.out, "eval", quiet=True,
            task={"exp_name": FLAGS.exp_name},
            method=method, operator=operator,
            evaluation={"grid_sha256": sha, "n_trials": len(trials_spec),
                        "argmax": FLAGS.argmax, "checkpoint_step": ckpt_step,
                        "f_max": f_max, "pose_control": kind, "outcomes": agg},
            reproduce=[f"./.venv/bin/python examples/eval_ruler.py --exp_name "
                       f"{FLAGS.exp_name} --checkpoint_path "
                       f"{os.path.abspath(FLAGS.checkpoint_path)} --grid_file "
                       f"{FLAGS.grid_file} --out {FLAGS.out} --argmax"])
    except Exception:
        pass
    print(f"[ruler] wrote {trials_path} and cell.json")


if __name__ == "__main__":
    app.run(main)
