#!/usr/bin/env python3
"""HiRIL training pipeline -- external comparison baseline (NOT AQI).

Y. Zhou, Y.-B. Zhao, C. Xu, C. Ouyang, P. Li, "Human-in-the-loop
reinforcement learning with risk-aware intervention and imitation",
Expert Systems With Applications 308 (2026) 131118.

A separate pipeline (not a --aqi_mode) because HiRIL replaces the learner
itself: TD3-style deterministic actor + BIQN critic (Bayesian implicit
quantile network) + one TDRD-prioritised replay buffer, instead of SAC /
RLPD with two 50/50 buffers. It reuses the rest of the infrastructure: the
experiment configs and envs, the pixel encoder, agentlace actor/learner
split, the synthetic operator, the convergence rule and certified
checkpoints, episodes.jsonl, provenance, and (via --agent_type) the eval
ruler and the campaign runner (arms with `mode: hiril`).

Algorithm 1 of the paper -> this file:
  line  5  a_RL = pi(s) + noise              actor(): sample_actions(noise_std)
  line  6  DU(s, a_RL)                       actor(): _du() -> agent.dual_uncertainty
  line 7-10 request if DU > tau_risk         actor(): request -> env.notify_request;
                                             the synthetic operator responds with
                                             its skill-modulated gate (reaction
                                             delay, misses) -- as for AQI's
                                             active arms -- so Delta_t = the
                                             ACTUAL takeover, not the request
  line 12  r + r_p at takeover onset          actor(): stored_reward (Eq. 25)
  line 13  store (s,a,r,s',DU,Delta)         actor(): hiril_human, hiril_bonus
  line 14-15 TDRD sampling + priority update learner(): RiskAwarePERDataStore,
                                             update_priorities(per_index, td_abs)
  line 17-22 critic (Eq. 29) / actor (Eq. 32) serl_launcher/hiril/agent.py
tau_risk: absolute (--hiril_tau_risk > 0, paper) or, by default, calibrated
ONCE from the DU distribution early in training and then frozen (the
paper's CARLA thresholds 12-16 are on a different return scale).

Run (two shells), exactly like train_aqi_rlpd.py:
  python train_hiril.py --exp_name=sim_peg_insertion --learner \
      --demo_path=<demos.pkl> --checkpoint_path=<ckpt>
  python train_hiril.py --exp_name=sim_peg_insertion --actor \
      --checkpoint_path=<ckpt> --synthetic_operator=expert
Docs: docs/hiril_baseline.md.
"""

import json
import os
import pickle as pkl
import shutil
import signal
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics

from agentlace.data.data_store import QueuedDataStore
from agentlace.trainer import TrainerClient, TrainerServer

from serl_launcher.aqi.logger import AQILogger
from serl_launcher.data.data_store import RiskAwarePERDataStore
from serl_launcher.hiril.config import CLI_FIELDS, DU_MODES, HiRILConfig
from serl_launcher.hiril.priorities import RiskAwarePriorities
from serl_launcher.utils.launcher import (make_hiril_pixel_agent,
                                          make_sac_pixel_agent,
                                          make_trainer_config,
                                          make_wandb_logger)
from serl_launcher.utils.sharding import DeviceSharding
from serl_launcher.utils.timer_utils import Timer

from experiments.mappings import CONFIG_MAPPING
from synthetic_operator_env import wrap_synthetic_operator

FLAGS = flags.FLAGS

# ---- same names/semantics as train_aqi_rlpd.py (the campaign runner and
# the VS Code tasks pass them to both pipelines) ------------------------------
flags.DEFINE_string("exp_name", None, "Experiment folder name.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("learner", False, "Run the learner.")
flags.DEFINE_boolean("actor", False, "Run the actor.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_multi_string("demo_path", None, "Demo .pkl file(s) (learner).")
flags.DEFINE_string("checkpoint_path", None, "Checkpoint directory.")
flags.DEFINE_integer("max_steps", 0, "Override config.max_steps (actor: env-"
                     "step budget; learner: leave large, it stops via STOP).")
flags.DEFINE_float("max_minutes", 0.0, "Actor wall-clock budget (0 = none).")
flags.DEFINE_integer("buffer_period", -1, "Accepted for CLI compatibility; "
                     "HiRIL never dumps replay buffers to disk.")
flags.DEFINE_integer("checkpoint_keep", 100, "Periodic checkpoints kept.")
flags.DEFINE_integer("batch_size", 0, "Learner batch size (0 = config).")
flags.DEFINE_boolean("demo_pretrain", False, "Learner: demo-only warm start "
                     "(not part of the paper; off by default).")
flags.DEFINE_integer("demo_pretrain_steps", 0, "Demo warm-start steps.")
flags.DEFINE_integer("demo_pretrain_batch_size", 0, "0 = batch size.")
flags.DEFINE_float("converge_success", 0.0, "Actor early stop threshold on "
                   "the rolling (autonomous) success rate; 0 = off.")
flags.DEFINE_integer("converge_window", 20, "Episodes in the rolling window.")
flags.DEFINE_integer("converge_min_steps", 0, "No convergence before this.")
flags.DEFINE_boolean("converge_require_autonomous", True,
                     "Count only episodes with no operator takeover.")
flags.DEFINE_float("converge_eval_threshold", 0.0, "Fixed-pose mini-eval "
                   "gate for convergence (0 = off).")
flags.DEFINE_integer("converge_eval_max_steps", 300, "Mini-eval episode cap.")
flags.DEFINE_string("converge_eval_poses",
                    "0.000,0.000;0.005,0.000;0.010,0.000;0.005,0.080;"
                    "0.010,-0.080", "Mini-eval poses 'dx_m,dyaw_rad;...'.")
flags.DEFINE_enum("resume", "continue", ["continue", "scratch"],
                  "Learner: continue from the last checkpoint or delete it.")
flags.DEFINE_boolean("debug", False, "Disable wandb.")
flags.DEFINE_string("operator_id", "op0", "Operator id stored in the logs.")
flags.DEFINE_string("synthetic_operator", "", "Synthetic operator tier / "
                    "spec; '' = human (SpaceMouse).")
flags.DEFINE_string("synthetic_theta", "", "Exact operator theta JSON.")
flags.DEFINE_integer("synthetic_seed", 0, "Synthetic operator seed.")
flags.DEFINE_boolean("synthetic_latch_until_success", False, "Legacy latch.")
flags.DEFINE_enum("synthetic_intervention_mode", "legacy",
                  ["legacy", "signal", "always_latch",
                   "latch_until_autonomous_success"],
                  "Synthetic operator takeover/handback policy.")
flags.DEFINE_integer("env_seed", -1, "Training-env reset seed (-1 = seed).")
flags.DEFINE_string("oracle_checkpoint", "", "Policy-oracle checkpoint (a "
                    "hilserl/e2hil 2-critic SAC checkpoint).")
flags.DEFINE_enum("training_oracle", "auto", ["auto", "geometric", "policy"],
                  "Oracle providing the operator's corrective action.")
flags.DEFINE_float("synthetic_env_mm_per_unit", 100.0, "mm/s per action unit.")
flags.DEFINE_string("arm", "", "Campaign arm name ('' = 'hiril').")
flags.DEFINE_string("run_name", None, "Log directory name under aqi_log_dir.")
flags.DEFINE_string("aqi_log_dir", "", "Root of the JSONL logs ('' = "
                    "./aqi_logs).")
flags.DEFINE_enum("peg_shape", None, ["cylinder", "hex", "cuboid"],
                  "Sim peg geometry.")

# ---- HiRIL hyperparameters: one --hiril_<name> per HiRILConfig CLI field
# (None = HiRILConfig default; see serl_launcher/hiril/config.py for the
# paper source of every default) ---------------------------------------------
for _name, _cast in CLI_FIELDS.items():
    _help = f"HiRILConfig.{_name} (default {getattr(HiRILConfig(), _name)!r})"
    if _name == "du_mode":
        flags.DEFINE_enum(f"hiril_{_name}", None, list(DU_MODES), _help)
    elif _cast is bool:
        flags.DEFINE_boolean(f"hiril_{_name}", None, _help)
    elif _cast is int:
        flags.DEFINE_integer(f"hiril_{_name}", None, _help)
    elif _cast is float:
        flags.DEFINE_float(f"hiril_{_name}", None, _help)
    else:
        flags.DEFINE_string(f"hiril_{_name}", None, _help)

devices = jax.local_devices()
sharding = DeviceSharding(devices)
config = None


def print_green(x):
    print("\033[92m {}\033[00m".format(x), flush=True)


def print_yellow(x):
    print("\033[93m {}\033[00m".format(x), flush=True)


def hiril_config_from_flags() -> HiRILConfig:
    kw = {name: cast(getattr(FLAGS, f"hiril_{name}"))
          for name, cast in CLI_FIELDS.items()
          if getattr(FLAGS, f"hiril_{name}") is not None}
    return HiRILConfig(**kw)


def exploration_std(step: int, hiril: HiRILConfig) -> float:
    """Linear decay explore_init -> explore_final (Table A1) over
    explore_decay_steps env steps after the random warm-up."""
    t = max(0, step - config.random_steps) / max(hiril.explore_decay_steps, 1)
    return float(hiril.explore_final
                 + (hiril.explore_init - hiril.explore_final)
                 * max(0.0, 1.0 - t))


def _du(agent, obs, action, key):
    """DU statistics of ONE (s, a): dict of floats (du, dual, aleatoric,
    epistemic, q)."""
    obs_b = jax.tree_util.tree_map(lambda x: jnp.asarray(x)[None], obs)
    out = agent.dual_uncertainty(obs_b, jnp.asarray(action)[None], key)
    return {k: float(np.asarray(v)[0]) for k, v in out.items()}


def _shapes_match(a_state, b_state):
    try:
        return all(np.shape(x) == np.shape(y) for x, y in zip(
            jax.tree_util.tree_leaves(a_state.params),
            jax.tree_util.tree_leaves(b_state.params)))
    except Exception:
        return False


# ===========================================================================
# actor
# ===========================================================================
def actor(agent, data_store, env, sampling_rng, hiril: HiRILConfig):
    client = TrainerClient("actor_env", FLAGS.ip, make_trainer_config(),
                           data_stores={"actor_env": data_store},
                           wait_for_server=True, timeout_ms=3000)

    def update_params(params):
        nonlocal agent
        agent = agent.replace(state=agent.state.replace(params=params))

    client.recv_network_callback(update_params)

    theta = None
    synthetic = bool(FLAGS.synthetic_operator)
    if synthetic:
        def make_oracle_agent():
            return make_sac_pixel_agent(
                seed=FLAGS.seed, sample_obs=env.observation_space.sample(),
                sample_action=env.action_space.sample(),
                image_keys=config.image_keys,
                encoder_type=config.encoder_type, discount=config.discount)
        env, theta = wrap_synthetic_operator(
            env, synthetic_operator=FLAGS.synthetic_operator,
            synthetic_theta=FLAGS.synthetic_theta,
            synthetic_seed=FLAGS.synthetic_seed,
            operator_id=FLAGS.operator_id,
            training_oracle=FLAGS.training_oracle,
            oracle_checkpoint=FLAGS.oracle_checkpoint,
            make_oracle_agent=make_oracle_agent,
            mm_per_unit=FLAGS.synthetic_env_mm_per_unit,
            intervention_mode=FLAGS.synthetic_intervention_mode,
            latch_until_success=FLAGS.synthetic_latch_until_success,
            respond_to_request=True,      # HiRIL: interventions are risk-triggered
            max_traj_length=config.max_traj_length, log=print_green)

    arm = FLAGS.arm or "hiril"
    run_name = FLAGS.run_name or "{}_hiril_{}_s{}_{}".format(
        FLAGS.exp_name, FLAGS.operator_id, FLAGS.seed,
        time.strftime("%Y%m%d-%H%M%S"))
    log_dir = (os.path.abspath(FLAGS.aqi_log_dir) if FLAGS.aqi_log_dir
               else "aqi_logs")
    logger = AQILogger(log_dir, run_name, FLAGS.operator_id, "hiril",
                       config_dict={"arm": arm, "pipeline": "train_hiril",
                                    "hiril": hiril.as_dict(),
                                    **({"synthetic_operator": theta.as_dict()}
                                       if theta is not None else {})},
                       log_every_step=False)
    print_green(f"HiRIL actor; logs -> {logger.dir}")
    rd = RiskAwarePriorities(1, eps=hiril.per_eps)   # only for rd_bonus()

    env_seed = FLAGS.env_seed if FLAGS.env_seed >= 0 else FLAGS.seed
    try:
        env.action_space.seed(int(env_seed))
    except Exception:
        pass
    obs, _ = env.reset(seed=int(env_seed))

    # ---- risk threshold (Eq. 23): absolute, or calibrated once --------------
    # KNOWN LIMITATION (cluster method test 2026-09-26, handoff FINDINGS F2):
    # DU(s, a_RL) roughly halved every episode under the sparse reward
    # (0.46 -> 0.0005 in 3000 steps), so the quantile frozen at calib_end was
    # never exceeded again and the run made 0 requests. Requests are only
    # possible after calibration (see `request` below). Decide how HiRIL
    # should trigger before running a HiRIL campaign; do not change it
    # between cells of one campaign.
    tau_risk = hiril.tau_risk if hiril.tau_risk > 0 else None
    calib_start = config.random_steps + hiril.calib_delay_steps
    calib_end = calib_start + hiril.calib_steps
    calib_vals = []
    if tau_risk is None:
        print_green(f"tau_risk will be calibrated as the "
                    f"{hiril.calib_quantile:.2f}-quantile of DU(s, a_RL) over "
                    f"env steps [{calib_start}, {calib_end}) and then frozen")

    # ---- convergence / budget bookkeeping (as train_aqi_rlpd) --------------
    converge_on = FLAGS.converge_success > 0.0
    converge_eval_on = FLAGS.converge_eval_threshold > 0.0
    eval_poses = [tuple(float(v) for v in p.split(","))
                  for p in FLAGS.converge_eval_poses.split(";") if p.strip()]
    recent, episodes_total, converged = [], 0, False
    t_start = time.time()
    max_seconds = FLAGS.max_minutes * 60.0 if FLAGS.max_minutes > 0 else 0.0
    timed_out = False

    def mini_eval(eval_env, key):
        """Fixed-pose autonomous check (the ruler's reset semantics)."""
        successes = 0
        try:
            for dx, dyaw in eval_poses:
                eval_env.set_eval_reset_pose(dx, dyaw)
                o, _ = eval_env.reset()
                done_e, n, r_e = False, 0, 0.0
                while not done_e and n < FLAGS.converge_eval_max_steps:
                    key, k = jax.random.split(key)
                    a = np.asarray(jax.device_get(agent.sample_actions(
                        observations=jax.device_put(o), seed=k, argmax=True)))
                    o, r_e, d, tr, _ = eval_env.step(a)
                    done_e, n = bool(d or tr), n + 1
                successes += int(r_e > 0)
        finally:
            eval_env.set_eval_reset_pose(None, None)
        return successes / float(len(eval_poses))

    # ---- per-episode state ----------------------------------------------------
    def new_episode():
        return dict(ret=0.0, n_int_steps=0, n_onsets=0, n_request_steps=0,
                    n_requests=0, du=[], seg=None, segments=[], step0=step_i,
                    prev_request=False)

    timer = Timer()
    step_i = 0
    ep = new_episode()
    ep_idx, prev_intervening = 0, False
    pbar = tqdm.tqdm(range(0, config.max_steps), dynamic_ncols=True)
    for step in pbar:
        step_i = step
        timer.tick("total")
        if max_seconds and time.time() - t_start >= max_seconds:
            timed_out = True
            print_yellow(f"\n>>> TIME BUDGET reached at step {step}; stopping.")
            break

        # line 5: a_RL = pi(s) + exploration noise (Eq. 21)
        sampling_rng, k_act, k_du, k_duh = jax.random.split(sampling_rng, 4)
        if step < config.random_steps:
            a_rl = env.action_space.sample()
        else:
            a_rl = np.asarray(jax.device_get(agent.sample_actions(
                observations=jax.device_put(obs), seed=k_act,
                noise_std=exploration_std(step, hiril))))

        # line 6-10: DU(s, a_RL) and the risk-triggered request (Eq. 23)
        with timer.context("dual_uncertainty"):
            du_rl = _du(agent, obs, a_rl, k_du)
        if tau_risk is None:
            if calib_start <= step < calib_end:
                calib_vals.append(du_rl["du"])
            elif step >= calib_end and calib_vals:
                tau_risk = float(np.quantile(calib_vals,
                                             hiril.calib_quantile))
                print_green(f">>> tau_risk calibrated = {tau_risk:.6g} "
                            f"({len(calib_vals)} DU samples, q="
                            f"{hiril.calib_quantile})")
                logger.save_state({"tau_risk": tau_risk,
                                   "calib_n": len(calib_vals),
                                   "calib_quantile": hiril.calib_quantile},
                                  {}, extra={"step": step})
        request = bool(tau_risk is not None and step >= config.random_steps
                       and du_rl["du"] > tau_risk)
        if synthetic:
            env.notify_request(request, du_rl["du"])
        elif request and not ep["prev_request"]:
            print_yellow(f"\a>>> HiRIL REQUESTS INTERVENTION (DU="
                         f"{du_rl['du']:.3g} > {tau_risk:.3g})")
        ep["n_request_steps"] += int(request)
        ep["n_requests"] += int(request and not ep["prev_request"])
        ep["prev_request"] = request
        ep["du"].append(du_rl["du"])

        # line 11: execute (the operator overrides during a takeover)
        with timer.context("step_env"):
            next_obs, reward, done, truncated, info = env.step(a_rl)
        for side in ("left", "right"):
            info.pop(side, None)
        intervening = "intervene_action" in info
        a_exec = (np.asarray(info.pop("intervene_action"), dtype=np.float32)
                  if intervening else np.asarray(a_rl, dtype=np.float32))
        onset = intervening and not prev_intervening
        ep["ret"] += float(reward)      # RAW reward: success detection

        # line 12: one-off intervention penalty at takeover onset (Eq. 25)
        stored_reward = float(reward) + (hiril.intervention_penalty
                                         if onset else 0.0)
        # line 13: the RD term of Eq. 28, fixed at collection
        bonus, du_h = 0.0, None
        if intervening:
            du_h = _du(agent, obs, a_exec, k_duh)
            if hiril.use_rd:
                bonus = rd.rd_bonus(du_rl["du"], du_h["du"], hiril.rd_clip)
        data_store.insert(dict(
            observations=obs, actions=a_exec, next_observations=next_obs,
            rewards=np.float32(stored_reward), masks=1.0 - done, dones=done,
            hiril_human=np.float32(intervening),
            hiril_bonus=np.float32(bonus)))

        # takeover segments (logged like AQI segments; readable by the
        # quality audit / q_summary)
        t_ep = step - ep["step0"]
        if intervening:
            ep["n_int_steps"] += 1
            if onset:
                ep["n_onsets"] += 1
                ep["seg"] = dict(start_step=t_ep, requested=request,
                                 du_rl_at_trigger=du_rl["du"], du_h=[],
                                 du_rl=[], rd=[], length=0)
            s = ep["seg"]
            s["length"] += 1
            s["du_rl"].append(du_rl["du"])
            s["du_h"].append(du_h["du"])
            s["rd"].append(bonus)
        elif ep["seg"] is not None:
            ep["segments"].append(ep["seg"])
            ep["seg"] = None
        prev_intervening = intervening
        obs = next_obs

        if done or truncated:
            if ep["seg"] is not None:
                ep["segments"].append(ep["seg"])
            success = bool(ep["ret"] > 0)
            length = t_ep + 1
            for s in ep["segments"]:
                logger.log_segment(dict(
                    episode=ep_idx, operator_id=FLAGS.operator_id,
                    start_step=s["start_step"], length=s["length"],
                    requested=bool(s["requested"]), excluded=False,
                    episode_success=success,
                    du_rl_at_trigger=s["du_rl_at_trigger"],
                    du_rl_mean=float(np.mean(s["du_rl"])),
                    du_h_mean=float(np.mean(s["du_h"])),
                    rd_bonus_mean=float(np.mean(s["rd"])),
                    features={"t_int": s["length"]}))
            du_arr = np.asarray(ep["du"], dtype=np.float64)
            ep_rec = dict(
                episode=ep_idx, operator_id=FLAGS.operator_id, mode="hiril",
                success=success, episode_return=ep["ret"], length=length,
                n_segments=len(ep["segments"]),
                n_intervention_steps=ep["n_int_steps"],
                n_requests=ep["n_requests"],
                n_request_steps=ep["n_request_steps"],
                du_mean=float(du_arr.mean()), du_max=float(du_arr.max()),
                tau_risk=tau_risk,
                explore_std=exploration_std(step, hiril), step=int(step))
            logger.log_episode(ep_rec)
            if FLAGS.checkpoint_path:    # mode-independent takeover log
                try:
                    with open(os.path.join(os.path.abspath(
                            FLAGS.checkpoint_path), "episodes.jsonl"),
                            "a") as f:
                        f.write(json.dumps({
                            "step": int(step),
                            "intervention_count": ep["n_onsets"],
                            "intervention_steps": ep["n_int_steps"],
                            "success": success, "length": length}) + "\n")
                except OSError:
                    pass

            if converge_on:
                episodes_total += 1
                counted = success and (not FLAGS.converge_require_autonomous
                                       or ep["n_onsets"] == 0)
                recent = (recent + [1.0 if counted else 0.0])[
                    -FLAGS.converge_window:]
                roll = sum(recent) / len(recent)
                if (len(recent) >= FLAGS.converge_window
                        and step >= FLAGS.converge_min_steps
                        and roll >= FLAGS.converge_success):
                    ruler_sr = None
                    if converge_eval_on:
                        ruler_sr = mini_eval(getattr(env, "env", env),
                                             sampling_rng)
                        if ruler_sr < FLAGS.converge_eval_threshold:
                            print_yellow(
                                f"\n>>> rolling convergence at step {step} "
                                f"but mini-eval SR {ruler_sr:.2f} < "
                                f"{FLAGS.converge_eval_threshold:.2f}; "
                                f"continuing.")
                    if not converge_eval_on or \
                            ruler_sr >= FLAGS.converge_eval_threshold:
                        converged = True
                        _write_convergence(agent, step, episodes_total, roll,
                                           ruler_sr, eval_poses)

            info.setdefault("episode", {})
            info["episode"]["intervention_count"] = ep["n_onsets"]
            info["episode"]["intervention_steps"] = ep["n_int_steps"]
            client.request("send-stats", {
                "environment": info,
                "hiril": {"du_mean": float(du_arr.mean()),
                          "request_rate": ep["n_request_steps"] / length,
                          "intervention_rate": ep["n_int_steps"] / length,
                          "tau_risk": tau_risk if tau_risk else 0.0,
                          "explore_std": exploration_std(step, hiril)}})
            pbar.set_description(f"last return: {ep['ret']}")
            client.update()
            obs, _ = env.reset()
            ep_idx += 1
            ep = new_episode()
            ep["step0"] = step + 1
            prev_intervening = False

        if converged:
            break
        timer.tock("total")
        if step % config.log_period == 0:
            client.request("send-stats", {"timer": timer.get_average_times()})

    if converge_on and not converged and not timed_out:
        print_yellow(f"\n>>> actor reached the {config.max_steps}-step budget "
                     f"WITHOUT converging (no CONVERGED.json).")
    logger.close()
    _document_actor_run(theta, converged, timed_out, t_start, step, hiril)
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(0)


def _write_convergence(agent, step, episodes, roll, ruler_sr, eval_poses):
    """CONVERGED.json + certified/ (the exact actor weights that converged),
    as train_aqi_rlpd writes them, so run_campaign and eval_ruler treat the
    cell the same way."""
    print_green(f"\n>>> CONVERGED at step {step}: rolling autonomous success "
                f"{roll:.2f} over {FLAGS.converge_window} episodes.")
    if not FLAGS.checkpoint_path:
        return
    ck = os.path.abspath(FLAGS.checkpoint_path)
    payload = dict(step=int(step), episodes=int(episodes),
                   rolling_success=float(roll),
                   window=int(FLAGS.converge_window),
                   threshold=float(FLAGS.converge_success),
                   autonomous=bool(FLAGS.converge_require_autonomous))
    try:
        checkpoints.save_checkpoint(
            os.path.join(ck, "certified"),
            jax.block_until_ready(agent.state),
            step=int(step), keep=1, overwrite=True)
        payload["certified_checkpoint_step"] = int(step)
    except Exception as exc:
        print_yellow(f">>> could not save certified checkpoint: {exc}")
    if ruler_sr is not None:
        payload.update(mini_eval_sr=float(ruler_sr),
                       mini_eval_threshold=float(FLAGS.converge_eval_threshold),
                       mini_eval_poses=[list(p) for p in eval_poses])
    try:
        with open(os.path.join(ck, "CONVERGED.json"), "w") as f:
            json.dump(payload, f, indent=2)
    except OSError:
        pass


def _document_actor_run(theta, converged, timed_out, t_start, step, hiril):
    if not FLAGS.checkpoint_path:
        return
    try:
        import provenance as prov
        op = ({"operator_id": FLAGS.operator_id, "human": True}
              if theta is None else
              {k: theta.as_dict().get(k) for k in (
                  "operator_id", "tier", "s_exec", "s_mon", "d_max",
                  "delta_in", "p_miss", "l_react")})
        prov.document(
            os.path.abspath(FLAGS.checkpoint_path), "run", quiet=True,
            task={"exp_name": FLAGS.exp_name,
                  "peg_shape": getattr(config, "peg_shape", None)},
            method=FLAGS.arm or "hiril", operator=op,
            seeds={"seed": FLAGS.seed, "synthetic_seed": FLAGS.synthetic_seed,
                   "env_seed": (FLAGS.env_seed if FLAGS.env_seed >= 0
                                else FLAGS.seed)},
            termination={"cap": config.max_steps, "converged": converged,
                         "reason": ("converged" if converged else
                                    "time_budget" if timed_out else
                                    "step_cap"),
                         "wall_minutes": round((time.time() - t_start) / 60, 2),
                         **({"step": int(step)} if converged else {})},
            conditions=prov.hiril_condition_rows(hiril.as_dict()))
    except Exception:
        pass


# ===========================================================================
# learner
# ===========================================================================
def learner(agent, replay, n_demo, hiril: HiRILConfig, wandb_logger=None):
    latest = (checkpoints.latest_checkpoint(os.path.abspath(
        FLAGS.checkpoint_path))
        if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path)
        else None)
    step = int(os.path.basename(latest)[11:]) + 1 if latest else 0
    batch_size = FLAGS.batch_size or config.batch_size

    def stats_callback(type, payload):
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=step)
        return {}

    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", replay)
    server.start(threaded=True)

    stop = {"flag": False}

    def request_stop(signum, _frame):
        if stop["flag"]:
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            raise KeyboardInterrupt
        stop["flag"] = True
        print_yellow(f"\nstop requested (signal {signum}); checkpointing.")
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    stop_file = (os.path.join(os.path.abspath(FLAGS.checkpoint_path), "STOP")
                 if FLAGS.checkpoint_path else None)

    def should_stop():
        return stop["flag"] or (stop_file is not None
                                and os.path.exists(stop_file))

    def save(overwrite=False):
        if FLAGS.checkpoint_path:
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path), agent.state, step=step,
                keep=FLAGS.checkpoint_keep, overwrite=overwrite)

    critic_only = frozenset({"critic"})
    full = frozenset({"critic", "actor"})

    def train_step(batch, networks):
        """One update + the TDRD priority refresh of the sampled slots."""
        nonlocal agent
        idx = np.asarray(jax.device_get(batch["per_index"]))
        agent, info = agent.update(batch, networks_to_update=networks)
        replay.update_priorities(
            idx, np.asarray(jax.device_get(info["critic"].pop("td_abs"))))
        return info

    # optional demo warm start (NOT in the paper; off unless asked)
    pretrain = max(int(FLAGS.demo_pretrain_steps), 0) if FLAGS.demo_pretrain \
        else 0
    if pretrain > 0:
        bs = FLAGS.demo_pretrain_batch_size or batch_size
        print_green(f">>> demo pretrain: {pretrain} steps, batch {bs}")
        it = replay.get_iterator(sample_args={"batch_size": bs,
                                              "pack_obs_and_next_obs": True},
                                 device=sharding.replicate())
        for _ in tqdm.tqdm(range(pretrain), desc="demo-pretrain"):
            if should_stop():
                break
            for _c in range(hiril.policy_delay - 1):
                train_step(next(it), critic_only)
            train_step(next(it), full)
            step += 1

    bar = "=" * 68
    print_green("\a" + bar)
    print_green(">>> LEARNER READY -- start the ACTOR now.")
    print_green(f">>> HiRIL: training starts once the actor has streamed "
                f"{config.training_starts} online steps.")
    if stop_file:
        print_green(f">>> Stop + save anytime:  touch {stop_file}")
    print_green(bar)
    print("", end="", flush=True)

    while len(replay) - n_demo < config.training_starts and not should_stop():
        time.sleep(1)
    if should_stop():
        if stop_file and os.path.exists(stop_file):
            os.remove(stop_file)
        save(overwrite=True)
        return
    server.publish_network(agent.state.params)
    print_green("sent initial network to actor")

    it = replay.get_iterator(sample_args={"batch_size": batch_size,
                                          "pack_obs_and_next_obs": True},
                             device=sharding.replicate())
    timer = Timer()
    for step in tqdm.tqdm(range(step, config.max_steps), dynamic_ncols=True,
                          desc="learner"):
        # TD3: policy_delay-1 critic-only updates, then critic + actor
        for _c in range(hiril.policy_delay - 1):
            with timer.context("train_critic"):
                train_step(next(it), critic_only)
        with timer.context("train"):
            info = train_step(next(it), full)
        if step > 0 and step % config.steps_per_update == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)
        if step % config.log_period == 0 and wandb_logger:
            wandb_logger.log(info, step=step)
            wandb_logger.log({"timer": timer.get_average_times(),
                              "hiril_replay": replay.priority_stats()},
                             step=step)
        if step > 0 and config.checkpoint_period and \
                step % config.checkpoint_period == 0:
            save()
        if should_stop():
            print_green(f"\ngraceful stop at step {step}")
            agent = jax.block_until_ready(agent)
            save(overwrite=True)
            if stop_file and os.path.exists(stop_file):
                os.remove(stop_file)
            break
    if wandb_logger is not None:
        wandb_logger.finish()


# ===========================================================================
# main
# ===========================================================================
def main(_):
    global config
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    if FLAGS.peg_shape is not None:
        config.peg_shape = FLAGS.peg_shape
    if FLAGS.max_steps > 0:
        config.max_steps = FLAGS.max_steps
    hiril = hiril_config_from_flags()
    if config.setup_mode not in ("single-arm-fixed-gripper",
                                 "dual-arm-fixed-gripper"):
        raise NotImplementedError(
            f"train_hiril.py supports fixed-gripper setups, not "
            f"'{config.setup_mode}'")
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)
    if FLAGS.actor and FLAGS.synthetic_operator:
        os.environ["SIM_DISABLE_SPACEMOUSE"] = "1"
    env = RecordEpisodeStatistics(config.get_environment(
        fake_env=FLAGS.learner, save_video=False, classifier=True))

    agent = make_hiril_pixel_agent(
        seed=FLAGS.seed, sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(), image_keys=config.image_keys,
        encoder_type=config.encoder_type, discount=config.discount,
        hiril=hiril)
    agent = jax.device_put(jax.tree_util.tree_map(jnp.array, agent),
                           sharding.replicate())
    print_green(f"HiRIL agent: K_tau={hiril.k_tau} K_theta={hiril.k_theta} "
                f"du_mode={hiril.du_mode} lambda_bc={hiril.lambda_bc} "
                f"r_p={hiril.intervention_penalty} tau_risk="
                f"{hiril.tau_risk if hiril.tau_risk > 0 else 'calibrated'}")

    ck = os.path.abspath(FLAGS.checkpoint_path) if FLAGS.checkpoint_path \
        else None
    if ck is not None:
        has_ckpt = os.path.exists(ck) and \
            checkpoints.latest_checkpoint(ck) is not None
        if FLAGS.learner and FLAGS.resume == "scratch" and os.path.exists(ck):
            if ck in {os.path.abspath(os.sep), os.path.abspath(
                    os.path.expanduser("~")), os.path.abspath(os.getcwd())}:
                raise SystemExit(f"refusing to delete '{ck}'")
            shutil.rmtree(ck)
            print_yellow(f"retrain from scratch: deleted {ck}")
            has_ckpt = False
        if has_ckpt:
            try:
                restored = checkpoints.restore_checkpoint(ck, agent.state)
            except Exception:
                restored = None
            if restored is not None and _shapes_match(agent.state, restored):
                agent = agent.replace(state=restored)
                print_green("resumed from the latest checkpoint")
            elif FLAGS.learner:
                raise SystemExit(f"cannot resume {ck}: incompatible "
                                 f"checkpoint; use --resume=scratch")
        if FLAGS.learner:
            # lets eval_ruler (--agent_type auto) rebuild this architecture
            os.makedirs(ck, exist_ok=True)
            with open(os.path.join(ck, "agent_type.json"), "w") as f:
                json.dump({"agent_type": "hiril", "hiril": hiril.as_dict(),
                           "encoder_type": config.encoder_type,
                           "image_keys": list(config.image_keys),
                           "discount": config.discount}, f, indent=2)

    if FLAGS.learner:
        sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())
        replay = RiskAwarePERDataStore(
            env.observation_space, env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys, per_alpha=hiril.per_alpha,
            per_beta=hiril.per_beta, per_eps=hiril.per_eps)
        # offline demos are human experience (D2); their a_RL was never
        # observed, so their RD term is the neutral demo_bonus (= exp(0))
        assert FLAGS.demo_path, "--demo_path is required for the learner"
        for path in FLAGS.demo_path:
            with open(path, "rb") as f:
                for tr in pkl.load(f):
                    tr = dict(tr)
                    tr.pop("infos", None)
                    tr["hiril_human"] = np.float32(1.0)
                    tr["hiril_bonus"] = np.float32(
                        hiril.demo_bonus if hiril.use_rd else 0.0)
                    replay.insert(tr)
        n_demo = len(replay)
        print_green(f"replay buffer: {n_demo} demo transitions (human, D2)")
        wandb_logger = make_wandb_logger(
            project="hil-serl",
            description=f"{FLAGS.exp_name}_{FLAGS.arm or 'hiril'}_"
                        f"{FLAGS.operator_id}",
            debug=FLAGS.debug,
            output_dir=os.path.join(ck, "wandb") if ck else None)
        learner(agent, replay, n_demo, hiril, wandb_logger)
    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        actor(agent, QueuedDataStore(50000), env, sampling_rng, hiril)
    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    app.run(main)
