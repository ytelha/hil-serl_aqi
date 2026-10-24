#!/usr/bin/env python3
"""AQI-HIL-SERL training (Active and Quality-Aware Intervention HIL-SERL).

This is examples/train_rlpd.py extended with the method of the research
proposal "Active and Quality-Aware Human-in-the-Loop RL for Operator-Robust
Robotic Insertion". The five run modes map 1:1 to the proposal's baselines
and phases (S6.2/S6.4); --aqi_mode selects them:

    hilserl   plain HIL-SERL. Byte-identical behaviour to train_rlpd.py
              (2 critics, no gate, no logging). Primary baseline.
    logged    same training as hilserl but with the K-critic ensemble and
              full trigger/intervention logging (Phase 2 diagnostic control;
              feeds the RQ1/RQ2 offline analysis in aqi_analysis.py).
    active    + active query gate A(s_t) (S5.2-S5.4). Replay/loss standard.
    quality   manual interventions, but interventions are scored by the
              counterfactual advantage (S5.5, Option B), filtered by the
              force veto / outcome filter, given quality-weighted replay
              priority and a weighted imitation loss (S5.6).
    aqi       active + quality. Full proposed system (Phase 5).
    e2hil     EXTERNAL comparison baseline (Deng et al., RA-L 2026,
              arXiv:2601.19969): hilserl + the entropy-bounded actor loss
              (agents/continuous/e2hil.py). Actor side identical to hilserl;
              knobs --e2hil_K / --e2hil_q_low / --e2hil_q_high / --e2hil_q_agg.

Ablations from S6.4 are expressed through the experiment's get_aqi_config()
(e.g. lambda_force=0 for "no force term", lambda_ood=0 for "no OOD term";
mode=active is "no q_int").

Extra flags vs train_rlpd.py: --aqi_mode, --operator_id (logged with every
record for the multi-operator variance analysis), --run_name.
"""

import glob
import time
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
import os
import sys
import signal
import copy
import shutil
import dataclasses
import json
import pickle as pkl
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from natsort import natsorted

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from serl_launcher.agents.continuous.sac_hybrid_dual import SACAgentHybridDualArm
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.utils.update_ratio import UpdateRatio
from serl_launcher.utils.train_utils import concat_batches
from serl_launcher.utils.sharding import DeviceSharding
from serl_launcher.utils.obs_view import compose_obs_view, resolve_state_slices

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from serl_launcher.utils.launcher import (
    make_sac_pixel_agent,
    make_sac_pixel_agent_hybrid_single_arm,
    make_sac_pixel_agent_hybrid_dual_arm,
    make_trainer_config,
    make_wandb_logger,
)
from serl_launcher.data.data_store import (
    MemoryEfficientReplayBufferDataStore,
    QualityAwareReplayBufferDataStore,
)
from serl_launcher.aqi import AQIRuntime, MODES

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("learner", False, "Whether this is a learner.")
flags.DEFINE_boolean("actor", False, "Whether this is an actor.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_multi_string("demo_path", None, "Path to the demo data.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_integer("eval_checkpoint_step", 0, "Step to evaluate the checkpoint.")
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
flags.DEFINE_boolean("save_video", False, "Save video.")
flags.DEFINE_integer("max_steps", 0,
                     "Override the experiment's config.max_steps (0 = use the "
                     "default). The campaign runner sets this as the per-cell "
                     "budget: on the ACTOR it caps env steps (the interaction "
                     "budget); leave the LEARNER large so it stops via the STOP "
                     "sentinel when the actor finishes.")
flags.DEFINE_float("max_minutes", 0.0,
                   "ACTOR wall-clock training budget in minutes (0 = no time "
                   "limit). A fixed time budget guarantees each cell finishes: "
                   "the actor stops gracefully at whichever comes first -- "
                   "convergence, the --max_steps cap, or this time budget -- "
                   "then the learner checkpoints via the STOP sentinel.")
flags.DEFINE_integer("buffer_period", -1,
                     "Override config.buffer_period: dump the replay/demo "
                     "transitions (with images, ~9GB/cell over a full run) to "
                     "<checkpoint_path>/buffer every N steps. 0 disables the "
                     "dumps entirely (only needed to warm-restart a crashed "
                     "learner's replay buffer); <0 keeps the experiment "
                     "default. Set 0 for batch campaigns to save disk.")
flags.DEFINE_integer("checkpoint_keep", 100,
                     "How many periodic checkpoints to retain (orbax 'keep'). "
                     "The default 100 keeps every 1000-step snapshot of a 50k "
                     "run (~10GB/cell); best_window eval only needs enough for "
                     "a stable window, so a smaller value saves disk.")
flags.DEFINE_integer(
    "demo_pretrain_steps", 0,
    "LEARNER only: number of demo-only gradient steps to run before waiting "
    "for online replay data (0 = disabled). Uses only demo/intervention "
    "buffer batches and publishes the warmed network before actor training. "
    "These steps advance the learner step counter.")
flags.DEFINE_boolean(
    "demo_pretrain", False,
    "LEARNER only: enable demo-only pretraining before online RL. "
    "When false, demo_pretrain_steps is ignored.")
flags.DEFINE_integer(
    "demo_pretrain_batch_size", 0,
    "LEARNER only: batch size for demo pretraining (0 = use config.batch_size).")
flags.DEFINE_float(
    "learning_rate", 0.0,
    "LEARNER: override the actor/critic/temperature Adam learning rate "
    "(0 = built-in 3e-4). Halving to 1.5e-4 trades speed for stability "
    "against the 0<->100% policy oscillation.")
flags.DEFINE_float("converge_success", 0.0,
                   "ACTOR early-stop: if >0, stop training once the rolling "
                   "success rate over the last --converge_window episodes "
                   "reaches this threshold (e.g. 0.9). --max_steps stays the "
                   "hard cap for cells that never converge. Writes "
                   "<checkpoint_path>/CONVERGED.json with the stopping step so "
                   "'steps-to-convergence' can be compared across operators.")
flags.DEFINE_integer("converge_window", 20,
                     "episodes in the rolling success window for --converge_"
                     "success.")
flags.DEFINE_integer("converge_min_steps", 0,
                     "do not declare convergence before this many env steps "
                     "(avoids stopping during the random-action warmup).")
flags.DEFINE_boolean("converge_require_autonomous", True,
                     "count only AUTONOMOUS successes (episode succeeded with "
                     "no operator takeover) toward convergence -- this matches "
                     "the ruler's autonomous-SR definition and means 'the "
                     "policy no longer needs the operator', not 'the operator "
                     "keeps rescuing it'.")
flags.DEFINE_float(
    "converge_eval_threshold", 0.0,
    "optional ruler-aligned convergence gate (0 = disabled): when > 0, "
    "the actor declares convergence only if a fixed-pose autonomous mini-eval "
    "also reaches this success-rate threshold.")
flags.DEFINE_integer(
    "converge_eval_max_steps", 300,
    "max env steps per trajectory in the fixed-pose mini-eval used by "
    "--converge_eval_threshold.")
flags.DEFINE_string(
    "converge_eval_poses", "0.000,0.000;0.005,0.000;0.010,0.000;0.005,0.080;0.010,-0.080",
    "semicolon-separated fixed reset poses for the convergence mini-eval: "
    "each pose is 'dx_m,dyaw_rad'.")
flags.DEFINE_enum(
    "converge_mode", "rolling", ["rolling", "eval"],
    "'rolling' = the original rule (--converge_success over the last "
    "--converge_window training episodes, then the optional mini-eval gate); "
    "'eval' = decide convergence with the autonomous fixed-pose mini-eval "
    "alone: every --converge_eval_every env steps (at episode ends, once the "
    "policy has succeeded without help at least once) and stop after "
    "--converge_eval_consecutive passes of --converge_eval_threshold in a row. "
    "Independent of how often the operator steps in during training.")
flags.DEFINE_integer("converge_eval_every", 2000,
                     "--converge_mode eval: env steps between mini-evals.")
flags.DEFINE_integer("converge_eval_consecutive", 2,
                     "--converge_mode eval: passes in a row needed to stop.")
flags.DEFINE_float(
    "utd_ratio", 0.0,
    "> 0: fix the learner steps per env step at this value (each learner "
    "step = cta_ratio-1 critic updates + 1 critic+actor update). The actor "
    "waits for the learner and the learner for data, through "
    "<checkpoint_path>/learner_progress.json, and the env runs without "
    "real-time pacing. 0 = the original asynchronous loop (the ratio then "
    "depends on the GPU, finding F3).")
flags.DEFINE_float("utd_timeout_s", 600.0,
                   "--utd_ratio: the actor stops waiting if the learner has "
                   "not advanced for this long (warning; no hang).")
flags.DEFINE_boolean(
    "log_takeover_channels", False,
    "Synthetic operator diagnostics: record, per episode in episodes.jsonl, "
    "which monitor channel caused each takeover (time, corridor, stagnation, "
    "hover, off_bore, force, force_excess, false_alarm, request) and how each "
    "takeover ended (signal, time_cap, latch_success, episode_end). Logging "
    "only; training is unchanged.")
flags.DEFINE_enum(
    "resume", "continue", ["continue", "scratch"],
    "Learner start policy when a checkpoint for this task+method already "
    "exists: 'continue' carries on from the last checkpoint (resume where we "
    "stopped); 'scratch' DELETES all previous checkpoints for this run and "
    "trains a new network from zero. Ignored while evaluating "
    "(--eval_checkpoint_step).")

flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging

# ---- AQI-HIL-SERL flags ----------------------------------------------------
flags.DEFINE_enum("aqi_mode", "aqi", list(MODES),
                  "hilserl | logged | active | quality | aqi (proposal S6.4).")
flags.DEFINE_string("operator_id", "op0",
                    "Identifier of the human operator for this run; stored "
                    "with every logged step/segment/episode (RQ1/RQ2).")
flags.DEFINE_string("synthetic_operator", "",
                    "run the actor with a synthetic operator instead of "
                    "a human: tier name, 'op1'..'op4', 's=0.7', or "
                    "'exec=expert,mon=novice' (methodology doc Part A)")
flags.DEFINE_string("synthetic_theta", "",
                    "path to a JSON operator-theta file (written by "
                    "operator_model.theta_to_json). When set it overrides "
                    "--synthetic_operator's tier anchor with an EXACT theta, "
                    "so a jittered operator drives demos and interventions "
                    "identically. The campaign runner always passes this.")
flags.DEFINE_integer("synthetic_seed", 0,
                     "seed for the synthetic operator (R6 determinism)")
flags.DEFINE_boolean("synthetic_latch_until_success", False,
                     "When true, once the synthetic operator takes over, it "
                     "keeps intervening until the episode ends successfully.")
flags.DEFINE_enum(
    "synthetic_intervention_mode", "legacy",
    ["legacy", "signal", "always_latch", "latch_until_autonomous_success"],
    "Synthetic actor intervention policy: 'signal' = take over/release via "
    "the normal signal gate; 'always_latch' = once takeover starts, keep "
    "control until success; 'latch_until_autonomous_success' = use latch "
    "policy until the policy records its first successful fully autonomous "
    "episode, then switch permanently to signal mode. 'legacy' preserves "
    "the historical --synthetic_latch_until_success boolean behavior.")
flags.DEFINE_boolean(
    "synthetic_rotation_control", False,
    "Geometric synthetic operator also corrects the tool orientation (roll, "
    "pitch, yaw back to the nearest insertable orientation) and commits only "
    "once position and orientation are aligned. Off = the original "
    "translation-only operator, so finished campaigns stay reproducible; "
    "needed for pegs that must be turned into the bore (cuboid, hex).")
flags.DEFINE_integer("env_seed", -1,
                     "Seed for the TRAINING env's reset randomization (start "
                     "pose / yaw) and random-exploration sampler; -1 = use "
                     "--seed. Varying this across otherwise-identical runs is "
                     "how the pipeline induces WITHIN-operator variance "
                     "(independent reset sequences per replicate). The eval "
                     "ruler is unaffected -- it applies its own fixed grid.")
flags.DEFINE_string("oracle_checkpoint", "",
                    "frozen policy checkpoint used as the competent core "
                    "pi_or when the env has no phase_oracle attribute "
                    "(doc S4 option B)")
flags.DEFINE_enum("training_oracle", "auto", ["auto", "geometric", "policy"],
                  "Which oracle drives the synthetic operator's intervention "
                  "ACTION during training. 'auto' (default) = the env's "
                  "geometric oracle if it attaches one (sim), else the "
                  "--oracle_checkpoint policy. 'policy' FORCES the frozen "
                  "--oracle_checkpoint even in the sim (the geometric monitor "
                  "still times the takeovers); 'geometric' forces the env "
                  "oracle.")
flags.DEFINE_float("synthetic_env_mm_per_unit", 100.0,
                   "mm/s of tool motion per env action unit; converts the "
                   "operator's Table-2 motor parameters (mm/s) to the env's "
                   "action scale. Measured for sim_peg_insertion: 1 action "
                   "unit = 0.01 m delta/step at 10 Hz = 100 mm/s.")
flags.DEFINE_string("arm", "",
                    "Name of the campaign ARM this run belongs to when it "
                    "differs from --aqi_mode (e.g. 'quality-beh' = quality + "
                    "behavioral score). Recorded in meta.json and the wandb "
                    "run name so analyses group by arm, not by mode. Set by "
                    "run_campaign.py; '' = the arm is the mode.")
flags.DEFINE_string("run_name", None,
                    "Log-directory name under <aqi log_dir>/. Defaults to "
                    "<exp_name>_<mode>_<operator>_s<seed>_<timestamp>.")
flags.DEFINE_string("aqi_log_dir", "",
                    "Root directory for the AQI JSONL logs (steps/segments/"
                    "episodes). Default '' keeps the config's 'aqi_logs' "
                    "(relative to CWD). The campaign runner sets this to "
                    "<campaign>/aqi_logs so each campaign's q_int logs are "
                    "colocated and never mix across campaigns.")
# Composition overrides (take precedence over the experiment's
# get_aqi_config when passed explicitly; see aqi/config.py):
flags.DEFINE_string("trigger_terms", None,
                    "Comma-separated trigger sections feeding A(s_t) out of "
                    "{uncertainty,failure,force,ood,cost}. "
                    "'uncertainty' alone = pure ensemble-epistemic trigger.")
flags.DEFINE_enum("trigger_combiner", None, ["weighted_sum", "max"],
                  "Logic combining the selected terms: proposal weighted "
                  "sum, or max (= request when ANY weighted term > tau_A).")
flags.DEFINE_boolean("quality_disagreement_gate", None,
                     "Apply g(sigma_Q) to the counterfactual advantage. "
                     "False = quality signal is A_hat alone.")
flags.DEFINE_float("bootstrap_p", None,
                   "Bernoulli keep-probability of the persistent per-"
                   "(transition, member) bootstrap masks; 1.0 = shared "
                   "data (init-only ensemble ablation).")
flags.DEFINE_boolean("per_member_targets", None,
                     "Independent target draw per ensemble member; "
                     "false = single shared REDQ target (ablation).")
flags.DEFINE_integer("critic_ensemble_size", None,
                     "Number of critics K in the ensemble (S5.3). More critics "
                     "give a better epistemic sigma_Q but cost more compute / GPU "
                     "memory. Ignored in hilserl mode (fixed at 2).")
flags.DEFINE_integer("critic_subsample_size", None,
                     "REDQ target: number M of critics subsampled for the "
                     "min-Q target (smaller = LESS pessimistic; 1 = a single "
                     "random critic, no min). Clamped to the ensemble size K. "
                     "Ignored in hilserl mode.")
flags.DEFINE_string("quality_filters", None,
                    "Comma-separated filter sections out of "
                    "{safety_veto,jerk_veto,outcome_filter,unsolicited}; "
                    "'' or 'none' disables all filters.")
flags.DEFINE_float("bc_loss_weight", None,
                   "lambda_BC on the weighted imitation loss (S5.6). 0.0 makes "
                   "the quality signal drive ONLY the replay priority and "
                   "nothing else (no imitation term); the counterfactual score "
                   "q_int is still attached to interventions and still filters "
                   "segments. quality/aqi arms only.")
flags.DEFINE_boolean("per_standardize", None,
                     "Divide q_int by its running scale before the replay "
                     "priority so per_eta is dimension-free (raw q_int is a "
                     "tiny Q-advantage swamped by |delta|). quality/aqi only.")
flags.DEFINE_float("per_delta_init", -1.0,
                   "PER optimistic |delta| init in the priority formula "
                   "(quality/aqi; <0 = experiment default).")
flags.DEFINE_float("per_eta", -1.0,
                   "PER weight on max(q_int,0) in the priority (quality/aqi; "
                   "<0 = experiment default).")
flags.DEFINE_float("per_eps", -1.0,
                   "PER epsilon floor in the priority (quality/aqi; <0 = "
                   "experiment default).")
flags.DEFINE_integer("quality_warmup_steps", -1,
                     "Env steps before interventions carry their quality "
                     "replay priority; before it they get baseline priority "
                     "(immature-ensemble warm-up). quality/aqi; <0 = default.")
flags.DEFINE_float("per_scale_decay", -1.0,
                   "EMA decay for the q-standardization scale (quality/aqi; "
                   "0 = cumulative mean; <0 = experiment default).")
flags.DEFINE_integer("per_refresh_period", -1,
                     "Recompute all stored priorities every N inserts "
                     "(quality/aqi; 0 = never; <0 = experiment default).")
flags.DEFINE_float("per_is_beta", -1.0,
                   "Importance-sampling exponent for prioritised replay "
                   "(quality/aqi; 0 = off; <0 = experiment default).")
flags.DEFINE_enum("quality_score_mode", None,
                  ["counterfactual", "behavioral", "hindsight", "progress",
                   "outcome", "uniform"],
                  "Graded intervention signal: 'counterfactual' = A_hat "
                  "critic Q-gap (default); 'behavioral' = critic-free "
                  "z(d_value)-z(t_int)-z(f_peak) composite; 'hindsight' = "
                  "n-step advantage from onset to handback+K on the visited "
                  "states' V(s); 'progress' = insertion-depth change from "
                  "onset to handback+K; 'outcome' = +1/-1 episode success; "
                  "'uniform' = no grading (control). quality/aqi only.")
flags.DEFINE_integer("quality_post_release_steps", None,
                     "K: steps after the handback included in the hindsight/"
                     "progress window (default 10). Logged in every logging "
                     "mode; drives q only with the matching score mode.")
flags.DEFINE_float("quality_hindsight_discount", None,
                   "Discount of the hindsight advantage (default: the "
                   "experiment's TD discount, config.discount).")
flags.DEFINE_boolean("quality_consistency_weight", None,
                     "Multiply the BC weight by the operator-consistency "
                     "factor 1/(1+(sigma_beta/sigma_ref)^2) (SiLRI-style). "
                     "Inert unless bc_loss_weight > 0. quality/aqi only.")
flags.DEFINE_integer("quality_consistency_k", None,
                     "Earlier takeover segments (one neighbour each) behind "
                     "sigma_beta (default 5). quality/aqi only.")
flags.DEFINE_float("quality_consistency_sigma_ref", None,
                   "Fixed sigma_ref of the consistency factor (default: the "
                   "running mean of sigma_beta). quality/aqi only.")
flags.DEFINE_boolean("quality_cf_diag", None,
                     "Log diagnostics of the counterfactual score (every "
                     "critic's Q at a_H / a_robot / the policy mode / policy "
                     "samples per takeover step) and the synthetic operator's "
                     "simulator ground truth per step. Logging only; no "
                     "effect on training. Every logging mode.")
flags.DEFINE_string("quality_behavioral_terms", None,
                    "Comma-separated terms of the behavioral score out of "
                    "{d_value,t_int,f_peak} (default all). E.g. "
                    "'d_value,f_peak' drops takeover length. quality/aqi with "
                    "--quality_score_mode=behavioral only.")
flags.DEFINE_integer("e2hil_K", None,
                     "E2HiL: Monte-Carlo actions per state for the entropy-"
                     "influence estimate c(s) (paper default 8). e2hil only.")
flags.DEFINE_float("e2hil_q_low", None,
                   "E2HiL: lower batch percentile (0-1) of |c(s)|; states "
                   "below it are dropped as 'noisy' (default 0.05). e2hil only.")
flags.DEFINE_float("e2hil_q_high", None,
                   "E2HiL: upper batch percentile (0-1) of |c(s)|; states "
                   "above it are dropped as 'shortcut' (default 0.90). "
                   "e2hil only.")
flags.DEFINE_enum("e2hil_q_agg", None, ["min", "mean"],
                  "E2HiL: critic reduction in the actor objective and in "
                  "c(s). 'min' = official E2HiL code (default); 'mean' = "
                  "this repo's hilserl actor, isolating the mask. e2hil only.")
flags.DEFINE_enum("peg_shape", None, ["cylinder", "hex", "cuboid"],
                  "Sim only: insertion geometry -- 'cylinder' (easy, round, no "
                  "yaw) or 'hex' (exact tight task). Defaults to the experiment "
                  "config's peg_shape.")
flags.DEFINE_bool("render", False,
                  "Actor: open a live window showing the scene with an AQI "
                  "status banner (autonomous / REQUESTS INTERVENTION / HUMAN / "
                  "safety) and ring the terminal bell when a request first fires.")
flags.DEFINE_enum("query_behavior", "continue", ["continue", "pause"],
                  "Actor behaviour when the active gate requests a takeover: "
                  "'continue' keeps executing the policy (the request is only a "
                  "cue); 'pause' holds position and waits for the operator, "
                  "resuming automatically once the trigger drops below tau_A.")

devices = jax.local_devices()
num_devices = len(devices)
sharding = DeviceSharding(devices)


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


def print_yellow(x):
    return print("\033[93m {}\033[00m".format(x))


def _shapes_match(a_state, b_state):
    """True iff two train states have the same pytree structure AND identical
    per-leaf shapes. Used to decide whether an on-disk checkpoint can be loaded
    into the freshly built agent: a changed critic ensemble size K (or any other
    architecture change) alters leaf shapes, so restoring would corrupt the run.
    Uses np.shape so python/0-d scalars compare equal to 0-d arrays."""
    a_leaves, a_def = jax.tree_util.tree_flatten(a_state)
    b_leaves, b_def = jax.tree_util.tree_flatten(b_state)
    if a_def != b_def:
        return False
    return all(np.shape(x) == np.shape(y) for x, y in zip(a_leaves, b_leaves))


def print_red(x):
    return print("\033[91m {}\033[00m".format(x))


##############################################################################


def _wrap_synthetic_operator(env, agent, aqi_cfg):
    """Wrap the actor env with the document's synthetic operator (Part
    A): the SpaceMouse-equivalent SyntheticOperatorWrapper. Oracle
    resolution: an env-provided `phase_oracle` attribute (sim, doc S4
    option A) wins; else --oracle_checkpoint loads a frozen policy as
    pi_or (option B) and the decision block degrades it with
    wrong-intent segments."""
    from serl_launcher.aqi.operator_gate import (MonitoringSignal,
                                                 SyntheticOperatorWrapper)
    from serl_launcher.aqi.operator_model import (PhaseOracle,
                                                  PolicyOracle,
                                                  SyntheticOperator,
                                                  make_operator_theta,
                                                  theta_from_json)
    rng = np.random.default_rng(FLAGS.synthetic_seed)
    if FLAGS.synthetic_theta:
        theta = theta_from_json(FLAGS.synthetic_theta)
        if FLAGS.operator_id:
            theta.operator_id = FLAGS.operator_id
        print_green(f"loaded exact operator theta from {FLAGS.synthetic_theta} "
                    f"(tier={theta.tier}, id={theta.operator_id})")
    else:
        theta = make_operator_theta(FLAGS.synthetic_operator, rng=rng,
                                    operator_id=FLAGS.operator_id)
    # Execution oracle (the competent core the operator degrades). Preference:
    #   1. env.synthetic_oracle -- the FULL geometric operator the sim attaches
    #      (execution SKILL affects the correction: motor noise, per-approach
    #      bias, recovery style); uses the env's action scale + sign fix;
    #   2. env.phase_oracle -- a phase oracle exposed directly (e.g. toy env);
    #   3. --oracle_checkpoint -- a frozen policy as pi_or (doc S4 option B):
    #      correct action frame but skill-INDEPENDENT corrections.
    exec_oracle = getattr(env, "synthetic_oracle", None) \
        or getattr(env, "phase_oracle", None) \
        or getattr(getattr(env, "unwrapped", env), "phase_oracle", None)
    # --training_oracle picks which oracle provides the intervention ACTION:
    #   geometric -> the env's attached geometric oracle (skill-degraded);
    #   policy    -> the frozen --oracle_checkpoint policy, even when the sim
    #                also attaches a geometric one (the geometric
    #                env.synthetic_monitor still times the takeovers below);
    #   auto      -> env oracle when present, else the policy checkpoint.
    if FLAGS.training_oracle == "geometric" and exec_oracle is None:
        raise ValueError("--training_oracle=geometric but the env exposes no "
                         "geometric oracle")
    use_policy_oracle = (FLAGS.training_oracle == "policy"
                         or (FLAGS.training_oracle == "auto"
                             and exec_oracle is None))
    action_conditioner = None
    if not use_policy_oracle:
        oracle = exec_oracle
        op_action_scale = float(getattr(env, "operator_action_scale", 1.0))
        action_conditioner = getattr(env, "operator_action_conditioner", None)
        kind = "geometric"
    else:
        if not FLAGS.oracle_checkpoint:
            raise ValueError(
                "policy training oracle needs --oracle_checkpoint (a frozen "
                "policy); none was given")
        ckpt = checkpoints.restore_checkpoint(
            os.path.abspath(FLAGS.oracle_checkpoint), agent.state)
        oracle_agent = agent.replace(state=ckpt)
        okey = jax.random.PRNGKey(FLAGS.synthetic_seed)

        def policy_fn(obs):
            a = oracle_agent.sample_actions(
                observations=jax.device_put(obs), argmax=True, seed=okey)
            return np.asarray(jax.device_get(a))
        mm_per_unit = FLAGS.synthetic_env_mm_per_unit
        oracle = PolicyOracle(policy_fn,
                              action_dim=env.action_space.shape[0],
                              action_scale=mm_per_unit)
        op_action_scale = 1.0 / mm_per_unit
        kind = "policy"
    # Prefer the environment's own calibrated risk signal (sim attaches a
    # geometric MonitoringSignal via env.synthetic_monitor -- corridor /
    # stagnation / force in real sim units); otherwise fall back to the generic
    # monitor (geometric only when the execution oracle is itself a PhaseOracle,
    # e.g. the toy env; else timeout-only).
    monitor = getattr(env, "synthetic_monitor", None)
    if monitor is None:
        monitor = MonitoringSignal(
            oracle if isinstance(oracle, PhaseOracle) else None,
            budget_steps=config.max_traj_length)
    op = SyntheticOperator(
        oracle, theta,
        rng=np.random.default_rng(FLAGS.synthetic_seed + 17),
        action_scale=op_action_scale,
        rotation_control=FLAGS.synthetic_rotation_control)

    if FLAGS.synthetic_intervention_mode == "legacy":
        intervention_mode = ("always_latch"
                             if FLAGS.synthetic_latch_until_success
                             else "signal")
    else:
        intervention_mode = FLAGS.synthetic_intervention_mode

    print_green(f"synthetic operator '{FLAGS.synthetic_operator}' "
                f"(exec s={theta.s_exec:.2f}, mon s={theta.s_mon:.2f}, "
                f"{kind} oracle, intervention={intervention_mode}"
                f"{', rotation control' if op.rotation_control else ''}) "
                f"wrapping the env; theta -> meta.json")
    return SyntheticOperatorWrapper(
        env, op, monitor,
        rng=np.random.default_rng(FLAGS.synthetic_seed + 31),
        respond_to_request=aqi_cfg.active_on,
        action_conditioner=action_conditioner,
        intervention_mode=intervention_mode,
        latch_until_success=FLAGS.synthetic_latch_until_success,
        log_ground_truth=aqi_cfg.quality_cf_diag,
        log_takeover_channels=FLAGS.log_takeover_channels), theta


def actor(agent, data_store, intvn_data_store, env, sampling_rng, aqi_cfg):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    Identical to train_rlpd.py except for the AQIRuntime hooks; with
    --aqi_mode=hilserl every hook is a no-op.
    """
    if FLAGS.eval_checkpoint_step:
        success_counter = 0
        time_list = []

        ckpt = checkpoints.restore_checkpoint(
            os.path.abspath(FLAGS.checkpoint_path),
            agent.state,
            step=FLAGS.eval_checkpoint_step,
        )
        agent = agent.replace(state=ckpt)

        for episode in range(FLAGS.eval_n_trajs):
            obs, _ = env.reset()
            done = False
            start_time = time.time()
            while not done:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    argmax=False,
                    seed=key
                )
                actions = np.asarray(jax.device_get(actions))

                next_obs, reward, done, truncated, info = env.step(actions)
                obs = next_obs

                if done:
                    if reward:
                        dt = time.time() - start_time
                        time_list.append(dt)
                        print(dt)

                    success_counter += reward
                    print(reward)
                    print(f"{success_counter}/{episode + 1}")

        print(f"success rate: {success_counter / FLAGS.eval_n_trajs}")
        print(f"average time: {np.mean(time_list)}")
        return  # after done eval, return and exit

    _buf_pkls = (
        natsorted(glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")))
        if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path)
        else []
    )
    start_step = (
        int(os.path.basename(_buf_pkls[-1])[12:-4]) + 1
        if _buf_pkls else 0
    )

    datastore_dict = {
        "actor_env": data_store,
        "actor_env_intvn": intvn_data_store,
    }

    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_stores=datastore_dict,
        wait_for_server=True,
        timeout_ms=3000,
    )

    # Function to update the agent with new params
    def update_params(params):
        nonlocal agent
        agent = agent.replace(state=agent.state.replace(params=params))

    client.recv_network_callback(update_params)

    # ---- synthetic operator (methodology doc Part A): wrap the env at
    # the SpaceMouse integration point before the runtime sees it -------
    synthetic_theta = None
    if FLAGS.synthetic_operator:
        env, synthetic_theta = _wrap_synthetic_operator(env, agent, aqi_cfg)

    # ---- AQI runtime (S5): gate, force monitor, OOD, P_fail, quality ------
    run_name = FLAGS.run_name or "{}_{}_{}_s{}_{}".format(
        FLAGS.exp_name, aqi_cfg.mode, FLAGS.operator_id, FLAGS.seed,
        time.strftime("%Y%m%d-%H%M%S"))
    aqi = AQIRuntime(
        aqi_cfg, env,
        operator_id=FLAGS.operator_id,
        run_name=run_name,
        max_episode_length=config.max_traj_length,
        config_dict_for_log={**vars(aqi_cfg),
                             "arm": FLAGS.arm or aqi_cfg.mode,
                             **({"synthetic_operator":
                                 synthetic_theta.as_dict()}
                                if synthetic_theta else {})},
    )
    if aqi.enabled:
        print_green(f"AQI mode={aqi_cfg.mode}; logs -> {aqi.logger.dir}")
        # Seed D_success for the OOD trigger with demonstration states
        # (S5.4). Pass --demo_path to the actor as well to enable this.
        if FLAGS.demo_path:
            for path in FLAGS.demo_path:
                with open(path, "rb") as f:
                    aqi.seed_ood_from_demo_transitions(pkl.load(f))
            if aqi.ood is not None:
                print_green(f"OOD reference seeded with {len(aqi.ood)} demo states")

    transitions = []
    demo_transitions = []

    # Reset randomization for WITHIN-operator variance: seed the training env's
    # start-pose RNG (and the random-exploration action sampler) from the run
    # seed so replicate runs of the SAME operator see INDEPENDENT reset
    # sequences. Without this the env RNG is fixed at RandomState(0) and every
    # run replays identical resets, deflating within-operator variance (and
    # thus inflating the operator-dependency ICC). The eval ruler is
    # unaffected -- it applies its own fixed, hash-locked pose grid.
    env_seed = FLAGS.env_seed if FLAGS.env_seed >= 0 else FLAGS.seed
    try:
        env.action_space.seed(int(env_seed))
    except Exception:
        pass
    obs, _ = env.reset(seed=int(env_seed))
    done = False
    aqi.begin_episode()

    # training loop
    timer = Timer()
    running_return = 0.0
    already_intervened = False
    intervention_count = 0
    intervention_steps = 0
    request_cue_on = False
    hard_cue_on = False
    takeover_channels, handbacks = {}, {}     # --log_takeover_channels

    # ---- convergence-based early stop (actor) -----------------------------
    # A rolling window of recent episode outcomes lets the actor stop as soon as
    # the policy is reliably solving the task (optionally WITHOUT operator help),
    # instead of always spending the full --max_steps budget. --max_steps stays
    # the hard cap for cells that never converge. The stopping step is recorded
    # so 'steps-to-convergence' becomes an operator-dependency metric of its own.
    converge_on = FLAGS.converge_success > 0.0  # on iff --converge_success set
    recent_outcomes = []          # 1.0 = counted success, 0.0 otherwise
    episodes_total = 0
    converged = False

    def _parse_eval_poses(spec):
        poses = []
        for chunk in str(spec or "").split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            parts = [p.strip() for p in chunk.split(",")]
            if len(parts) != 2:
                raise ValueError(
                    f"invalid --converge_eval_poses entry '{chunk}'; expected 'dx,dyaw'")
            poses.append((float(parts[0]), float(parts[1])))
        if not poses:
            raise ValueError("--converge_eval_poses must include at least one pose")
        return poses

    converge_eval_on = FLAGS.converge_eval_threshold > 0.0
    converge_eval_poses = _parse_eval_poses(FLAGS.converge_eval_poses) if converge_eval_on else []

    # ---- convergence by the fixed-pose mini-eval (--converge_mode eval) -----
    # The rolling rule counts only training episodes the operator never
    # touched, so an operator who keeps stepping in delays it although the
    # policy may already succeed alone. In eval mode the mini-eval decides.
    eval_mode = FLAGS.converge_mode == "eval"
    if eval_mode and not converge_eval_on:
        raise ValueError("--converge_mode eval needs --converge_eval_threshold > 0")
    eval_passes, eval_history, last_eval_step = 0, [], None
    seen_autonomous = False

    def _run_convergence_mini_eval(eval_env, rng_key):
        """Small fixed-pose autonomous check that mirrors the ruler's reset semantics."""
        if not hasattr(eval_env, "set_eval_reset_pose"):
            raise RuntimeError(
                "convergence mini-eval requested but env lacks set_eval_reset_pose")
        successes = 0
        try:
            for dx_m, dyaw in converge_eval_poses:
                eval_env.set_eval_reset_pose(dx_m, dyaw)
                obs_eval, _ = eval_env.reset()
                done_eval = False
                steps_eval = 0
                while not done_eval and steps_eval < FLAGS.converge_eval_max_steps:
                    rng_key, key_eval = jax.random.split(rng_key)
                    act_eval = agent.sample_actions(
                        observations=jax.device_put(obs_eval),
                        seed=key_eval,
                        argmax=True,
                    )
                    act_eval = np.asarray(jax.device_get(act_eval))
                    obs_eval, rew_eval, done_eval, trunc_eval, _ = eval_env.step(act_eval)
                    done_eval = bool(done_eval or trunc_eval)
                    steps_eval += 1
                successes += int(bool(rew_eval > 0))
        finally:
            eval_env.set_eval_reset_pose(None, None)
        return successes / float(len(converge_eval_poses))

    def _mini_eval_unpaced(eval_env, rng_key):
        """The mini-eval without real-time pacing (eval mode only)."""
        base = getattr(eval_env, "unwrapped", eval_env)
        was = getattr(base, "realtime", None)
        if was is not None:
            base.realtime = False
        try:
            return _run_convergence_mini_eval(eval_env, rng_key)
        finally:
            if was is not None:
                base.realtime = was

    def _record_convergence(step, payload):
        """Certified checkpoint + CONVERGED.json, as in the rolling rule."""
        if not FLAGS.checkpoint_path:
            return
        try:
            _agent_state = jax.block_until_ready(agent.state)
            checkpoints.save_checkpoint(
                os.path.join(os.path.abspath(FLAGS.checkpoint_path), "certified"),
                _agent_state, step=int(step), keep=1, overwrite=True)
            payload["certified_checkpoint_step"] = int(step)
            print_green(f">>> certified checkpoint saved at step {int(step)} -> certified/")
        except Exception as _e:
            print_yellow(f">>> could not save certified checkpoint: {_e}")
        try:
            with open(os.path.join(os.path.abspath(FLAGS.checkpoint_path),
                                   "CONVERGED.json"), "w") as _cf:
                json.dump(payload, _cf, indent=2)
        except OSError:
            pass

    # ---- fixed wall-clock training budget (actor) -------------------------
    # A hard time budget guarantees the cell terminates even if it never
    # converges and the step cap is huge. The actor stops at whichever comes
    # first: convergence, the step cap, or this time budget.
    t_start = time.time()
    max_seconds = FLAGS.max_minutes * 60.0 if FLAGS.max_minutes > 0 else 0.0
    timed_out = False

    # optional live operator window with the AQI request/intervention banner
    render_win = bool(FLAGS.render)
    win_name = "aqi_actor"
    if render_win:
        import cv2  # local import: only needed for the optional live view
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    # per-key layout of the flattened 'state' vector, for the on-screen readout
    state_slices = resolve_state_slices(env, config) if render_win else []
    # 'pause' only makes sense while the ACTIVE gate can request a takeover;
    # in modes without the gate (hilserl / logged / quality) it is forced to
    # 'continue' so the actor's query_behavior choice cannot change the run.
    query_behavior = FLAGS.query_behavior if aqi_cfg.active_on else "continue"

    # ---- fixed learner updates per env step (--utd_ratio) --------------------
    utd = None
    if FLAGS.utd_ratio > 0.0 and FLAGS.checkpoint_path:
        utd = UpdateRatio(FLAGS.checkpoint_path, FLAGS.utd_ratio,
                          timeout_s=FLAGS.utd_timeout_s, warn=print_yellow)
        _base_env = getattr(env, "unwrapped", env)
        if hasattr(_base_env, "realtime"):
            _base_env.realtime = False
        print_green(f"fixed update ratio: {FLAGS.utd_ratio:g} learner steps per "
                    f"env step; the env runs without real-time pacing")

    pbar = tqdm.tqdm(range(start_step, config.max_steps), dynamic_ncols=True)
    for step in pbar:
        timer.tick("total")
        if utd is not None:
            with timer.context("utd_wait"):
                utd.actor_wait()

        # fixed wall-clock budget: stop promptly (checked every step) once the
        # time limit is hit, regardless of how many steps remain.
        if max_seconds and (time.time() - t_start) >= max_seconds:
            timed_out = True
            print_yellow(
                f"\n>>> TIME BUDGET reached ({FLAGS.max_minutes:.1f} min) at "
                f"step {step}; stopping the actor.")
            break

        with timer.context("sample_actions"):
            if step < config.random_steps:
                actions = env.action_space.sample()
            else:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    seed=key,
                    argmax=False,
                )
                actions = np.asarray(jax.device_get(actions))

        # ---- AQI: trigger terms + query decision at the current state ----
        with timer.context("aqi_trigger"):
            robot_action = np.asarray(actions).copy()
            gate_rec = aqi.observe_step(agent, obs, robot_action,
                                        intervening=already_intervened)
            hard = bool(gate_rec.get("hard_violation", False))
            # edge-triggered cues: '\a' rings the terminal bell on the step a
            # request first fires (it will not spam during the hold window).
            if hard and not hard_cue_on:
                print_red("\a>>> FORCE/TORQUE LIMIT - please take over (safety)")
            elif gate_rec["request"] and not request_cue_on and not hard:
                print_yellow("\a>>> ROBOT REQUESTS INTERVENTION "
                             f"(A={gate_rec.get('a_score', 0.0):.2f}, "
                             f"reason={gate_rec.get('reason', '')})")
            hard_cue_on = hard
            request_cue_on = bool(gate_rec["request"])

        # active/aqi with a synthetic operator: hand the robot's request to the
        # operator so it RESPONDS to the active-learning query (skill-modulated
        # reaction / miss / false alarm) instead of acting on its own monitor.
        # In baseline/quality (active gate off) this is a no-op; the operator
        # keeps using its own task monitor.
        if synthetic_theta is not None:
            env.notify_request(bool(gate_rec["request"]),
                               gate_rec.get("a_score", 0.0))

        # ---- query behaviour: 'pause' holds position and waits for a takeover
        # while a request is active (it resumes when the trigger clears);
        # 'continue' keeps executing the policy action. Either way the
        # SpaceMouse wrapper overrides the instant the operator moves it.
        paused = (query_behavior == "pause"
                  and (bool(gate_rec["request"]) or hard)
                  and not already_intervened)
        exec_action = np.zeros_like(actions) if paused else actions

        # Step environment
        with timer.context("step_env"):

            next_obs, reward, done, truncated, info = env.step(exec_action)
            if aqi_cfg.quality_cf_diag:     # logging only (diagnostics)
                aqi.note_ground_truth(info.get("synthetic_operator"))
            if FLAGS.log_takeover_channels:  # logging only (diagnostics)
                _syn = info.get("synthetic_operator") or {}
                if _syn.get("takeover_channel"):
                    _c = _syn["takeover_channel"]
                    takeover_channels[_c] = takeover_channels.get(_c, 0) + 1
                if _syn.get("handback_kind"):
                    _k = _syn["handback_kind"]
                    handbacks[_k] = handbacks.get(_k, 0) + 1
            if "left" in info:
                info.pop("left")
            if "right" in info:
                info.pop("right")

            # override the action with the intervention action
            if "intervene_action" in info:
                actions = info.pop("intervene_action")
                intervention_steps += 1
                if not already_intervened:
                    intervention_count += 1
                already_intervened = True
            else:
                actions = exec_action   # record what actually ran (hold if paused)
                already_intervened = False

            running_return += reward
            transition = dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=reward,
                masks=1.0 - done,
                dones=done,
            )
            if 'grasp_penalty' in info:
                transition['grasp_penalty'] = info['grasp_penalty']
            if aqi_cfg.e2hil_on:
                # source label for E2HiL's per-source diagnostics
                # (1 = online policy step, 2 = human/synthetic takeover)
                transition["labels"] = 2 if already_intervened else 1
            if aqi_cfg.quality_on:
                # every transition must carry the AQI fields because the
                # learner buffers allocate them; autonomous steps are zeros.
                transition["q_int"] = np.float32(0.0)
                transition["quality_weight"] = np.float32(0.0)
                transition["intervention_mask"] = np.float32(0.0)
            data_store.insert(transition)
            transitions.append(copy.deepcopy(transition))

            # ---- intervention data routing --------------------------------
            # quality modes (S5.6): buffer segments until episode end, score
            # them (counterfactual advantage + vetoes/filters) and only then
            # ship the survivors to the intervention buffer.
            # other modes: baseline behaviour, ship every step immediately.
            aqi.record_transition(
                obs, transition, robot_action,
                intervening=already_intervened,
                requested=bool(gate_rec["request"]),
                hard_violation=bool(gate_rec.get("hard_violation", False)),
            )
            if already_intervened and not aqi_cfg.quality_on:
                intvn_data_store.insert(transition)
                demo_transitions.append(copy.deepcopy(transition))

            # ---- live operator window (optional) --------------------------
            if render_win:
                frame = env.render()
                if frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    h_fr, w = frame.shape[:2]

                    is_synthetic = synthetic_theta is not None
                    syn = info.get("synthetic_operator", {})
                    syn_over = bool(syn.get("taking_over", False))
                    syn_z = float(syn.get("z", 0.0))
                    syn_proven = bool(syn.get("policy_proven", False))

                    a_score = float(gate_rec.get("a_score", 0.0))
                    sigma_q = float(gate_rec.get("sigma_q", 0.0))
                    tau = float(aqi_cfg.tau_A)
                    trig_above = hard or (aqi_cfg.active_on and a_score > tau)
                    requested = bool(gate_rec["request"])

                    # --- line 1: status banner ---
                    if hard:
                        c1, banner = (0, 0, 255), "SAFETY STOP - TAKE OVER"
                    elif requested and query_behavior == "pause" and not already_intervened:
                        c1, banner = (0, 140, 255), "PAUSED - WAITING FOR TAKEOVER"
                    elif is_synthetic and syn_over:
                        latch_tag = " [signal-only]" if syn_proven else " [latched]"
                        c1 = (0, 140, 255)   # orange in BGR
                        banner = (f"SYNTHETIC ({FLAGS.synthetic_operator.upper()})"
                                  f" IN CONTROL{latch_tag}")
                    elif requested and not already_intervened:
                        c1, banner = (0, 215, 255), "REQUESTS INTERVENTION"
                    elif already_intervened:
                        c1, banner = (0, 200, 0), "HUMAN IN CONTROL"
                    else:
                        c1, banner = (180, 180, 180), "AUTONOMOUS"

                    # --- line 2: mode-specific metrics ---
                    if not aqi_cfg.active_on and not aqi_cfg.quality_on:
                        # baseline/logged: no AQI concepts
                        if is_synthetic:
                            zc = (0, 0, 200) if syn_z > 0.6 else (0, 180, 255) if syn_z > 0.3 else (0, 200, 0)
                            line2 = (f"z_t={syn_z:.2f}"
                                     f"  {'PROVEN - signal-only' if syn_proven else 'phase-1 latch'}"
                                     f"  operator={FLAGS.synthetic_operator}")
                        else:
                            zc, line2 = (180, 180, 180), f"step={step}  return={running_return:.0f}"
                    elif aqi_cfg.active_on and not aqi_cfg.quality_on:
                        # active only
                        zc = (0, 0, 220) if trig_above else (0, 200, 0)
                        line2 = (f"A(z)={a_score:+.2f}  tau={tau:.2f}"
                                 f"  [{'ABOVE' if trig_above else 'below'}]")
                        if is_synthetic:
                            line2 += f"  z_t={syn_z:.2f}"
                    elif aqi_cfg.quality_on and not aqi_cfg.active_on:
                        # quality only
                        zc = (0, 180, 255) if sigma_q > 0.3 else (0, 200, 0)
                        line2 = f"sigma_Q={sigma_q:.2f}"
                        if is_synthetic:
                            line2 += f"  z_t={syn_z:.2f}"
                    else:
                        # full AQI
                        zc = (0, 0, 220) if trig_above else (0, 200, 0)
                        line2 = (f"A(z)={a_score:+.2f}  tau={tau:.2f}"
                                 f"  sigma_Q={sigma_q:.2f}"
                                 f"  [{'ABOVE' if trig_above else 'below'}]")
                        if is_synthetic:
                            line2 += f"  z_t={syn_z:.2f}"

                    # --- line 3: guidance ---
                    if hard:
                        gc, guide = (0, 0, 255), "force/torque limit - take over now"
                    elif is_synthetic and syn_over:
                        gc = (0, 140, 255)
                        guide = ("synthetic operator: signal-only mode"
                                 if syn_proven else
                                 "synthetic operator: latched until success")
                    elif aqi_cfg.active_on and trig_above and not already_intervened:
                        gc, guide = (0, 140, 255), "above threshold - intervention needed"
                    elif aqi_cfg.active_on and already_intervened:
                        gc, guide = (0, 220, 0), "below threshold - safe to RELEASE"
                    elif already_intervened:
                        gc, guide = (0, 200, 0), "operator in control"
                    else:
                        gc, guide = (0, 220, 0), "autonomous"

                    cv2.rectangle(frame, (0, 0), (w, 98), (0, 0, 0), -1)
                    cv2.putText(frame, banner, (10, 26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, c1, 2)
                    cv2.putText(frame, line2,
                                (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.6, zc, 2)
                    cv2.putText(frame, guide, (10, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, gc, 1)
                    syn_tag = (f"  synthetic={FLAGS.synthetic_operator}"
                               if is_synthetic else "")
                    cv2.putText(frame,
                                f"mode={aqi_cfg.mode}{syn_tag}"
                                f"  step={step}  return={running_return:.0f}",
                                (10, h_fr - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (200, 200, 200), 1)
                    frame = compose_obs_view(cv2, frame, next_obs,
                                             config.image_keys, state_slices)
                    cv2.imshow(win_name, frame)
                    cv2.waitKey(1)

            obs = next_obs
            if done or truncated:
                episode_success = bool(running_return > 0)
                scored = aqi.end_episode(agent, episode_success,
                                         running_return)
                if aqi_cfg.quality_on:
                    for tr in scored:
                        intvn_data_store.insert(tr)
                        demo_transitions.append(copy.deepcopy(tr))
                    if scored:
                        pbar.write(
                            "AQI: {} intervention transitions kept "
                            "(q_int mean {:.3f}, w mean {:.3f})".format(
                                len(scored),
                                float(np.mean([t["q_int"] for t in scored])),
                                float(np.mean([t["quality_weight"]
                                               for t in scored])),
                            ))
                info["episode"]["intervention_count"] = intervention_count
                info["episode"]["intervention_steps"] = intervention_steps
                # Per-episode takeover log (mode-independent, survives wandb-off)
                # so the intervention-frequency curves are exact for any run.
                if FLAGS.checkpoint_path:
                    try:
                        with open(os.path.join(
                                os.path.abspath(FLAGS.checkpoint_path),
                                "episodes.jsonl"), "a") as _ef:
                            _rec = {
                                "step": int(step),
                                "intervention_count": int(intervention_count),
                                "intervention_steps": int(intervention_steps),
                                "success": bool(episode_success),
                                "length": int(info["episode"].get("l", 0)),
                            }
                            if FLAGS.log_takeover_channels:
                                _rec["takeover_channels"] = dict(takeover_channels)
                                _rec["handbacks"] = dict(handbacks)
                            _ef.write(json.dumps(_rec) + "\n")
                    except Exception:
                        pass
                # rolling-window convergence check (actor early stop). An
                # episode "counts" toward convergence when it succeeded and --
                # if --converge_require_autonomous -- the operator never took
                # over, i.e. the POLICY solved it unaided.
                if converge_on and not eval_mode:
                    episodes_total += 1
                    counted = episode_success and (
                        not FLAGS.converge_require_autonomous
                        or intervention_count == 0)
                    recent_outcomes.append(1.0 if counted else 0.0)
                    if len(recent_outcomes) > FLAGS.converge_window:
                        recent_outcomes.pop(0)
                    roll = sum(recent_outcomes) / max(len(recent_outcomes), 1)
                    if (len(recent_outcomes) >= FLAGS.converge_window
                            and step >= FLAGS.converge_min_steps
                            and roll >= FLAGS.converge_success):
                        ruler_sr = None
                        if converge_eval_on:
                            base_env = getattr(env, "env", env)
                            ruler_sr = _run_convergence_mini_eval(base_env, sampling_rng)
                            if ruler_sr < FLAGS.converge_eval_threshold:
                                print_yellow(
                                    f"\n>>> rolling convergence reached at step {step} "
                                    f"(SR={roll:.2f}) but fixed-pose mini-eval SR "
                                    f"{ruler_sr:.2f} < {FLAGS.converge_eval_threshold:.2f}; "
                                    f"continuing training.")
                        if (not converge_eval_on) or (ruler_sr >= FLAGS.converge_eval_threshold):
                            converged = True
                            _kind = ("autonomous "
                                     if FLAGS.converge_require_autonomous else "")
                            gate_txt = (""
                                        if ruler_sr is None else
                                        f", mini-eval SR {ruler_sr:.2f} >= "
                                        f"{FLAGS.converge_eval_threshold:.2f}")
                            print_green(
                                f"\n>>> CONVERGED at step {step}: rolling {_kind}"
                                f"success {roll:.2f} >= {FLAGS.converge_success:.2f} "
                                f"over {FLAGS.converge_window} episodes "
                                f"({episodes_total} total{gate_txt}) -> stopping the actor.")
                            # Snapshot the EXACT actor weights that just cleared
                            # convergence (and the ruler-aligned mini-eval) into
                            # <checkpoint_path>/certified. On an oscillating policy
                            # the learner's post-STOP checkpoint can be a different
                            # (worse) phase, so eval_ruler prefers this one.
                            certified_step = None
                            if FLAGS.checkpoint_path:
                                try:
                                    _agent_state = jax.block_until_ready(agent.state)
                                    checkpoints.save_checkpoint(
                                        os.path.join(
                                            os.path.abspath(FLAGS.checkpoint_path),
                                            "certified"),
                                        _agent_state, step=int(step), keep=1,
                                        overwrite=True)
                                    certified_step = int(step)
                                    print_green(
                                        f">>> certified checkpoint saved at step "
                                        f"{certified_step} -> certified/")
                                except Exception as _e:
                                    print_yellow(
                                        f">>> could not save certified checkpoint: {_e}")
                            if FLAGS.checkpoint_path:
                                try:
                                    with open(os.path.join(
                                            os.path.abspath(FLAGS.checkpoint_path),
                                            "CONVERGED.json"), "w") as _cf:
                                        payload = dict(
                                            step=int(step),
                                            episodes=int(episodes_total),
                                            rolling_success=float(roll),
                                            window=int(FLAGS.converge_window),
                                            threshold=float(FLAGS.converge_success),
                                            autonomous=bool(
                                                FLAGS.converge_require_autonomous),
                                        )
                                        if certified_step is not None:
                                            payload["certified_checkpoint_step"] = certified_step
                                        if ruler_sr is not None:
                                            payload["mini_eval_sr"] = float(ruler_sr)
                                            payload["mini_eval_threshold"] = float(
                                                FLAGS.converge_eval_threshold)
                                            payload["mini_eval_poses"] = [
                                                [float(dx), float(dyaw)]
                                                for dx, dyaw in converge_eval_poses
                                            ]
                                        json.dump(payload, _cf, indent=2)
                                except OSError:
                                    pass
                if eval_mode:
                    episodes_total += 1
                    if episode_success and intervention_count == 0:
                        seen_autonomous = True
                    if (seen_autonomous and step >= FLAGS.converge_min_steps
                            and (last_eval_step is None
                                 or step - last_eval_step >= FLAGS.converge_eval_every)):
                        last_eval_step = step
                        ruler_sr = _mini_eval_unpaced(getattr(env, "env", env), sampling_rng)
                        eval_history.append([int(step), float(ruler_sr)])
                        eval_passes = (eval_passes + 1
                                       if ruler_sr >= FLAGS.converge_eval_threshold else 0)
                        if FLAGS.checkpoint_path:
                            try:
                                with open(os.path.join(os.path.abspath(FLAGS.checkpoint_path),
                                                       "convergence_evals.jsonl"), "a") as _cv:
                                    _cv.write(json.dumps({"step": int(step), "sr": float(ruler_sr),
                                                          "passes": int(eval_passes)}) + "\n")
                            except OSError:
                                pass
                        pbar.write(f">>> mini-eval at step {step}: SR {ruler_sr:.2f} "
                                   f"(pass {eval_passes}/{FLAGS.converge_eval_consecutive})")
                        if eval_passes >= FLAGS.converge_eval_consecutive:
                            converged = True
                            print_green(
                                f"\n>>> CONVERGED at step {step}: fixed-pose mini-eval SR >= "
                                f"{FLAGS.converge_eval_threshold:.2f} in "
                                f"{eval_passes} evaluations in a row "
                                f"({episodes_total} episodes) -> stopping the actor.")
                            _record_convergence(step, dict(
                                step=int(step), episodes=int(episodes_total),
                                mode="eval",
                                mini_eval_sr=float(ruler_sr),
                                mini_eval_threshold=float(FLAGS.converge_eval_threshold),
                                consecutive=int(FLAGS.converge_eval_consecutive),
                                every=int(FLAGS.converge_eval_every),
                                mini_eval_history=eval_history,
                                mini_eval_poses=[[float(dx), float(dyaw)]
                                                 for dx, dyaw in converge_eval_poses]))
                if aqi.enabled:
                    info["episode"]["aqi_requests"] = aqi._ep_requests if hasattr(aqi, "_ep_requests") else 0
                    info["episode"]["aqi_mode"] = aqi_cfg.mode
                stats = {"environment": info}  # send stats to the learner to log
                client.request("send-stats", stats)
                pbar.set_description(f"last return: {running_return}")
                running_return = 0.0
                intervention_count = 0
                intervention_steps = 0
                takeover_channels, handbacks = {}, {}
                already_intervened = False
                request_cue_on = False
                client.update()
                if utd is not None:
                    utd.actor_note_sent(step - start_step + 1)
                obs, _ = env.reset()
                aqi.begin_episode()

        if converged:
            break

        if step > 0 and config.buffer_period > 0 and step % config.buffer_period == 0:
            # dump to pickle file
            buffer_path = os.path.join(FLAGS.checkpoint_path, "buffer")
            demo_buffer_path = os.path.join(FLAGS.checkpoint_path, "demo_buffer")
            if not os.path.exists(buffer_path):
                os.makedirs(buffer_path)
            if not os.path.exists(demo_buffer_path):
                os.makedirs(demo_buffer_path)
            with open(os.path.join(buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                pkl.dump(transitions, f)
                transitions = []
            with open(
                os.path.join(demo_buffer_path, f"transitions_{step}.pkl"), "wb"
            ) as f:
                pkl.dump(demo_transitions, f)
                demo_transitions = []

        timer.tock("total")

        if step % config.log_period == 0:
            stats = {"timer": timer.get_average_times()}
            client.request("send-stats", stats)

    if utd is not None:
        print_green(f"fixed update ratio: the actor waited {utd.waited_s / 60.0:.1f} min "
                    f"for the learner{' (stopped waiting: learner silent)' if utd.disabled else ''}")
    if converge_on and not converged and not timed_out and not eval_mode:
        print_yellow(
            f"\n>>> actor reached the {config.max_steps}-step budget WITHOUT "
            f"converging (rolling success "
            f"{sum(recent_outcomes) / max(len(recent_outcomes), 1):.2f} < "
            f"{FLAGS.converge_success:.2f}). This cell 'did not converge' -- a "
            f"real operator-dependency signal; no CONVERGED.json is written.")
    if eval_mode and not converged and not timed_out:
        print_yellow(
            f"\n>>> actor reached the {config.max_steps}-step budget WITHOUT "
            f"converging: "
            + (f"{len(eval_history)} mini-evals, last SR {eval_history[-1][1]:.2f}"
               if eval_history else "no mini-eval (no autonomous success yet)")
            + f" (needs >= {FLAGS.converge_eval_threshold:.2f} in "
              f"{FLAGS.converge_eval_consecutive} evaluations in a row); no "
              f"CONVERGED.json is written.")
    if render_win:
        cv2.destroyAllWindows()
    aqi.close()

    # self-describing run provenance next to the checkpoints (best-effort).
    # The campaign runner overwrites this with a fuller version; for a
    # standalone actor run this is the record of what/how it was trained.
    if FLAGS.checkpoint_path and not FLAGS.eval_checkpoint_step:
        try:
            import provenance as _prov
            op = None
            if synthetic_theta is not None:
                td = synthetic_theta.as_dict()
                op = {"operator_id": td.get("operator_id", FLAGS.operator_id),
                      "tier": td.get("tier"), "s_exec": td.get("s_exec"),
                      "s_mon": td.get("s_mon"),
                      **{k: td.get(k) for k in ("sigma0", "v_max", "sigma_perc",
                                                "delta_in", "p_miss", "rho_fa",
                                                "l_react")}}
            elif FLAGS.operator_id:
                op = {"operator_id": FLAGS.operator_id, "human": True}
            term = {"cap": config.max_steps, "converged": bool(converged),
                    "reason": ("converged" if converged else
                               "time_budget" if timed_out else "step_cap"),
                    "wall_minutes": round((time.time() - t_start) / 60.0, 2)}
            if FLAGS.max_minutes > 0:
                term["time_budget_min"] = FLAGS.max_minutes
            if converged:
                term["step"] = int(step)
            _prov.document(
                os.path.abspath(FLAGS.checkpoint_path), "run", quiet=True,
                task={"exp_name": FLAGS.exp_name,
                      "peg_shape": getattr(config, "peg_shape", None)},
                method=FLAGS.arm or aqi_cfg.mode, operator=op,
                seeds={"seed": FLAGS.seed,
                       "synthetic_seed": FLAGS.synthetic_seed,
                       "env_seed": (FLAGS.env_seed if FLAGS.env_seed >= 0
                                    else FLAGS.seed)},
                termination=term,
                conditions=_prov.condition_rows(aqi_cfg))
        except Exception:
            pass

    # Prompt, guaranteed shutdown so a blocked teardown/daemon thread can never
    # stall the campaign (the orchestrator waits on this process to exit before
    # it stops the learner). All transitions were already streamed in the loop.
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(0)


##############################################################################


def learner(rng, agent, replay_buffer, demo_buffer, aqi_cfg, wandb_logger=None):
    """
    The learner loop, which runs when "--learner" is set to True.
    Identical to train_rlpd.py; in quality modes the demo/intervention
    buffer is a QualityAwareReplayBufferDataStore, so 'sampling' below is
    the quality-prioritized sampling of proposal S5.6, and the agents'
    policy loss contains the weighted imitation term (lambda_BC > 0).
    """
    # latest_checkpoint returns None when the dir exists but holds no checkpoint
    # yet (fresh run, --resume scratch, or wandb having just created
    # <ckpt>/wandb/) -- guard against basename(None).
    _latest = (
        checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path))
        if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path)
        else None
    )
    start_step = int(os.path.basename(_latest)[11:]) + 1 if _latest else 0
    step = start_step

    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats request."""
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=step)
        return {}  # not expecting a response

    # Create server
    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.register_data_store("actor_env_intvn", demo_buffer)
    server.start(threaded=True)

    # ---- graceful stop --------------------------------------------------------
    # Stop training WITHOUT ctrl-C and still keep the weights: either create the
    # sentinel file <checkpoint_path>/STOP (touch it from any terminal or a task)
    # or send SIGINT/SIGTERM (ctrl-C, `kill <pid>`, the VS Code task Stop button).
    # The learner finishes the current step, writes a final checkpoint, and exits.
    _stop = {"flag": False}
    def _request_stop(signum, _frame):
        if _stop["flag"]:                       # second signal -> hard exit
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            raise KeyboardInterrupt
        _stop["flag"] = True
        print_yellow(f"\nstop requested (signal {signum}); finishing the current "
                     "step, then checkpointing and exiting (repeat to force-quit).")
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    stop_file = (os.path.join(os.path.abspath(FLAGS.checkpoint_path), "STOP")
                 if FLAGS.checkpoint_path else None)
    def _should_stop():
        return _stop["flag"] or (stop_file is not None and os.path.exists(stop_file))

    # --utd_ratio: fixed learner steps per env step (serl_launcher/utils/
    # update_ratio.py). Off: the original loop, unchanged. Set up before the
    # banner so a progress file left by an earlier run is gone before the
    # actor can read it; env steps count from here (a resumed buffer is not
    # new data).
    utd = (UpdateRatio(FLAGS.checkpoint_path, FLAGS.utd_ratio)
           if FLAGS.utd_ratio > 0.0 and FLAGS.checkpoint_path else None)
    if utd is not None:
        utd.learner_reset()
        _rb0 = len(replay_buffer)
        _rb_cap = getattr(replay_buffer, "_capacity", None)

        def _received():
            n = len(replay_buffer)
            return float("inf") if _rb_cap and n >= _rb_cap else n - _rb0

    if isinstance(agent, SACAgent):
        train_critic_networks_to_update = frozenset({"critic"})
        train_networks_to_update = frozenset({"critic", "actor", "temperature"})
    else:
        train_critic_networks_to_update = frozenset({"critic", "grasp_critic"})
        train_networks_to_update = frozenset({"critic", "grasp_critic", "actor", "temperature"})

    # Optional demo-only pretraining before online RL starts. This warms the
    # network using demonstration/intervention data so the actor starts from a
    # better policy/value prior once it connects.
    pretrain_steps = max(int(FLAGS.demo_pretrain_steps), 0)
    if not FLAGS.demo_pretrain:
        pretrain_steps = 0
    pretrain_bs = (int(FLAGS.demo_pretrain_batch_size)
                   if int(FLAGS.demo_pretrain_batch_size) > 0
                   else int(config.batch_size))
    if pretrain_steps > 0:
        print_green(
            f">>> demo pretrain: {pretrain_steps} step(s), batch_size={pretrain_bs}")
        demo_pretrain_iterator = demo_buffer.get_iterator(
            sample_args={
                "batch_size": pretrain_bs,
                "pack_obs_and_next_obs": True,
            },
            device=sharding.replicate(),
        )
        for _ in tqdm.tqdm(range(pretrain_steps), dynamic_ncols=True,
                           desc="demo-pretrain"):
            if _should_stop():
                break
            # Keep update semantics consistent with the main learner loop:
            # (cta_ratio-1) critic-only updates, then one full update.
            for _critic_step in range(config.cta_ratio - 1):
                batch = next(demo_pretrain_iterator)
                agent, _ = agent.update(
                    batch,
                    networks_to_update=train_critic_networks_to_update,
                )

            batch = next(demo_pretrain_iterator)
            agent, update_info = agent.update(
                batch,
                networks_to_update=train_networks_to_update,
            )

            if step > 0 and step % (config.steps_per_update) == 0:
                agent = jax.block_until_ready(agent)
                server.publish_network(agent.state.params)

            if step % config.log_period == 0 and wandb_logger:
                wandb_logger.log(
                    {"demo_pretrain": {"step": int(step),
                                        "batch_size": int(pretrain_bs)}},
                    step=step,
                )
                wandb_logger.log(update_info, step=step)

            if (
                step > 0
                and config.checkpoint_period
                and step % config.checkpoint_period == 0
            ):
                checkpoints.save_checkpoint(
                    os.path.abspath(FLAGS.checkpoint_path),
                    agent.state,
                    step=step,
                    keep=FLAGS.checkpoint_keep,
                )

            step += 1

        if _should_stop():
            if stop_file and os.path.exists(stop_file):
                os.remove(stop_file)
            print_yellow("stopped during demo pretraining; checkpointing final state.")
            if FLAGS.checkpoint_path:
                agent = jax.block_until_ready(agent)
                checkpoints.save_checkpoint(
                    os.path.abspath(FLAGS.checkpoint_path),
                    agent.state,
                    step=max(step, 0),
                    keep=FLAGS.checkpoint_keep,
                    overwrite=True,
                )
            return

    # The server is now accepting connections -- THIS is the moment to launch the
    # actor. Make it impossible to miss (banner + terminal bell '\a'). Starting
    # the actor before this point just makes it wait; starting it now connects.
    _bar = "=" * 68
    print_green("\a" + _bar)
    print_green(">>> LEARNER READY -- start the ACTOR now.")
    print_green("      VS Code:  Run Task -> 'sim: 3) Train - Actor'")
    print_green("      shell:    SIM_ACTION=actor bash "
                "examples/experiments/sim_peg_insertion/run.sh")
    print_green(f">>> Training auto-starts once the actor has streamed "
                f"{config.training_starts} online steps into the buffer below.")
    if stop_file:
        print_green(f">>> Stop + save anytime without ctrl-C:  touch {stop_file}")
    print_green(_bar)
    # flush now so the banner is never held in a block buffer (e.g. when the
    # learner's output is piped or redirected to a log file) -- the whole point
    # is that the operator sees it the instant the actor can connect.
    print("", end="", flush=True)

    # Loop to wait until replay_buffer is filled
    pbar = tqdm.tqdm(
        total=config.training_starts,
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
    )
    while len(replay_buffer) < config.training_starts and not _should_stop():
        pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
    pbar.close()
    if _should_stop():
        if stop_file and os.path.exists(stop_file):
            os.remove(stop_file)
        print_yellow("stopped before training began; nothing new to checkpoint.")
        return

    # send the initial network to the actor (after optional demo pretraining)
    server.publish_network(agent.state.params)
    print_green("sent initial network to actor")

    # 50/50 sampling from RLPD, half from demo and half from online experience
    replay_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size // 2,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )
    demo_iterator = demo_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size // 2,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )

    # wait till the replay buffer is filled with enough data
    timer = Timer()

    # NOTE (handoff FINDINGS F3): this loop is not tied to the actor. The
    # actor is paced at 10 Hz, while the learner updates as fast as its GPU
    # allows, so the updates per env step depend on the method's per-update
    # cost and on the GPU. Measured on an H100: quality/aqi ~2.6, hiril ~2.1,
    # e2hil ~1.3 learner steps per env step. Report it per method; changing
    # it here would make new cells incomparable with every finished one.
    online_start = step
    if utd is not None:             # training begins: the actor may now wait
        utd.learner_report(0)
        print_green(f"fixed update ratio: {FLAGS.utd_ratio:g} learner steps per env step")

    for step in tqdm.tqdm(
        range(step, config.max_steps), dynamic_ncols=True, desc="learner"
    ):
        # run n-1 critic updates and 1 critic + actor update.
        # This makes training on GPU faster by reducing the large batch transfer time from CPU to GPU
        for critic_step in range(config.cta_ratio - 1):
            with timer.context("sample_replay_buffer"):
                batch = next(replay_iterator)
                demo_batch = next(demo_iterator)
                if aqi_cfg.per_is_beta > 0.0:
                    batch = batch.copy(add_or_replace={"is_weight": jnp.ones(
                        (batch["rewards"].shape[0],), dtype=jnp.float32)})
                batch = concat_batches(batch, demo_batch, axis=0)

            with timer.context("train_critics"):
                agent, critics_info = agent.update(
                    batch,
                    networks_to_update=train_critic_networks_to_update,
                )

        with timer.context("train"):
            batch = next(replay_iterator)
            demo_batch = next(demo_iterator)
            if aqi_cfg.per_is_beta > 0.0:
                batch = batch.copy(add_or_replace={"is_weight": jnp.ones(
                    (batch["rewards"].shape[0],), dtype=jnp.float32)})
            batch = concat_batches(batch, demo_batch, axis=0)
            agent, update_info = agent.update(
                batch,
                networks_to_update=train_networks_to_update,
            )
        if utd is not None:
            _n_online = step - online_start + 1
            utd.learner_report(_n_online)
            utd.learner_wait(_n_online, _received, _should_stop)

        # publish the updated network
        if step > 0 and step % (config.steps_per_update) == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)

        if step % config.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=step)
            wandb_logger.log({"timer": timer.get_average_times()}, step=step)
            if aqi_cfg.quality_on and hasattr(demo_buffer, "priority_stats"):
                wandb_logger.log({"aqi": demo_buffer.priority_stats()},
                                 step=step)

        if (
            step > 0
            and config.checkpoint_period
            and step % config.checkpoint_period == 0
        ):
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path), agent.state, step=step, keep=FLAGS.checkpoint_keep
            )

        # graceful stop (STOP file / SIGINT / SIGTERM): write a final checkpoint
        # at the current step and exit cleanly. overwrite=True in case this exact
        # step was just checkpointed above.
        if _should_stop():
            print_green(f"\ngraceful stop at step {step} -> writing final checkpoint")
            if FLAGS.checkpoint_path:
                agent = jax.block_until_ready(agent)
                checkpoints.save_checkpoint(
                    os.path.abspath(FLAGS.checkpoint_path), agent.state,
                    step=step, keep=FLAGS.checkpoint_keep, overwrite=True,
                )
            if stop_file and os.path.exists(stop_file):
                os.remove(stop_file)   # so the next run doesn't stop immediately
            print_green(f"final checkpoint saved at step {step}; learner exiting.")
            break

    if wandb_logger is not None:
        wandb_logger.finish()


##############################################################################


def main(_):
    global config
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    if FLAGS.peg_shape is not None:
        setattr(config, "peg_shape", FLAGS.peg_shape)
    if FLAGS.max_steps > 0:
        config.max_steps = FLAGS.max_steps   # per-cell budget (campaign runner)
    if FLAGS["buffer_period"].present and FLAGS.buffer_period >= 0:
        config.buffer_period = FLAGS.buffer_period  # 0 disables the ~9GB dumps
    aqi_cfg = config.get_aqi_config(mode=FLAGS.aqi_mode)

    # ---- composition overrides from the command line ----------------------
    def _csv(v):
        v = (v or "").strip()
        return () if v in ("", "none") else tuple(s.strip() for s in v.split(","))
    overrides = {}
    if FLAGS["trigger_terms"].present:
        overrides["trigger_terms"] = _csv(FLAGS.trigger_terms)
    if FLAGS["trigger_combiner"].present:
        overrides["trigger_combiner"] = FLAGS.trigger_combiner
    if FLAGS["quality_disagreement_gate"].present:
        overrides["quality_disagreement_gate"] = FLAGS.quality_disagreement_gate
    if FLAGS["quality_filters"].present:
        overrides["quality_filters"] = _csv(FLAGS.quality_filters)
    if FLAGS["bc_loss_weight"].present:
        overrides["bc_loss_weight"] = FLAGS.bc_loss_weight
    if FLAGS["per_standardize"].present:
        overrides["per_standardize"] = FLAGS.per_standardize
    if FLAGS["per_delta_init"].present:
        overrides["per_delta_init"] = FLAGS.per_delta_init
    if FLAGS["per_eta"].present:
        overrides["per_eta"] = FLAGS.per_eta
    if FLAGS["per_eps"].present:
        overrides["per_eps"] = FLAGS.per_eps
    if FLAGS["quality_warmup_steps"].present:
        overrides["quality_warmup_steps"] = FLAGS.quality_warmup_steps
    if FLAGS["per_scale_decay"].present:
        overrides["per_scale_decay"] = FLAGS.per_scale_decay
    if FLAGS["per_refresh_period"].present:
        overrides["per_refresh_period"] = FLAGS.per_refresh_period
    if FLAGS["per_is_beta"].present:
        overrides["per_is_beta"] = FLAGS.per_is_beta
    if FLAGS["quality_score_mode"].present:
        overrides["quality_score_mode"] = FLAGS.quality_score_mode
    if FLAGS["quality_behavioral_terms"].present:
        overrides["quality_behavioral_terms"] = _csv(
            FLAGS.quality_behavioral_terms)
    for _key in ("quality_post_release_steps", "quality_hindsight_discount",
                 "quality_consistency_weight", "quality_consistency_k",
                 "quality_consistency_sigma_ref", "quality_cf_diag"):
        if FLAGS[_key].present:
            overrides[_key] = getattr(FLAGS, _key)
    # the hindsight advantage must discount like the critic it reads
    if (not FLAGS["quality_hindsight_discount"].present
            and aqi_cfg.logging_on
            and getattr(config, "discount", None) is not None
            and float(config.discount) != aqi_cfg.quality_hindsight_discount):
        overrides["quality_hindsight_discount"] = float(config.discount)
    if FLAGS["bootstrap_p"].present:
        overrides["critic_bootstrap_p"] = FLAGS.bootstrap_p
    if FLAGS["per_member_targets"].present:
        overrides["critic_per_member_targets"] = FLAGS.per_member_targets
    if FLAGS["critic_ensemble_size"].present:
        overrides["critic_ensemble_size"] = FLAGS.critic_ensemble_size
        # keep the REDQ min-Q subsample valid (<= ensemble size)
        overrides["critic_subsample_size"] = min(
            aqi_cfg.critic_subsample_size or FLAGS.critic_ensemble_size,
            FLAGS.critic_ensemble_size)
    if FLAGS["critic_subsample_size"].present:
        # explicit subsample M wins over the auto value above; clamp to K.
        _K = overrides.get("critic_ensemble_size", aqi_cfg.critic_ensemble_size)
        overrides["critic_subsample_size"] = min(FLAGS.critic_subsample_size, _K)
    for _key in ("e2hil_K", "e2hil_q_low", "e2hil_q_high", "e2hil_q_agg"):
        if FLAGS[_key].present:
            overrides[_key] = getattr(FLAGS, _key)

    # Drop any override that is IRRELEVANT to the selected mode, so an option
    # the task always asks for (e.g. a quality filter while running the
    # hilserl baseline) can never change the run -- picking any value is then
    # identical to picking any other. Relevance follows the mode switches:
    #   trigger_*  matter only when the active gate acts   (active, aqi)
    #   quality_*  matter only for quality replay + loss   (quality, aqi),
    #              except the post-release window (K, discount), which also
    #              shapes the features logged in every logging mode
    #   critic_*   matter only for the K-critic ensemble   (every mode but
    #              hilserl/e2hil, fixed at the 2-critic baseline).
    #   e2hil_*    matter only for the e2hil baseline.
    def _override_relevant(key: str) -> bool:
        if key.startswith("e2hil_"):
            return aqi_cfg.e2hil_on
        if key in ("trigger_terms", "trigger_combiner"):
            return aqi_cfg.active_on
        if key in ("quality_disagreement_gate", "quality_filters",
                   "bc_loss_weight", "per_standardize", "per_delta_init",
                   "per_eta", "per_eps", "quality_warmup_steps",
                   "per_scale_decay", "per_refresh_period", "per_is_beta",
                   "quality_score_mode", "quality_behavioral_terms",
                   "quality_consistency_weight", "quality_consistency_k",
                   "quality_consistency_sigma_ref"):
            return aqi_cfg.quality_on
        if key in ("quality_post_release_steps", "quality_hindsight_discount",
                   "quality_cf_diag"):
            # also shape the LOGGED a_hind / d_progress_post features;
            # quality_cf_diag is logging-only in every logging mode
            return aqi_cfg.logging_on
        if key in ("critic_ensemble_size", "critic_subsample_size",
                   "critic_bootstrap_p", "critic_per_member_targets"):
            return not aqi_cfg.baseline_arch
        return True
    ignored_overrides = {k: v for k, v in overrides.items()
                         if not _override_relevant(k)}
    overrides = {k: v for k, v in overrides.items() if _override_relevant(k)}
    if ignored_overrides:
        print_yellow(
            f"mode '{aqi_cfg.mode}': ignoring options that do not apply to it "
            f"(no effect on this run): {sorted(ignored_overrides)}")

    if overrides:
        aqi_cfg = dataclasses.replace(aqi_cfg, **overrides)
        print_green(f"AQI composition overrides: {overrides}")

    if FLAGS.aqi_log_dir:
        aqi_cfg = dataclasses.replace(
            aqi_cfg, log_dir=os.path.abspath(FLAGS.aqi_log_dir))
        print_green(f"AQI logs -> {aqi_cfg.log_dir}")

    assert config.batch_size % num_devices == 0
    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)

    # Synthetic actor runs do not need a physical SpaceMouse. Set an env toggle
    # that sim_peg_insertion/config.py reads to skip the Spacemouse wrapper.
    if FLAGS.actor and bool(FLAGS.synthetic_operator):
        os.environ["SIM_DISABLE_SPACEMOUSE"] = "1"

    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    env = config.get_environment(
        fake_env=FLAGS.learner,
        save_video=FLAGS.save_video,
        classifier=True,
    )
    env = RecordEpisodeStatistics(env)

    rng, sampling_rng = jax.random.split(rng)

    # ---- agent construction ------------------------------------------------
    # mode=hilserl keeps the exact baseline architecture (2 critics, no
    # subsampling, lambda_BC=0). Every other mode uses the K-critic ensemble
    # of S5.3 (REDQ-style: sigma_Q over all K, min-Q backup over a random
    # subsample), which is what "Logged HIL-SERL" in S6.4 is for: the
    # same-architecture diagnostic control against the active/quality modes.
    # e2hil keeps hilserl's exact 2-critic architecture (it only changes the
    # actor loss), so hilserl and e2hil checkpoints are interchangeable.
    if aqi_cfg.baseline_arch:
        ensemble_size, subsample_size = 2, None
    else:
        ensemble_size = aqi_cfg.critic_ensemble_size
        subsample_size = aqi_cfg.critic_subsample_size
    bc_weight = aqi_cfg.bc_loss_weight if aqi_cfg.quality_on else 0.0
    # bootstrapped ensemble (different data per member + decorrelated
    # targets) applies to every instrumented mode; hilserl stays the
    # exact 2-critic shared-target baseline.
    per_member_targets = (aqi_cfg.critic_per_member_targets
                          if not aqi_cfg.baseline_arch else False)
    bootstrap_kwargs = (
        dict(bootstrap_ensemble_size=ensemble_size,
             bootstrap_p=aqi_cfg.critic_bootstrap_p)
        if not aqi_cfg.baseline_arch and aqi_cfg.critic_bootstrap_p < 1.0
        else {})

    agent_kwargs = dict(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=config.image_keys,
        encoder_type=config.encoder_type,
        discount=config.discount,
        critic_ensemble_size=ensemble_size,
        critic_subsample_size=subsample_size,
        aqi_bc_weight=bc_weight,
        critic_per_member_targets=per_member_targets,
    )
    if aqi_cfg.e2hil_on and config.setup_mode not in (
            'single-arm-fixed-gripper', 'dual-arm-fixed-gripper'):
        raise NotImplementedError(
            "--aqi_mode=e2hil is implemented for SACAgent (fixed-gripper "
            f"setups) only, not '{config.setup_mode}'.")
    if config.setup_mode == 'single-arm-fixed-gripper' or config.setup_mode == 'dual-arm-fixed-gripper':
        if FLAGS.learning_rate and FLAGS.learning_rate > 0:
            agent_kwargs = {**agent_kwargs, "learning_rate": float(FLAGS.learning_rate)}
        if aqi_cfg.e2hil_on:
            agent_kwargs = {**agent_kwargs, "e2hil": True,
                            "e2hil_K": aqi_cfg.e2hil_K,
                            "e2hil_q_low": aqi_cfg.e2hil_q_low,
                            "e2hil_q_high": aqi_cfg.e2hil_q_high,
                            "e2hil_q_agg": aqi_cfg.e2hil_q_agg}
            print_green(
                f"E2HiL actor loss ON: K={aqi_cfg.e2hil_K}, |c| band = "
                f"[{aqi_cfg.e2hil_q_low:.2f}, {aqi_cfg.e2hil_q_high:.2f}] "
                f"batch percentiles, Q reduction = {aqi_cfg.e2hil_q_agg}")
        agent: SACAgent = make_sac_pixel_agent(**agent_kwargs)
        include_grasp_penalty = False
    elif config.setup_mode == 'single-arm-learned-gripper':
        agent: SACAgentHybridSingleArm = make_sac_pixel_agent_hybrid_single_arm(**agent_kwargs)
        include_grasp_penalty = True
    elif config.setup_mode == 'dual-arm-learned-gripper':
        agent: SACAgentHybridDualArm = make_sac_pixel_agent_hybrid_dual_arm(**agent_kwargs)
        include_grasp_penalty = True
    else:
        raise NotImplementedError(f"Unknown setup mode: {config.setup_mode}")

    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent = jax.device_put(
        jax.tree_util.tree_map(jnp.array, agent), sharding.replicate()
    )

    # ---- continue vs. retrain-from-scratch --------------------------------
    # Demos (recorded once) are architecture-independent; a training run's
    # CHECKPOINTS are not. When a run for this task+method already exists the
    # LEARNER either carries on from the last checkpoint (--resume=continue) or
    # deletes it and trains a new network from zero (--resume=scratch). The
    # actor never deletes: it warm-starts read-only from whatever the learner
    # has written and then receives live weights. Skipped while evaluating
    # (--eval_checkpoint_step restores its own specific step above).
    if not FLAGS.eval_checkpoint_step and FLAGS.checkpoint_path is not None:
        abspath = os.path.abspath(FLAGS.checkpoint_path)
        has_ckpt = (
            os.path.exists(abspath)
            and checkpoints.latest_checkpoint(abspath) is not None
        )

        def _restore_compatible():
            """Restore iff the on-disk checkpoint matches the current agent's
            architecture. Returns (restored_state_or_None, error_or_None)."""
            try:
                restored = checkpoints.restore_checkpoint(abspath, agent.state)
            except Exception as exc:            # structural mismatch / corruption
                return None, exc
            if not _shapes_match(agent.state, restored):
                return None, None               # shape mismatch, e.g. K changed
            return restored, None

        def _delete_run_dir():
            """Delete a previous run directory (checkpoints + buffers) so the
            architecture is retrained from zero. Guarded against obviously
            unsafe paths (filesystem root / home / cwd)."""
            if not os.path.exists(abspath):
                return
            protected = {os.path.abspath(os.sep),
                         os.path.abspath(os.path.expanduser("~")),
                         os.path.abspath(os.getcwd())}
            if abspath in protected:
                raise SystemExit(
                    f"refusing to delete '{abspath}': unsafe checkpoint path.")
            shutil.rmtree(abspath)
            print_yellow(f"retrain from scratch: deleted previous "
                         f"checkpoints at {abspath}")

        if FLAGS.learner:
            if FLAGS.resume == "scratch":
                _delete_run_dir()
                print_green("retrain from scratch: training a new network from zero.")
            elif has_ckpt:
                restored, err = _restore_compatible()
                if restored is not None:
                    agent = agent.replace(state=restored)
                    step_no = os.path.basename(
                        checkpoints.latest_checkpoint(abspath))[11:]
                    print_green(f"continue: resumed from checkpoint step {step_no}.")
                else:
                    detail = f"\n    {type(err).__name__}: {err}" if err else ""
                    raise SystemExit(
                        f"[continue] cannot resume {abspath}: the saved network's "
                        "architecture differs from the current settings (e.g. a "
                        "different --critic_ensemble_size or --aqi_mode)." + detail +
                        "\nThere is nothing compatible to continue. Re-run the "
                        "Learner and choose 'retrain from scratch' to delete the "
                        "old checkpoints and train this architecture from zero.")
            else:
                print_green("continue: no previous checkpoint found; "
                            "starting a new run from zero.")

        elif FLAGS.actor and has_ckpt and FLAGS.resume != "scratch":
            # Read-only warm start; the learner owns the directory and will
            # publish authoritative weights shortly. Never delete here.
            restored, _err = _restore_compatible()
            if restored is not None:
                agent = agent.replace(state=restored)
                step_no = os.path.basename(
                    checkpoints.latest_checkpoint(abspath))[11:]
                print_green(f"actor: warm-started from checkpoint {step_no}.")
            else:
                print_yellow("actor: on-disk checkpoint doesn't match the "
                             "current flags; waiting for the learner's weights.")

    include_aqi_fields = aqi_cfg.quality_on
    # e2hil: every transition carries its source label (0 demo, 1 online
    # policy, 2 intervention) so the learner can log E2HiL's per-source
    # entropy-influence / keep-rate diagnostics. Diagnostics only.
    include_label = aqi_cfg.e2hil_on

    def create_replay_buffer_and_wandb_logger():
        replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
            include_aqi_fields=include_aqi_fields,
            include_label=include_label,
            **bootstrap_kwargs,
        )
        # set up wandb and logging
        wandb_logger = make_wandb_logger(
            project="hil-serl",
            description=(f"{FLAGS.exp_name}_{FLAGS.arm or aqi_cfg.mode}_"
                         f"{FLAGS.operator_id}"),
            debug=FLAGS.debug,
            output_dir=(os.path.join(os.path.abspath(FLAGS.checkpoint_path),
                                     "wandb")
                        if FLAGS.checkpoint_path else None),
        )
        return replay_buffer, wandb_logger

    if FLAGS.learner:
        sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())
        replay_buffer, wandb_logger = create_replay_buffer_and_wandb_logger()
        if include_aqi_fields:
            # quality/aqi: intervention buffer with S5.6 priority sampling
            #   p_i = (eps + |delta_i| + eta * max(q_int_i, 0)) ** alpha
            demo_buffer = QualityAwareReplayBufferDataStore(
                env.observation_space,
                env.action_space,
                capacity=config.replay_buffer_capacity,
                image_keys=config.image_keys,
                include_grasp_penalty=include_grasp_penalty,
                per_eps=aqi_cfg.per_eps,
                per_eta=aqi_cfg.per_eta,
                per_alpha=aqi_cfg.per_alpha,
                per_delta_init=aqi_cfg.per_delta_init,
                demo_weight=aqi_cfg.demo_weight,
                demo_mask=1.0 if aqi_cfg.demo_weight > 0 else 0.0,
                per_standardize=aqi_cfg.per_standardize,
                per_scale_decay=aqi_cfg.per_scale_decay,
                per_refresh_period=aqi_cfg.per_refresh_period,
                per_is_beta=aqi_cfg.per_is_beta,
                **bootstrap_kwargs,
            )
        else:
            demo_buffer = MemoryEfficientReplayBufferDataStore(
                env.observation_space,
                env.action_space,
                capacity=config.replay_buffer_capacity,
                image_keys=config.image_keys,
                include_grasp_penalty=include_grasp_penalty,
                include_label=include_label,
                **bootstrap_kwargs,
            )

        assert FLAGS.demo_path is not None
        for path in FLAGS.demo_path:
            with open(path, "rb") as f:
                transitions = pkl.load(f)
                for transition in transitions:
                    if 'infos' in transition and 'grasp_penalty' in transition['infos']:
                        transition['grasp_penalty'] = transition['infos']['grasp_penalty']
                    if include_label:
                        transition.setdefault("labels", 0)
                    demo_buffer.insert(transition)
        print_green(f"demo buffer size: {len(demo_buffer)}")
        print_green(f"online buffer size: {len(replay_buffer)}")

        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "buffer")
        ):
            for file in glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        if include_aqi_fields:
                            transition.setdefault("q_int", np.float32(0.0))
                            transition.setdefault("quality_weight", np.float32(0.0))
                            transition.setdefault("intervention_mask", np.float32(0.0))
                        if include_label:
                            transition.setdefault("labels", 1)
                        replay_buffer.insert(transition)
            print_green(
                f"Loaded previous buffer data. Replay buffer size: {len(replay_buffer)}"
            )

        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "demo_buffer")
        ):
            for file in glob.glob(
                os.path.join(FLAGS.checkpoint_path, "demo_buffer/*.pkl")
            ):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        if include_label:
                            transition.setdefault("labels", 2)
                        demo_buffer.insert(transition)
            print_green(
                f"Loaded previous demo buffer data. Demo buffer size: {len(demo_buffer)}"
            )

        # learner loop
        print_green(f"starting learner loop (aqi_mode={aqi_cfg.mode})")
        learner(
            sampling_rng,
            agent,
            replay_buffer,
            demo_buffer=demo_buffer,
            aqi_cfg=aqi_cfg,
            wandb_logger=wandb_logger,
        )

    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        data_store = QueuedDataStore(50000)  # the queue size on the actor
        intvn_data_store = QueuedDataStore(50000)

        # actor loop
        print_green(f"starting actor loop (aqi_mode={aqi_cfg.mode})")
        actor(
            agent,
            data_store,
            intvn_data_store,
            env,
            sampling_rng,
            aqi_cfg,
        )

    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    app.run(main)
