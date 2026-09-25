"""Collect synthetic demos using a POLICY oracle as the competent core.

This is Role-1 demo collection with a SyntheticOperator whose intent comes from
PolicyOracle (doc S4 option B): a frozen policy checkpoint provides the core
intent, and the operator degrades it via wrong-intent segments + motor channel
noise exactly as in interventions.

Output format matches examples/record_demos.py and collect_synthetic_demos.py:
transitions pkl with observations/actions/next_observations/rewards/masks/dones/
infos, plus a *_groundtruth.json sidecar with theta and episode metadata.

Typical usage:
  python collect_policy_synthetic_demos.py \
    --exp_name sim_peg_insertion \
    --checkpoint_path ./run_cylinder_hilserl \
    --profile novice --episodes 20 --filtered \
    --out ./demo_data/sim_peg_insertion_cylinder_policy_novice_demos.pkl
"""

import copy
import json
import os
import pickle as pkl

import jax
import jax.numpy as jnp
import numpy as np
from absl import app, flags
from flax.training import checkpoints

try:
    import cv2
except ImportError:
    cv2 = None

from experiments.mappings import CONFIG_MAPPING
from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.agents.continuous.sac_hybrid_dual import SACAgentHybridDualArm
from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from serl_launcher.aqi.operator_model import (
    PolicyOracle,
    SyntheticOperator,
    make_operator_theta,
    theta_from_json,
)
from serl_launcher.utils.launcher import (
    make_sac_pixel_agent,
    make_sac_pixel_agent_hybrid_dual_arm,
    make_sac_pixel_agent_hybrid_single_arm,
)

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", "sim_peg_insertion", "Experiment name in CONFIG_MAPPING.")
flags.DEFINE_string("checkpoint_path", "", "Path to the frozen policy checkpoint directory.")
flags.DEFINE_integer("checkpoint_step", 0, "Specific checkpoint step (0 = latest in path).")
flags.DEFINE_string("aqi_mode", "hilserl", "Mode used to build the agent architecture.")
flags.DEFINE_integer("critic_ensemble_size", 0,
                     "Override K critics used to build the oracle agent (0 = mode default).")

flags.DEFINE_string("profile", "expert",
                    "operator profile: tier name, 'op1'..'op4', 's=0.7', or 'exec=expert,mon=novice'")
flags.DEFINE_string("theta_json", "",
                    "Path to an exact operator theta JSON (overrides --profile).")
flags.DEFINE_string("operator_id", "", "Override operator id in logs/sidecar.")
flags.DEFINE_boolean("jitter", False, "Apply per-operator theta jitter when using --profile.")

flags.DEFINE_integer("episodes", 20, "Episodes to KEEP (filtered) or to run (unfiltered).")
flags.DEFINE_integer("max_attempts", 400, "Attempt cap in filtered mode.")
flags.DEFINE_boolean("filtered", True, "Keep only successful episodes.")
flags.DEFINE_integer("seed", 0, "Random seed for env/operator.")
flags.DEFINE_float("policy_env_mm_per_unit", 100.0,
                   "mm/s represented by one env action unit for policy intent scaling.")
flags.DEFINE_string("out", "./policy_synthetic_demos.pkl", "Output pkl path.")
flags.DEFINE_boolean("render", False, "Show a live window while collecting demos.")


def _build_agent(config, env):
    aqi_cfg = config.get_aqi_config(mode=FLAGS.aqi_mode)
    if FLAGS.critic_ensemble_size > 0:
        ensemble_size = FLAGS.critic_ensemble_size
        subsample_size = min(aqi_cfg.critic_subsample_size or ensemble_size,
                             ensemble_size)
    elif aqi_cfg.mode == "hilserl":
        ensemble_size, subsample_size = 2, None
    else:
        ensemble_size = aqi_cfg.critic_ensemble_size
        subsample_size = aqi_cfg.critic_subsample_size

    agent_kwargs = dict(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=config.image_keys,
        encoder_type=config.encoder_type,
        discount=config.discount,
        critic_ensemble_size=ensemble_size,
        critic_subsample_size=subsample_size,
        aqi_bc_weight=0.0,
        critic_per_member_targets=(aqi_cfg.critic_per_member_targets
                                   if aqi_cfg.mode != "hilserl" else False),
    )

    if config.setup_mode in ("single-arm-fixed-gripper", "dual-arm-fixed-gripper"):
        agent: SACAgent = make_sac_pixel_agent(**agent_kwargs)
    elif config.setup_mode == "single-arm-learned-gripper":
        agent: SACAgentHybridSingleArm = make_sac_pixel_agent_hybrid_single_arm(**agent_kwargs)
    elif config.setup_mode == "dual-arm-learned-gripper":
        agent: SACAgentHybridDualArm = make_sac_pixel_agent_hybrid_dual_arm(**agent_kwargs)
    else:
        raise NotImplementedError(f"Unknown setup mode: {config.setup_mode}")

    return jax.device_put(jax.tree_util.tree_map(jnp.array, agent), jax.devices()[0])


def _make_theta(rng):
    if FLAGS.theta_json:
        theta = theta_from_json(FLAGS.theta_json)
        if FLAGS.operator_id:
            theta.operator_id = FLAGS.operator_id
        return theta
    return make_operator_theta(FLAGS.profile, rng=rng,
                               jitter=FLAGS.jitter,
                               operator_id=FLAGS.operator_id or None)


def _success_from_info(info):
    return bool(info.get("succeeded", info.get("succeed", False)))


def main(_):
    if not FLAGS.checkpoint_path:
        raise SystemExit("--checkpoint_path is required")
    if FLAGS.exp_name not in CONFIG_MAPPING:
        raise SystemExit(f"unknown --exp_name {FLAGS.exp_name}")

    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=True, save_video=False, classifier=True)

    agent = _build_agent(config, env)
    ckpt_dir = os.path.abspath(FLAGS.checkpoint_path)
    restored = checkpoints.restore_checkpoint(
        ckpt_dir, agent.state,
        step=(FLAGS.checkpoint_step if FLAGS.checkpoint_step > 0 else None),
    )
    oracle_agent = agent.replace(state=restored)
    key = jax.random.PRNGKey(FLAGS.seed)

    def policy_fn(obs):
        act = oracle_agent.sample_actions(
            observations=jax.device_put(obs),
            argmax=True,
            seed=key,
        )
        return np.asarray(jax.device_get(act))

    mm_per_unit = float(FLAGS.policy_env_mm_per_unit)
    policy_oracle = PolicyOracle(
        policy_fn,
        action_dim=env.action_space.shape[0],
        action_scale=mm_per_unit,
    )
    theta = _make_theta(np.random.default_rng(FLAGS.seed))
    op = SyntheticOperator(
        policy_oracle,
        theta,
        rng=np.random.default_rng(FLAGS.seed + 17),
        action_scale=1.0 / mm_per_unit,
    )

    render_on = bool(FLAGS.render and cv2 is not None)
    if FLAGS.render and cv2 is None:
        print("render requested but opencv-python is not installed; continuing headless")
    if render_on:
        cv2.namedWindow("policy_synthetic_demos", cv2.WINDOW_NORMAL)

    transitions, ep_meta = [], []
    kept = attempts = 0
    while kept < FLAGS.episodes and attempts < FLAGS.max_attempts:
        ep_seed = FLAGS.seed * 100000 + attempts
        obs, info = env.reset(seed=ep_seed)
        op.reset()
        traj = []
        done = False
        steps = 0
        ep_return = 0.0

        while not done:
            act = np.asarray(op.act(obs, contact=bool(info.get("contact", False))),
                             dtype=np.float32)
            next_obs, rew, term, trunc, info = env.step(act)
            done = bool(term or trunc)

            if render_on:
                frame = env.render()
                if frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.imshow("policy_synthetic_demos", frame)
                    cv2.waitKey(1)

            traj.append(copy.deepcopy(dict(
                observations=obs,
                actions=act,
                next_observations=next_obs,
                rewards=float(rew),
                masks=1.0 - float(done),
                dones=float(done),
                infos=dict(succeed=_success_from_info(info)),
            )))
            obs = next_obs
            ep_return += float(rew)
            steps += 1

        success = _success_from_info(info)
        attempts += 1
        keep = success or (not FLAGS.filtered)
        if keep:
            transitions.extend(traj)
            kept += 1

        ep_meta.append(dict(
            attempt=attempts - 1,
            seed=ep_seed,
            kept=bool(keep),
            success=bool(success),
            steps=steps,
            duration_s=steps / 10.0,
            return_sum=ep_return,
            commit_err=(op.commit_events[0][1] if op.commit_events else None),
            attempts_within=op.n_approaches,
        ))
        print(f"[{attempts:3d}] success={success} kept={keep} steps={steps} "
              f"(kept {kept}/{FLAGS.episodes})")

    os.makedirs(os.path.dirname(os.path.abspath(FLAGS.out)), exist_ok=True)
    with open(FLAGS.out, "wb") as f:
        pkl.dump(transitions, f)

    side = dict(
        source="policy_oracle",
        exp_name=FLAGS.exp_name,
        checkpoint_path=ckpt_dir,
        checkpoint_step=(FLAGS.checkpoint_step if FLAGS.checkpoint_step > 0 else "latest"),
        mode=FLAGS.aqi_mode,
        profile=FLAGS.profile,
        theta=theta.as_dict(),
        filtered=bool(FLAGS.filtered),
        seed=FLAGS.seed,
        policy_env_mm_per_unit=mm_per_unit,
        episodes_kept=kept,
        attempts=attempts,
        attempts_per_kept=attempts / max(kept, 1),
        success_rate=sum(m["success"] for m in ep_meta) / max(attempts, 1),
        episodes=ep_meta,
    )
    side_path = os.path.splitext(FLAGS.out)[0] + "_groundtruth.json"
    with open(side_path, "w", encoding="utf-8") as f:
        json.dump(side, f, indent=1)

    print(f"\nwrote {len(transitions)} transitions from {kept} episodes -> {FLAGS.out}")
    print(f"ground truth -> {side_path}")

    if render_on:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app.run(main)
