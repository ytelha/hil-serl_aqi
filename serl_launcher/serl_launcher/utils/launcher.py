# !/usr/bin/env python3

import os

import jax
from jax import nn
import jax.numpy as jnp

from agentlace.trainer import TrainerConfig

from serl_launcher.common.typing import Batch, PRNGKey
from serl_launcher.common.wandb import WandBLogger
from serl_launcher.agents.continuous.bc import BCAgent
from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from serl_launcher.agents.continuous.sac_hybrid_dual import SACAgentHybridDualArm
from serl_launcher.vision.data_augmentations import batched_random_crop

##############################################################################


def make_bc_agent(
    seed, 
    sample_obs, 
    sample_action, 
    image_keys=("image",), 
    encoder_type="resnet-pretrained"
):
    return BCAgent.create(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [512, 512, 512],
            "dropout_rate": 0.25,
        },
        policy_kwargs={
            "tanh_squash_distribution": False,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        use_proprio=True,
        encoder_type=encoder_type,
        image_keys=image_keys,
        augmentation_function=make_batch_augmentation_func(image_keys),
    )


def make_sac_pixel_agent(
    seed,
    sample_obs,
    sample_action,
    image_keys=("image",),
    encoder_type="resnet-pretrained",
    reward_bias=0.0,
    target_entropy=None,
    discount=0.97,
    # AQI-HIL-SERL (proposal S5.3/S5.6): a larger critic ensemble provides
    # the epistemic-uncertainty signal sigma_Q; subsampling the TD target
    # keeps the conservative min-Q backup (REDQ-style). aqi_bc_weight is
    # lambda_BC of the quality-weighted intervention loss; the baseline
    # defaults (2, None, 0.0) leave behaviour unchanged.
    critic_ensemble_size=2,
    critic_subsample_size=None,
    aqi_bc_weight=0.0,
    critic_per_member_targets=False,
    learning_rate=None,
    # E2HiL comparison baseline (agents/continuous/e2hil.py). Keys enter the
    # agent config only when enabled, so other agents' configs are unchanged.
    e2hil=False,
    e2hil_K=8,
    e2hil_q_low=0.05,
    e2hil_q_high=0.90,
    e2hil_q_agg="min",
):
    opt_kwargs = {}
    if learning_rate is not None:
        lr = float(learning_rate)
        opt_kwargs = dict(
            actor_optimizer_kwargs={"learning_rate": lr},
            critic_optimizer_kwargs={"learning_rate": lr},
            temperature_optimizer_kwargs={"learning_rate": lr},
        )
    e2hil_kwargs = (dict(e2hil=True, e2hil_K=int(e2hil_K),
                         e2hil_q_low=float(e2hil_q_low),
                         e2hil_q_high=float(e2hil_q_high),
                         e2hil_q_agg=str(e2hil_q_agg))
                    if e2hil else {})
    agent = SACAgent.create_pixels(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        encoder_type=encoder_type,
        use_proprio=True,
        image_keys=image_keys,
        policy_kwargs={
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        temperature_init=1e-2,
        discount=discount,
        backup_entropy=False,
        critic_ensemble_size=critic_ensemble_size,
        critic_subsample_size=critic_subsample_size,
        reward_bias=reward_bias,
        target_entropy=target_entropy,
        augmentation_function=make_batch_augmentation_func(image_keys),
        aqi_bc_weight=aqi_bc_weight,
        critic_per_member_targets=critic_per_member_targets,
        **opt_kwargs,
        **e2hil_kwargs,
    )
    return agent


def make_hiril_pixel_agent(
    seed,
    sample_obs,
    sample_action,
    image_keys=("image",),
    encoder_type="resnet-pretrained",
    reward_bias=0.0,
    discount=0.97,
    hiril=None,
):
    """HiRIL baseline agent (serl_launcher/hiril/agent.py): TD3 actor + BIQN
    critic, on the SAME pixel encoder / pretrained ResNet-10 / random-crop
    augmentation as make_sac_pixel_agent, so the comparison differs only in
    the learning method. ``hiril`` = HiRILConfig (None = paper defaults)."""
    from serl_launcher.hiril.agent import (HiRILAgent, make_hiril_defs,
                                           make_pixel_encoder)
    from serl_launcher.hiril.config import HiRILConfig
    from serl_launcher.utils.train_utils import load_resnet10_params

    hiril = hiril or HiRILConfig()
    encoder = make_pixel_encoder(encoder_type, image_keys)
    actor_def, critic_def = make_hiril_defs(sample_action.shape[-1], hiril,
                                            encoder=encoder)
    agent = HiRILAgent.create(
        jax.random.PRNGKey(seed), sample_obs, sample_action,
        actor_def=actor_def, critic_def=critic_def, hiril=hiril,
        discount=discount, image_keys=image_keys,
        augmentation_function=make_batch_augmentation_func(image_keys),
        reward_bias=reward_bias)
    if "pretrained" in encoder_type:
        agent = load_resnet10_params(agent, image_keys)
        # start the target networks from the loaded weights as well
        agent = agent.replace(state=agent.state.replace(
            target_params=agent.state.params))
    return agent


def make_sac_pixel_agent_hybrid_single_arm(
    seed,
    sample_obs,
    sample_action,
    image_keys=("image",),
    encoder_type="resnet-pretrained",
    reward_bias=0.0,
    target_entropy=None,
    discount=0.97,
    # AQI-HIL-SERL (proposal S5.3/S5.6): a larger critic ensemble provides
    # the epistemic-uncertainty signal sigma_Q; subsampling the TD target
    # keeps the conservative min-Q backup (REDQ-style). aqi_bc_weight is
    # lambda_BC of the quality-weighted intervention loss; the baseline
    # defaults (2, None, 0.0) leave behaviour unchanged.
    critic_ensemble_size=2,
    critic_subsample_size=None,
    aqi_bc_weight=0.0,
    critic_per_member_targets=False,
):
    agent = SACAgentHybridSingleArm.create_pixels(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        encoder_type=encoder_type,
        use_proprio=True,
        image_keys=image_keys,
        policy_kwargs={
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        grasp_critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        temperature_init=1e-2,
        discount=discount,
        backup_entropy=False,
        critic_ensemble_size=critic_ensemble_size,
        critic_subsample_size=critic_subsample_size,
        reward_bias=reward_bias,
        target_entropy=target_entropy,
        augmentation_function=make_batch_augmentation_func(image_keys),
        aqi_bc_weight=aqi_bc_weight,
        critic_per_member_targets=critic_per_member_targets,
    )
    return agent


def make_sac_pixel_agent_hybrid_dual_arm(
    seed,
    sample_obs,
    sample_action,
    image_keys=("image",),
    encoder_type="resnet-pretrained",
    reward_bias=0.0,
    target_entropy=None,
    discount=0.97,
    # AQI-HIL-SERL (proposal S5.3/S5.6): a larger critic ensemble provides
    # the epistemic-uncertainty signal sigma_Q; subsampling the TD target
    # keeps the conservative min-Q backup (REDQ-style). aqi_bc_weight is
    # lambda_BC of the quality-weighted intervention loss; the baseline
    # defaults (2, None, 0.0) leave behaviour unchanged.
    critic_ensemble_size=2,
    critic_subsample_size=None,
    aqi_bc_weight=0.0,
    critic_per_member_targets=False,
):
    agent = SACAgentHybridDualArm.create_pixels(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        encoder_type=encoder_type,
        use_proprio=True,
        image_keys=image_keys,
        policy_kwargs={
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        grasp_critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        temperature_init=1e-2,
        discount=discount,
        backup_entropy=False,
        critic_ensemble_size=critic_ensemble_size,
        critic_subsample_size=critic_subsample_size,
        reward_bias=reward_bias,
        target_entropy=target_entropy,
        augmentation_function=make_batch_augmentation_func(image_keys),
        aqi_bc_weight=aqi_bc_weight,
        critic_per_member_targets=critic_per_member_targets,
    )
    return agent


def linear_schedule(step):
    init_value = 10.0
    end_value = 50.0
    decay_steps = 15_000


    linear_step = jnp.minimum(step, decay_steps)
    decayed_value = init_value + (end_value - init_value) * (linear_step / decay_steps)
    return decayed_value
    
def make_batch_augmentation_func(image_keys) -> callable:

    def data_augmentation_fn(rng, observations):
        for pixel_key in image_keys:
            observations = observations.copy(
                add_or_replace={
                    pixel_key: batched_random_crop(
                        observations[pixel_key], rng, padding=4, num_batch_dims=2
                    )
                }
            )
        return observations
    
    def augment_batch(batch: Batch, rng: PRNGKey) -> Batch:
        rng, obs_rng, next_obs_rng = jax.random.split(rng, 3)
        obs = data_augmentation_fn(obs_rng, batch["observations"])
        next_obs = data_augmentation_fn(next_obs_rng, batch["next_observations"])
        batch = batch.copy(
            add_or_replace={
                "observations": obs,
                "next_observations": next_obs,
            }
        )
        return batch
    
    return augment_batch


def make_trainer_config(port_number: int = None, broadcast_port: int = None):
    """agentlace ports of the learner/actor pair: 5588 / 5589 by default.

    AQI_TRAINER_PORT=<p> moves them to p / p+1, so several learner/actor pairs
    can share one machine (e.g. separate Slurm jobs on one 4-GPU node). Both
    processes of a pair read the same variable, so they always agree."""
    base = int(os.environ.get("AQI_TRAINER_PORT", "0") or 0)
    if port_number is None:
        port_number = base if base > 0 else 5588
    if broadcast_port is None:
        broadcast_port = port_number + 1
    return TrainerConfig(
        port_number=port_number,
        broadcast_port=broadcast_port,
        request_types=["send-stats"],
    )


def make_wandb_logger(
    project: str = "hil-serl",
    description: str = "serl_launcher",
    debug: bool = False,
    output_dir: str = None,
    group: str = None,
):
    wandb_config = WandBLogger.get_default_config()
    wandb_config.update(
        {
            "project": project,
            "exp_descriptor": description,
            # a list: wandb.init(tags=...) iterates its argument, so a bare
            # string became one tag per character ("s", "i", "m", "_", ...)
            "tag": [description],
            "group": group,
        }
    )
    wandb_logger = WandBLogger(
        wandb_config=wandb_config,
        variant={},
        wandb_output_dir=output_dir,
        debug=debug,
    )
    return wandb_logger
