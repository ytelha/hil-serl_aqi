"""HIL-SERL / AQI-HIL-SERL experiment config for the MuJoCo peg-insertion task.

This mirrors the real `ram_insertion` task (fixed-gripper, vision-based
insertion) but targets the MuJoCo `SimPegEnv` instead of the Franka hardware.
Two differences from the real tasks:

  * No reward classifier -- the simulator provides ground-truth success, so we
    use `SimPegRewardWrapper` instead of `MultiCameraBinaryRewardClassifierWrapper`.
  * Training uses two wrist cameras ("wrist_1", "wrist_2", one on each side of
    the peg), rendered by MuJoCo, so the standard pixel-based DrQ/RLPD agent and
    the AQI extension run unchanged. A third "front" scene camera exists for
    visualization/video only and is not a policy input.

Run everything through the usual scripts with `--exp_name=sim_peg_insertion`.
"""

import os
import numpy as np

from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
    GripperCloseEnv,
)
from franka_env.envs.relative_env import RelativeFrame
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper

from experiments.config import DefaultTrainingConfig
from experiments.sim_peg_insertion.wrapper import SimPegEnv, SimPegRewardWrapper


class TrainConfig(DefaultTrainingConfig):
    # Two wrist cameras are the policy's image inputs (one on each side of the
    # peg). The "front" scene camera exists in the XML for visualization/video
    # only and is intentionally NOT trained on (add it here later if desired).
    image_keys = ["wrist_1", "wrist_2"]
    classifier_keys = ["wrist_1", "wrist_2"]  # unused in sim (ground-truth reward)
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    buffer_period = 1000
    # Save a policy checkpoint every 1000 learner steps (was 5000) so far less
    # progress is lost if training is interrupted; aligns with buffer_period.
    checkpoint_period = 1000
    steps_per_update = 50
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-fixed-gripper"
    # Socket/peg geometry: "cylinder" (easy round peg, no yaw alignment, 8 mm
    # clearance -- recommended starting point), "cuboid" (square peg, needs yaw
    # alignment to within +/-45 deg, 5 mm clearance -- intermediate) or "hex"
    # (the exact reference hexagonal task: tight, requires xy + yaw alignment).
    # All are kept; switch by setting this.
    peg_shape = "cylinder"
    # Lateral success tolerance (m): tip-to-bore-axis error below which the peg
    # counts as inserted. The round bore physically allows ~8 mm of play, so a
    # value below that forces the policy to actively CENTRE the peg (graded
    # difficulty) instead of just dropping it in -- this is how the otherwise
    # saturated cylinder task gets head-room for an operator-dependence signal.
    # Override per-run with the SIM_SUCCESS_LATERAL env var (metres).
    success_lateral = 0.012
    # ~30 s per episode at 10 Hz control so a human teleoperator has time to
    # align and insert before the episode times out and resets.
    max_traj_length = 300
    random_steps = 200

    def get_aqi_config(self, mode: str = "aqi"):
        from serl_launcher.aqi.config import AQIConfig
        # NOTE: MuJoCo wrist force/torque magnitudes are not calibrated to real
        # newtons. Run `--aqi_mode=logged` first and set force_warn/force_hard
        # from the logged f_norm distribution (warn ~= 90th percentile of
        # nominal contact, hard ~= just under what jams the peg) before enabling
        # the active gate. These are placeholder starting values.
        return AQIConfig(
            mode=mode,
            force_warn=15.0,
            force_hard=30.0,
            torque_warn=1.5,
            torque_hard=3.0,
            fail_horizon=20,
            query_warmup_steps=500,  # ~5 episodes at 100 steps
        )

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        success_lateral = float(
            os.environ.get("SIM_SUCCESS_LATERAL", self.success_lateral))
        sim_base = SimPegEnv(
            fake_env=fake_env,
            save_video=save_video,
            image_obs=True,
            image_keys=tuple(self.image_keys),
            max_episode_length=self.max_traj_length,
            peg_shape=self.peg_shape,
            success_lateral=success_lateral,
        )
        env = GripperCloseEnv(sim_base)
        disable_spacemouse = os.environ.get("SIM_DISABLE_SPACEMOUSE", "0") == "1"
        if not fake_env and not disable_spacemouse:
            env = SpacemouseIntervention(env)
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        # Ground-truth sparse reward from the simulator (no classifier needed).
        env = SimPegRewardWrapper(env)
        # Synthetic-operator support (doc Part A/S6): attach the FULL geometric
        # operator's pieces so the actor's synthetic operator both TAKES OVER on
        # true misalignment/stagnation/contact (monitoring) AND corrects with a
        # skill-dependent motor model (execution). One SimPegPhaseOracle serves
        # both roles; operator_action_scale converts mm/s -> env units.
        # Harmless for the learner (fake_env), which never runs an operator.
        from experiments.sim_peg_insertion.wrapper import (
            SimPegPhaseOracle, VerticalSignConditioner,
            make_synthetic_monitor)
        oracle = SimPegPhaseOracle(sim_base, self.peg_shape)
        env.synthetic_oracle = oracle          # execution oracle (S4 opt A)
        env.monitor_oracle = oracle            # same geometry for the monitor
        env.synthetic_monitor = make_synthetic_monitor(
            oracle, self.max_traj_length)
        env.operator_action_scale = 0.01       # 1 action unit ~= 100 mm/s
        env.operator_action_conditioner = VerticalSignConditioner(sim_base)
        return env
