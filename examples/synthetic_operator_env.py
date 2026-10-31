"""Attach the synthetic operator to an actor env, for pipelines other than
train_aqi_rlpd.py (currently train_hiril.py).

Same logic as train_aqi_rlpd._wrap_synthetic_operator (kept there
unchanged, so existing AQI runs are byte-identical), with its absl FLAGS
turned into arguments. The operator is the SpaceMouse-equivalent
SyntheticOperatorWrapper: during a takeover the executed action is the
operator's and info["intervene_action"] carries it.

respond_to_request=True makes the operator react to the ROBOT's request
(fed each step through env.notify_request) with its skill-modulated gate
(reaction delay, misses, false alarms) instead of watching the task itself
-- the mode for methods whose interventions are robot-triggered (AQI
active/aqi, HiRIL).
"""

import os

import numpy as np


def wrap_synthetic_operator(env, *, synthetic_operator, synthetic_theta="",
                            synthetic_seed=0, operator_id="",
                            training_oracle="auto", oracle_checkpoint="",
                            make_oracle_agent=None, mm_per_unit=100.0,
                            intervention_mode="signal",
                            latch_until_success=False,
                            respond_to_request=False, max_traj_length=300,
                            rotation_control=False, orientation_monitor=False,
                            orientation_tilt_rad=0.0, orientation_yaw_rad=0.0,
                            log_takeover_channels=False, log=print):
    """Returns (wrapped_env, theta).

    rotation_control, orientation_monitor (limits 0 = the oracle's default for
    the peg shape) and log_takeover_channels mean the same as the
    --synthetic_rotation_control, --synthetic_orientation_monitor and
    --log_takeover_channels flags of train_aqi_rlpd.py. All off: as before.

    make_oracle_agent: () -> an agent whose .state a policy-oracle
        checkpoint restores into and which has sample_actions(); needed only
        when the policy oracle is used (training_oracle "policy", or "auto"
        on an env without a geometric oracle).
    """
    from serl_launcher.aqi.operator_gate import (MonitoringSignal,
                                                 SyntheticOperatorWrapper)
    from serl_launcher.aqi.operator_model import (PhaseOracle, PolicyOracle,
                                                  SyntheticOperator,
                                                  make_operator_theta,
                                                  theta_from_json)
    rng = np.random.default_rng(synthetic_seed)
    if synthetic_theta:
        theta = theta_from_json(synthetic_theta)
        if operator_id:
            theta.operator_id = operator_id
        log(f"loaded exact operator theta from {synthetic_theta} "
            f"(tier={theta.tier}, id={theta.operator_id})")
    else:
        theta = make_operator_theta(synthetic_operator, rng=rng,
                                    operator_id=operator_id)

    exec_oracle = getattr(env, "synthetic_oracle", None) \
        or getattr(env, "phase_oracle", None) \
        or getattr(getattr(env, "unwrapped", env), "phase_oracle", None)
    if training_oracle == "geometric" and exec_oracle is None:
        raise ValueError("training_oracle=geometric but the env exposes no "
                         "geometric oracle")
    use_policy_oracle = (training_oracle == "policy"
                         or (training_oracle == "auto" and exec_oracle is None))
    action_conditioner = None
    if not use_policy_oracle:
        oracle = exec_oracle
        op_action_scale = float(getattr(env, "operator_action_scale", 1.0))
        action_conditioner = getattr(env, "operator_action_conditioner", None)
        kind = "geometric"
    else:
        if not oracle_checkpoint or make_oracle_agent is None:
            raise ValueError("the policy training oracle needs "
                             "oracle_checkpoint and make_oracle_agent")
        import jax
        from flax.training import checkpoints
        base = make_oracle_agent()
        oracle_agent = base.replace(state=checkpoints.restore_checkpoint(
            os.path.abspath(oracle_checkpoint), base.state))
        okey = jax.random.PRNGKey(synthetic_seed)

        def policy_fn(obs):
            a = oracle_agent.sample_actions(
                observations=jax.device_put(obs), argmax=True, seed=okey)
            return np.asarray(jax.device_get(a))
        oracle = PolicyOracle(policy_fn, action_dim=env.action_space.shape[0],
                              action_scale=mm_per_unit)
        op_action_scale = 1.0 / mm_per_unit
        kind = "policy"

    monitor = getattr(env, "synthetic_monitor", None)
    if monitor is None:
        monitor = MonitoringSignal(
            oracle if isinstance(oracle, PhaseOracle) else None,
            budget_steps=max_traj_length)
    op = SyntheticOperator(oracle, theta,
                           rng=np.random.default_rng(synthetic_seed + 17),
                           action_scale=op_action_scale,
                           rotation_control=rotation_control)
    if orientation_monitor:
        if not op.rotation_control:
            raise ValueError(
                "the orientation monitor needs rotation control and the "
                "geometric oracle: a takeover for a twisted tool must be able "
                "to undo the twist")
        mon_orc = getattr(monitor, "oracle", None)
        if not (hasattr(mon_orc, "tilt_yaw_error")
                and hasattr(mon_orc, "orientation_tolerance")):
            raise ValueError("this env's monitor oracle has no tilt_yaw_error "
                             "/ orientation_tolerance")
        tilt0, yaw0 = mon_orc.orientation_tolerance()
        monitor.set_orientation_limits(orientation_tilt_rad or tilt0,
                                       orientation_yaw_rad or yaw0)
    if intervention_mode == "legacy":
        intervention_mode = "always_latch" if latch_until_success else "signal"
    log(f"synthetic operator '{synthetic_operator}' (exec s={theta.s_exec:.2f}"
        f", mon s={theta.s_mon:.2f}, {kind} oracle, intervention="
        f"{intervention_mode}, responds to robot requests={respond_to_request}"
        f"{', rotation control' if op.rotation_control else ''}"
        f"{', orientation monitor' if orientation_monitor else ''})")
    wrapped = SyntheticOperatorWrapper(
        env, op, monitor, rng=np.random.default_rng(synthetic_seed + 31),
        respond_to_request=respond_to_request,
        action_conditioner=action_conditioner,
        intervention_mode=intervention_mode,
        latch_until_success=latch_until_success,
        log_takeover_channels=log_takeover_channels)
    return wrapped, theta
