"""Critic-ensemble uncertainty estimation (S5.3) and counterfactual
advantage (S5.5, Option B).

All functions treat the agent (a `flax.struct.PyTreeNode`) as a pytree
argument, so they can be jitted once and reused as parameters are updated
from the learner. They work with `SACAgent` and both hybrid variants, all of
which expose `forward_critic` on the *continuous* critic ensemble named
"critic" with output shape (K, B).

Definitions implemented here:
    mu_Q(s,a)    = (1/K) sum_i Q_i(s,a)
    sigma_Q(s,a) = std_i Q_i(s,a)
    U_Q          = sigma_Q                          (epistemic proxy, S5.3)
    LCB_Q        = mu_Q - beta * sigma_Q            (pessimistic value, S5.3)
    A_hat(s,a_H) = mu_Q(s,a_H) - mu_Q(s,a_robot)    (counterfactual adv, S5.5)
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def _q_stats_batched(agent, observations, actions):
    """mu_Q, sigma_Q for a batch. observations: pytree with leading batch
    axis; actions: (B, A). Returns (mu, sigma), each (B,)."""
    qs = agent.forward_critic(observations, actions, rng=None, train=False)
    # qs: (critic_ensemble_size, B)
    mu = qs.mean(axis=0)
    sigma = qs.std(axis=0)
    return mu, sigma


@jax.jit
def _q_stats_pair_batched(agent, observations, actions_a, actions_b):
    """Ensemble stats for two candidate actions at the same states.

    Used for the counterfactual advantage: a = a_H (human), b = a_robot.
    Returns (mu_a, sigma_a, mu_b, sigma_b), each (B,).
    """
    qs_a = agent.forward_critic(observations, actions_a, rng=None, train=False)
    qs_b = agent.forward_critic(observations, actions_b, rng=None, train=False)
    return qs_a.mean(0), qs_a.std(0), qs_b.mean(0), qs_b.std(0)


def _add_batch_dim(tree):
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x)[None], tree)


def q_stats_single(agent, obs, action):
    """mu_Q, sigma_Q for a single (unbatched) observation/action.

    `obs` is the exact observation the actor feeds to `sample_actions`
    (e.g. images with a chunk axis); a batch axis is added here.
    Returns two python floats.
    """
    obs_b = _add_batch_dim(obs)
    act_b = jnp.asarray(action)[None]
    mu, sigma = _q_stats_batched(agent, obs_b, act_b)
    return float(np.asarray(mu)[0]), float(np.asarray(sigma)[0])


def lcb(mu: float, sigma: float, beta: float) -> float:
    """LCB_Q = mu_Q - beta * sigma_Q (S5.3)."""
    return mu - beta * sigma


def counterfactual_advantage_batch(agent, observations, human_actions,
                                   robot_actions, max_batch=32):
    """Batched A_hat and sigma for logged intervention steps (S5.5).

    Args:
        observations: pytree with leading batch axis B (stacked obs at the
            intervention states).
        human_actions: (B, A) operator corrective actions a_H.
        robot_actions: (B, A) the policy's own actions a_robot sampled at
            the same states during the rollout.
        max_batch: cap on the per-forward batch. A long intervention
            segment is scored in fixed-size chunks and concatenated, which
            is numerically identical to one B-wide pass (the ensemble
            forward is per-row at train=False) but bounds the transient GPU
            allocation so a low-skill operator's long takeover cannot OOM
            the actor's memory slice.

    Returns:
        a_hat:   (B,) np.ndarray, mu_Q(s,a_H) - mu_Q(s,a_robot)
        sigma:   (B,) np.ndarray, ensemble std at (s, a_H) -- the gate input
        mu_h:    (B,) np.ndarray, mu_Q(s, a_H)
        mu_r:    (B,) np.ndarray, mu_Q(s, a_robot)
    """
    human_actions = jnp.asarray(human_actions)
    robot_actions = jnp.asarray(robot_actions)
    n = int(human_actions.shape[0])
    if n == 0:
        z = np.zeros((0,), dtype=np.float64)
        return z, z, z, z
    mu_h_parts, sig_h_parts, mu_r_parts = [], [], []
    for start in range(0, n, max_batch):
        stop = min(start + max_batch, n)
        m = stop - start
        obs_c = jax.tree_util.tree_map(
            lambda x, a=start, b=stop: x[a:b], observations)
        a_h = human_actions[start:stop]
        a_r = robot_actions[start:stop]
        if m < max_batch:
            # pad up to one fixed shape so the jit compiles once; the padded
            # rows are sliced back off and never influence the real rows.
            pad = max_batch - m
            obs_c = jax.tree_util.tree_map(
                lambda x: jnp.concatenate(
                    [x, jnp.repeat(x[-1:], pad, axis=0)], axis=0), obs_c)
            a_h = jnp.concatenate([a_h, jnp.repeat(a_h[-1:], pad, axis=0)], 0)
            a_r = jnp.concatenate([a_r, jnp.repeat(a_r[-1:], pad, axis=0)], 0)
        c_mu_h, c_sig_h, c_mu_r, _c_sig_r = _q_stats_pair_batched(
            agent, obs_c, a_h, a_r)
        mu_h_parts.append(np.asarray(c_mu_h)[:m])
        sig_h_parts.append(np.asarray(c_sig_h)[:m])
        mu_r_parts.append(np.asarray(c_mu_r)[:m])
    mu_h = np.concatenate(mu_h_parts)
    mu_r = np.concatenate(mu_r_parts)
    sigma = np.concatenate(sig_h_parts)
    return mu_h - mu_r, sigma, mu_h, mu_r


def counterfactual_advantage_batch_safe(agent, observations, human_actions,
                                        robot_actions, max_batch=32):
    """OOM-resilient counterfactual scoring; returns identical values.

    On a GPU out-of-memory it retries with a smaller batch and, as a last
    resort, reruns on CPU, so one long intervention segment can never abort
    the actor mid-run. Non-OOM errors are re-raised unchanged.
    """
    try:
        return counterfactual_advantage_batch(
            agent, observations, human_actions, robot_actions, max_batch)
    except jax.errors.JaxRuntimeError as err:
        msg = str(err).lower()
        if "resource_exhausted" not in msg and "out of memory" not in msg:
            raise
        if max_batch > 4:
            return counterfactual_advantage_batch_safe(
                agent, observations, human_actions, robot_actions,
                max_batch // 2)
        cpu = jax.devices("cpu")[0]
        with jax.default_device(cpu):
            return counterfactual_advantage_batch(
                jax.device_put(agent, cpu),
                jax.device_put(observations, cpu),
                human_actions, robot_actions, max_batch)


def stack_obs(obs_list):
    """Stack a list of single observations (pytrees) into a batched pytree."""
    return jax.tree_util.tree_map(
        lambda *xs: jnp.stack([jnp.asarray(x) for x in xs], axis=0), *obs_list
    )
