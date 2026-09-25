"""E2HiL comparison baseline: entropy-guided sample selection for the actor.

Port of E2HiL (H. Deng et al., "E2HiL: Entropy-Guided Sample Selection for
Efficient Real-World Human-in-the-Loop Reinforcement Learning", IEEE RA-L
2026, arXiv:2601.19969) onto this repo's SACAgent. The reference is the
authors' own JAX HIL-SERL fork, github.com/E2HiL/E2HiL-project-a1x,
examples/cov_actor_loss.py (cov_actor_loss_fn / _compute_cov_per_state).
The math reproduces it step for step:

  for every batch STATE s (the stored action is not used: the influence is a
  property of the state under the current policy):
    a_k ~ pi(.|s), k = 1..K, with log-probs lp_k
    g_k      = Q(s, a_k) - alpha * lp_k          Q = min (or mean) over critics
    A_soft_k = g_k - mean_k g_k                  V(s) = MC mean        (Eq. 7)
    y_k      = min(exp(lp_k), 1e6) * A_soft_k    pi * A_soft           (Eq. 8)
    c(s)     = -Cov_k(lp_k, y_k)                 population covariance (Eq. 9-10)
  I(s)    = 1[ P_low(|c|) <= |c(s)| <= P_high(|c|) ]  batch percentiles (Eq. 11)
  L_actor = -sum_s I(s) l(s) / (sum_s I(s) + eps),  l = Q - alpha log pi (Eq. 12)

The paper's learning rate eta in c = -eta * Cov is a positive constant that
cancels in the percentile mask, so it is dropped (as in the official code).
c is stop-gradient: it only selects which states enter the actor update.
The critic and temperature losses are unchanged.

One implementation difference from the official code, numerically
identical: the K critic evaluations run as a sequential lax.map over k
instead of a single (B*K)-row forward, so peak GPU memory stays at one
batch-sized critic pass (the official version repeats every image K times).
"""

from typing import Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp

# Transition labels (official convention): 0 = offline demo, 1 = online
# policy step, 2 = human intervention. Used for diagnostics only.
LABEL_NAMES = {0: "demo", 1: "online", 2: "intervention"}

_PI_CLIP = 1e6   # cap on pi(a|s) = exp(log pi); densities can be huge


def reduce_q(qs: jax.Array, q_agg: str) -> jax.Array:
    """Reduce critic-ensemble outputs (n_critics, B) to (B,)."""
    if q_agg == "min":
        return qs.min(axis=0)
    if q_agg == "mean":
        return qs.mean(axis=0)
    raise ValueError(f"unknown e2hil_q_agg {q_agg!r}; use 'min' or 'mean'")


def influence_per_state(
    action_dist,
    observations,
    forward_critic: Callable,
    temperature: jax.Array,
    rng: jax.Array,
    K: int,
    q_agg: str = "min",
) -> jax.Array:
    """c(s) = -Cov_k(log pi(a_k|s), pi(a_k|s) * A_soft(s, a_k)), shape (B,).

    Args:
        action_dist: the policy distribution at `observations` (batch shape B).
        forward_critic: (observations, actions, rng) -> (n_critics, B).
        temperature: SAC temperature alpha (scalar).
        K: Monte-Carlo actions per state.
    """
    sample_rng, critic_rng = jax.random.split(rng)
    a_k, lp_k = action_dist.sample_and_log_prob(
        seed=sample_rng, sample_shape=(K,))            # (K, B, A), (K, B)
    a_k = jax.lax.stop_gradient(a_k)
    lp_k = jax.lax.stop_gradient(lp_k)

    def _q(args):
        actions, key = args
        return reduce_q(forward_critic(observations, actions, key), q_agg)

    q_k = jax.lax.map(_q, (a_k, jax.random.split(critic_rng, K)))   # (K, B)
    g = q_k - temperature * lp_k
    a_soft = g - g.mean(axis=0, keepdims=True)
    y = jnp.minimum(jnp.exp(lp_k), _PI_CLIP) * a_soft
    cov = jnp.mean((lp_k - lp_k.mean(axis=0, keepdims=True))
                   * (y - y.mean(axis=0, keepdims=True)), axis=0)
    return jax.lax.stop_gradient(-cov)


def entropy_bounded_mask(
    c: jax.Array, q_low: float, q_high: float
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """I(s) = 1[|c(s)| within the batch [q_low, q_high] percentile band]."""
    c_abs = jnp.abs(c)
    lo = jnp.percentile(c_abs, 100.0 * q_low)
    hi = jnp.percentile(c_abs, 100.0 * q_high)
    mask = ((c_abs >= lo) & (c_abs <= hi)).astype(jnp.float32)
    return mask, lo, hi


def label_stats(c: jax.Array, mask: jax.Array, labels: jax.Array,
                eps: float = 1e-8) -> Dict[str, jax.Array]:
    """Per source (demo / online / intervention): count, mean |c| and the
    fraction of those states KEPT by the mask (paper Table III / Fig. 8)."""
    c_abs = jnp.abs(c)
    out = {}
    for lid, name in LABEL_NAMES.items():
        sel = (labels == lid).astype(c_abs.dtype)
        n = sel.sum()
        out[f"e2hil_{name}_count"] = n
        out[f"e2hil_{name}_abs_c_mean"] = (c_abs * sel).sum() / (n + eps)
        out[f"e2hil_{name}_keep_frac"] = (mask * sel).sum() / (n + eps)
    return out


def e2hil_actor_loss(
    actor_objective: jax.Array,
    action_dist,
    observations,
    forward_critic: Callable,
    temperature: jax.Array,
    rng: jax.Array,
    *,
    K: int = 8,
    q_low: float = 0.05,
    q_high: float = 0.90,
    q_agg: str = "min",
    labels: Optional[jax.Array] = None,
    eps: float = 1e-8,
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """Entropy-bounded actor loss (Eq. 12) and its diagnostics.

    Args:
        actor_objective: per-state l(s) = Q(s, a) - alpha * log pi(a|s) for
            the reparameterised action a ~ pi(.|s), shape (B,), carrying the
            actor gradient.
        labels: optional (B,) source labels (see LABEL_NAMES) for logging.
    """
    batch_size = actor_objective.shape[0]
    c = influence_per_state(action_dist, observations, forward_critic,
                            temperature, rng, K, q_agg)
    mask, lo, hi = entropy_bounded_mask(c, q_low, q_high)
    keep = mask.sum()
    loss = -(mask * actor_objective).sum() / (keep + eps)
    info = {
        "e2hil_keep_ratio": keep / batch_size,
        "e2hil_c_mean": c.mean(),
        "e2hil_abs_c_mean": jnp.abs(c).mean(),
        "e2hil_abs_c_lo": lo,
        "e2hil_abs_c_hi": hi,
    }
    if labels is not None:
        info.update(label_stats(c, mask, labels, eps))
    return loss, info
