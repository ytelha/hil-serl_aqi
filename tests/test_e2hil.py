"""Tests for the E2HiL comparison baseline (serl_launcher/agents/continuous/e2hil.py).

Needs the training environment (jax, flax, distrax, chex). Run from the repo
root with either
    python -m pytest tests/test_e2hil.py -q
    python tests/test_e2hil.py
"""

import os
import sys
from functools import partial

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "serl_launcher"))

import jax                                    # noqa: E402
import jax.numpy as jnp                       # noqa: E402
import flax.linen as nn                       # noqa: E402
from flax.core.frozen_dict import freeze      # noqa: E402

from serl_launcher.agents.continuous.e2hil import (   # noqa: E402
    e2hil_actor_loss, entropy_bounded_mask, influence_per_state)
from serl_launcher.agents.continuous.sac import SACAgent   # noqa: E402
from serl_launcher.common.encoding import EncodingWrapper  # noqa: E402
from serl_launcher.networks.actor_critic_nets import (      # noqa: E402
    Critic, Policy, ensemblize)
from serl_launcher.networks.lagrange import GeqLagrangeMultiplier  # noqa: E402
from serl_launcher.networks.mlp import MLP    # noqa: E402

K, B, A = 8, 64, 3


# ---------------------------------------------------------------------------
# 1. c(s) matches the official E2HiL formula on fixed inputs
# ---------------------------------------------------------------------------
class _FixedDist:
    """Stands in for pi(.|s): returns fixed MC samples regardless of seed."""

    def __init__(self, a_k, lp_k):
        self.a_k, self.lp_k = jnp.asarray(a_k), jnp.asarray(lp_k)

    def sample_and_log_prob(self, seed, sample_shape):
        assert sample_shape == (self.a_k.shape[0],)
        return self.a_k, self.lp_k


def _toy_critic(obs, actions, key):
    """Two deterministic critics, (2, B)."""
    q1 = -jnp.sum((actions - 0.3) ** 2, -1) + obs[:, 0]
    q2 = -jnp.sum((actions + 0.1) ** 2, -1) + 0.5 * obs[:, 0]
    return jnp.stack([q1, q2])


def _official_c(a_k, lp_k, obs, alpha, q_agg):
    """numpy transcription of _compute_cov_per_state in the authors'
    examples/cov_actor_loss.py (github.com/E2HiL/E2HiL-project-a1x)."""
    a_bk = np.transpose(a_k, (1, 0, 2))            # (B, K, A)
    lp_bk = np.transpose(lp_k, (1, 0))             # (B, K)
    obs_rep = np.repeat(obs, K, axis=0)            # official repeats obs K times
    qs = np.asarray(_toy_critic(jnp.asarray(obs_rep),
                                jnp.asarray(a_bk.reshape(B * K, A)), None))
    q = (qs.min(0) if q_agg == "min" else qs.mean(0)).reshape(B, K)
    g = q - alpha * lp_bk
    a_soft = g - g.mean(1, keepdims=True)
    y = np.minimum(np.exp(lp_bk), 1e6) * a_soft
    cov = np.mean((lp_bk - lp_bk.mean(1, keepdims=True))
                  * (y - y.mean(1, keepdims=True)), axis=1)
    return -cov


def test_influence_matches_official_formula():
    rng = np.random.default_rng(0)
    a_k = rng.uniform(-1, 1, (K, B, A)).astype(np.float32)
    lp_k = rng.normal(0.0, 2.0, (K, B)).astype(np.float32)
    obs = rng.normal(size=(B, 4)).astype(np.float32)
    for q_agg in ("min", "mean"):
        c = influence_per_state(_FixedDist(a_k, lp_k), jnp.asarray(obs),
                                _toy_critic, jnp.float32(0.05),
                                jax.random.PRNGKey(0), K, q_agg)
        ref = _official_c(a_k, lp_k, obs, 0.05, q_agg)
        np.testing.assert_allclose(np.asarray(c), ref, rtol=2e-4, atol=1e-5)


def test_mask_is_percentile_band():
    c = jnp.asarray(np.random.default_rng(1).normal(size=200) ** 3)
    mask, lo, hi = entropy_bounded_mask(c, 0.05, 0.90)
    ca = np.abs(np.asarray(c))
    np.testing.assert_allclose(float(lo), np.percentile(ca, 5), rtol=1e-5)
    np.testing.assert_allclose(float(hi), np.percentile(ca, 90), rtol=1e-5)
    expected = ((ca >= np.percentile(ca, 5))
                & (ca <= np.percentile(ca, 90))).astype(np.float32)
    np.testing.assert_array_equal(np.asarray(mask), expected)
    assert 0.83 <= float(mask.mean()) <= 0.87


def test_masked_loss_and_label_stats():
    rng = np.random.default_rng(2)
    a_k = rng.uniform(-1, 1, (K, B, A)).astype(np.float32)
    lp_k = rng.normal(0.0, 2.0, (K, B)).astype(np.float32)
    obs = jnp.asarray(rng.normal(size=(B, 4)).astype(np.float32))
    objective = jnp.asarray(rng.normal(size=B).astype(np.float32))
    labels = jnp.asarray(np.arange(B) % 3)
    loss, info = e2hil_actor_loss(
        objective, _FixedDist(a_k, lp_k), obs, _toy_critic, jnp.float32(0.05),
        jax.random.PRNGKey(0), K=K, labels=labels)
    c = _official_c(a_k, lp_k, obs, 0.05, "min")
    ca = np.abs(c)
    keep = (ca >= np.percentile(ca, 5)) & (ca <= np.percentile(ca, 90))
    np.testing.assert_allclose(float(loss),
                               -np.asarray(objective)[keep].mean(), rtol=1e-4)
    np.testing.assert_allclose(float(info["e2hil_keep_ratio"]), keep.mean())
    n_int = int((np.arange(B) % 3 == 2).sum())
    assert float(info["e2hil_intervention_count"]) == n_int


# ---------------------------------------------------------------------------
# 2. integration: a real (tiny) SACAgent.update with e2hil on / off
# ---------------------------------------------------------------------------
class _TinyImageEncoder(nn.Module):
    @nn.compact
    def __call__(self, x, train=False, encode=True):
        return nn.Dense(8)(x.reshape(x.shape[0], -1))


def _make_agent(**config):
    enc = EncodingWrapper(encoder={"image": _TinyImageEncoder()},
                          use_proprio=True, enable_stacking=True,
                          image_keys=("image",))
    critic_def = partial(
        Critic, encoder=enc,
        network=ensemblize(partial(MLP, hidden_dims=[32, 32],
                                   activate_final=True), 2)(
            name="critic_ensemble"))(name="critic")
    policy_def = Policy(encoder=enc,
                        network=MLP(hidden_dims=[32, 32], activate_final=True),
                        action_dim=A, tanh_squash_distribution=True,
                        std_parameterization="exp", name="actor")
    temp_def = GeqLagrangeMultiplier(init_value=1e-2, constraint_shape=(),
                                     constraint_type="geq", name="temperature")
    obs, act = _batch(0)["observations"], _batch(0)["actions"]
    return SACAgent.create(jax.random.PRNGKey(0), obs, act,
                           actor_def=policy_def, critic_def=critic_def,
                           temperature_def=temp_def, discount=0.97,
                           critic_ensemble_size=2, image_keys=("image",),
                           **config)


def _batch(seed):
    r = np.random.default_rng(seed)

    def obs():
        return {"image": r.uniform(0, 1, (B, 1, 4, 4, 3)).astype(np.float32),
                "state": r.normal(size=(B, 1, 5)).astype(np.float32)}
    return {"observations": obs(), "next_observations": obs(),
            "actions": r.uniform(-1, 1, (B, A)).astype(np.float32),
            "rewards": (r.random(B) < 0.1).astype(np.float32),
            "masks": np.ones(B, np.float32),
            "dones": np.zeros(B, bool),
            "labels": (np.arange(B) % 3).astype(np.int32)}


def _device_batch(seed):
    """As the learner feeds it: a FrozenDict of device arrays."""
    return freeze(jax.tree_util.tree_map(jnp.asarray, _batch(seed)))


E2 = dict(e2hil=True, e2hil_K=8, e2hil_q_low=0.05, e2hil_q_high=0.90,
          e2hil_q_agg="min")


def test_agent_update_runs_and_logs():
    agent = _make_agent(**E2)
    batch = _device_batch(1)
    new_agent, info = agent.update(batch)
    actor = info["actor"]
    for key in ("e2hil_keep_ratio", "e2hil_abs_c_lo", "e2hil_abs_c_hi",
                "e2hil_intervention_keep_frac", "entropy", "actor_loss"):
        assert key in actor, key
        assert np.isfinite(float(actor[key])), key
    assert 0.8 <= float(actor["e2hil_keep_ratio"]) <= 0.9
    # a second update re-uses the compiled step and changes the actor
    new_agent2, _ = new_agent.update(batch)
    before = jax.tree_util.tree_leaves(new_agent.state.params["modules_actor"])
    after = jax.tree_util.tree_leaves(new_agent2.state.params["modules_actor"])
    assert any(not np.allclose(a, b) for a, b in zip(before, after))


def test_only_actor_loss_differs_from_hilserl():
    batch = _device_batch(3)
    _, base = _make_agent().update(batch)
    _, e2 = _make_agent(**E2).update(batch)
    # critic + temperature updates are untouched by E2HiL
    np.testing.assert_allclose(float(base["critic"]["critic_loss"]),
                               float(e2["critic"]["critic_loss"]), rtol=1e-6)
    # a full band with the hilserl Q reduction keeps every state -> the
    # E2HiL loss collapses to the hilserl loss (up to its 1e-8 eps)
    _, full = _make_agent(**{**E2, "e2hil_q_low": 0.0, "e2hil_q_high": 1.0,
                             "e2hil_q_agg": "mean"}).update(batch)
    assert float(full["actor"]["e2hil_keep_ratio"]) == 1.0
    np.testing.assert_allclose(float(base["actor"]["actor_loss"]),
                               float(full["actor"]["actor_loss"]), rtol=1e-5)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
