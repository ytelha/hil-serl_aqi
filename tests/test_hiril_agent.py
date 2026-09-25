"""Tests for the HiRIL agent and replay buffer (serl_launcher/hiril/agent.py,
serl_launcher/data/data_store.py::RiskAwarePERDataStore).

Needs the training environment (jax, flax, optax, gymnasium, agentlace).
Run from the repo root with
    python -m pytest tests/test_hiril_agent.py -q
    python tests/test_hiril_agent.py
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "serl_launcher"))

import flax.linen as nn                                    # noqa: E402
import gymnasium as gym                                    # noqa: E402
import jax                                                 # noqa: E402
import jax.numpy as jnp                                    # noqa: E402
from flax.core.frozen_dict import freeze                   # noqa: E402

from serl_launcher.common.encoding import EncodingWrapper  # noqa: E402
from serl_launcher.hiril.agent import (                    # noqa: E402
    BayesianDense, HiRILAgent, dual_uncertainty_stats, gaussian_kl,
    make_hiril_defs, quantile_huber_loss, split_mean)
from serl_launcher.hiril.config import HiRILConfig         # noqa: E402

B, A = 16, 3


# ---------------------------------------------------------------------------
# pure functions
# ---------------------------------------------------------------------------
def test_bayesian_dense_samples_and_mean():
    x = jnp.asarray(np.random.default_rng(0).normal(size=(3, 5, 2)),
                    jnp.float32)
    layer = BayesianDense(4, rho_init=0.0)          # large std for the test
    params = layer.init({"params": jax.random.PRNGKey(0)}, x, False)
    p = params["params"]
    mean_out = layer.apply(params, x, False)
    np.testing.assert_allclose(
        mean_out, x @ p["kernel_mu"] + p["bias_mu"], rtol=1e-5, atol=1e-6)
    s1 = layer.apply(params, x, True, rngs={"weights": jax.random.PRNGKey(1)})
    s2 = layer.apply(params, x, True, rngs={"weights": jax.random.PRNGKey(1)})
    assert s1.shape == (3, 5, 4)
    np.testing.assert_allclose(s1, s2)                  # same rng -> same draw
    assert not np.allclose(s1[0], s1[1])               # M slices differ


def test_dual_uncertainty_is_total_variance():
    z = np.random.default_rng(1).normal(size=(6, 4, 5)) * \
        np.arange(1, 5)[None, :, None]
    st = dual_uncertainty_stats(jnp.asarray(z), sigma_z=0.0)
    np.testing.assert_allclose(st["dual"], z.reshape(6, 4, 5).transpose(
        1, 0, 2).reshape(4, -1).var(axis=1), rtol=1e-5)
    np.testing.assert_allclose(st["dual"], st["aleatoric"] + st["epistemic"],
                               rtol=1e-5)
    np.testing.assert_allclose(st["q"], z.mean(axis=(0, 2)), rtol=1e-5)
    st2 = dual_uncertainty_stats(jnp.asarray(z), sigma_z=0.5)
    np.testing.assert_allclose(st2["dual"] - st["dual"], 0.5, rtol=1e-5)


def test_quantile_huber_matches_numpy():
    r = np.random.default_rng(2)
    z = r.normal(size=(2, 3, 4)).astype(np.float32)
    tgt = r.normal(size=(2, 3, 5)).astype(np.float32)
    taus = r.uniform(size=(3, 4)).astype(np.float32)
    kappa = 1.0
    ref = np.zeros(3)
    for m in range(2):
        for b in range(3):
            tot = 0.0
            for i in range(4):
                for j in range(5):
                    d = tgt[m, b, j] - z[m, b, i]
                    h = 0.5 * d * d if abs(d) <= kappa else \
                        kappa * (abs(d) - 0.5 * kappa)
                    tot += abs(taus[b, i] - (d < 0)) * h / kappa
            ref[b] += tot / 5 / 2              # mean over j, mean over m
    np.testing.assert_allclose(
        quantile_huber_loss(jnp.asarray(z), jnp.asarray(tgt),
                            jnp.asarray(taus), kappa), ref, rtol=1e-5)


def test_split_mean_and_kl():
    v = jnp.asarray([1.0, 2.0, 3.0, 4.0])
    h = jnp.asarray([0.0, 0.0, 1.0, 1.0])
    a, b = split_mean(v, h)
    assert float(a) == 1.5 and float(b) == 3.5
    a, b = split_mean(v, jnp.zeros(4))
    assert float(a) == 2.5 and float(b) == 0.0
    params = {"l": {"kernel_mu": jnp.zeros(3),
                    "kernel_rho": jnp.full(3, np.log(np.e - 1.0))}}  # sd = 1
    assert abs(float(gaussian_kl(params, 1.0))) < 1e-6


# ---------------------------------------------------------------------------
# agent
# ---------------------------------------------------------------------------
class _TinyImageEncoder(nn.Module):
    """(..., H, W, C) -> (..., 8); handles unbatched images like the real
    encoder (sample_actions is called with ONE observation)."""
    @nn.compact
    def __call__(self, x, train=False, encode=True):
        return nn.Dense(8)(x.reshape(x.shape[:-3] + (-1,)))


def _obs(r, n=None):
    shape = () if n is None else (n,)
    return {"image": r.uniform(0, 1, shape + (1, 4, 4, 3)).astype(np.float32),
            "state": r.normal(size=shape + (1, 5)).astype(np.float32)}


def _agent(**cfg):
    hiril = HiRILConfig(k_tau=4, k_tau_target=4, k_theta=3, du_k_tau=4,
                        du_k_theta=3, critic_hidden_dims=(32, 32),
                        actor_hidden_dims=(32, 32), n_cos=8, **cfg)
    enc = EncodingWrapper(encoder={"image": _TinyImageEncoder()},
                          use_proprio=True, enable_stacking=True,
                          image_keys=("image",))
    actor_def, critic_def = make_hiril_defs(A, hiril, encoder=enc)
    r = np.random.default_rng(0)
    return HiRILAgent.create(jax.random.PRNGKey(0), _obs(r),
                             np.zeros(A, np.float32), actor_def=actor_def,
                             critic_def=critic_def, hiril=hiril,
                             discount=0.97, image_keys=("image",))


def _batch(seed, human=None):
    r = np.random.default_rng(seed)
    h = (np.arange(B) % 2).astype(np.float32) if human is None else \
        np.full(B, human, np.float32)
    return freeze(jax.tree_util.tree_map(jnp.asarray, {
        "observations": _obs(r, B), "next_observations": _obs(r, B),
        "actions": r.uniform(-1, 1, (B, A)).astype(np.float32),
        "rewards": (r.random(B) < 0.2).astype(np.float32),
        "masks": np.ones(B, np.float32), "dones": np.zeros(B, bool),
        "hiril_human": h, "hiril_bonus": h * 2.0,
        "per_index": np.arange(B, dtype=np.int32),
        "is_weight": np.ones(B, np.float32)}))


def _leaves(tree):
    return [np.asarray(x) for x in jax.tree_util.tree_leaves(tree)]


def test_td3_delayed_update():
    agent = _agent()
    batch = _batch(1)
    a1, info = agent.update(batch, networks_to_update=frozenset({"critic"}))
    c = info["critic"]
    assert np.isfinite(float(c["critic_loss"]))
    assert np.asarray(c["td_abs"]).shape == (B,)
    assert info["actor"] == {}
    # critic-only step: actor params and ALL target params untouched
    for x, y in zip(_leaves(agent.state.params["modules_actor"]["network"]),
                    _leaves(a1.state.params["modules_actor"]["network"])):
        np.testing.assert_array_equal(x, y)
    for x, y in zip(_leaves(agent.state.target_params),
                    _leaves(a1.state.target_params)):
        np.testing.assert_array_equal(x, y)
    a2, info2 = a1.update(batch)                 # critic + actor
    assert np.isfinite(float(info2["actor"]["actor_loss"]))
    assert any(not np.array_equal(x, y) for x, y in zip(
        _leaves(a1.state.params["modules_actor"]["network"]),
        _leaves(a2.state.params["modules_actor"]["network"])))
    assert any(not np.array_equal(x, y) for x, y in zip(
        _leaves(a1.state.target_params), _leaves(a2.state.target_params)))


def test_losses_split_rl_and_human():
    agent = _agent(lambda_bc=2.0)
    key = jax.random.PRNGKey(3)
    rl_only = _batch(2, human=0.0)
    loss, info = agent.critic_loss_fn(rl_only, agent.state.params, key)
    assert float(info["critic_loss_human"]) == 0.0
    np.testing.assert_allclose(float(loss), float(info["critic_loss_rl"]),
                               rtol=1e-6)
    human_only = _batch(2, human=1.0)
    loss, info = agent.actor_loss_fn(human_only, agent.state.params, key)
    a_pi = agent.forward_policy(human_only["observations"])
    bc = float(jnp.mean(jnp.sum((a_pi - human_only["actions"]) ** 2, -1)))
    np.testing.assert_allclose(float(loss), 2.0 * bc, rtol=1e-5)   # no Q term


def test_acting_and_dual_uncertainty():
    agent = _agent()
    r = np.random.default_rng(4)
    obs = _obs(r)
    det = agent.sample_actions(obs, argmax=True)
    np.testing.assert_allclose(det, agent.forward_policy(
        jax.tree_util.tree_map(lambda x: x[None], obs))[0], rtol=1e-5)
    noisy = agent.sample_actions(obs, seed=jax.random.PRNGKey(0),
                                 noise_std=0.5)
    assert np.all(np.abs(np.asarray(noisy)) <= 1.0)
    assert not np.allclose(noisy, det)
    batch = _batch(5)
    du = agent.dual_uncertainty(batch["observations"], batch["actions"],
                                jax.random.PRNGKey(1))
    for k in ("du", "dual", "aleatoric", "epistemic", "q"):
        assert np.asarray(du[k]).shape == (B,)
        assert np.all(np.isfinite(np.asarray(du[k])))
    np.testing.assert_allclose(du["du"], du["dual"])
    np.testing.assert_allclose(du["dual"], du["aleatoric"] + du["epistemic"],
                               rtol=1e-4, atol=1e-7)
    alea = _agent(du_mode="aleatoric").dual_uncertainty(
        batch["observations"], batch["actions"], jax.random.PRNGKey(1))
    np.testing.assert_allclose(alea["du"], alea["aleatoric"])


# ---------------------------------------------------------------------------
# replay buffer
# ---------------------------------------------------------------------------
def test_risk_aware_per_datastore():
    from serl_launcher.data.data_store import RiskAwarePERDataStore
    obs_space = gym.spaces.Dict({
        "image": gym.spaces.Box(0, 255, (1, 4, 4, 3), np.uint8),
        "state": gym.spaces.Box(-np.inf, np.inf, (1, 5), np.float32)})
    act_space = gym.spaces.Box(-1, 1, (A,), np.float32)
    buf = RiskAwarePERDataStore(obs_space, act_space, capacity=200,
                                image_keys=("image",))
    r = np.random.default_rng(0)
    for t in range(100):
        human = t % 4 == 0
        buf.insert(dict(
            observations={"image": r.integers(0, 255, (1, 4, 4, 3), np.uint8),
                          "state": r.normal(size=(1, 5)).astype(np.float32)},
            next_observations={
                "image": r.integers(0, 255, (1, 4, 4, 3), np.uint8),
                "state": r.normal(size=(1, 5)).astype(np.float32)},
            actions=r.uniform(-1, 1, A).astype(np.float32),
            rewards=np.float32(0.0), masks=1.0, dones=(t % 25 == 24),
            hiril_human=np.float32(human),
            hiril_bonus=np.float32(20.0 if human else 0.0)))
    batch = buf.sample(batch_size=400, pack_obs_and_next_obs=True)
    assert batch["per_index"].dtype == np.int32
    w = np.asarray(batch["is_weight"])
    assert w.shape == (400,) and w.max() == 1.0 and w.min() > 0.0
    frac_human = float(np.mean(batch["hiril_human"]))
    assert frac_human > 0.6, frac_human       # 25% of data, heavy RD bonus
    st = buf.priority_stats()
    assert 0.2 < st["human_frac"] < 0.3 and st["human_bonus_mean"] == 20.0
    buf.update_priorities(np.asarray(batch["per_index"])[:5], np.full(5, 7.0))
    assert buf.priority_stats()["max_abs_td"] == 7.0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
