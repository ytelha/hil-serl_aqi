"""Tests for the counterfactual-score diagnostics (quality_cf_diag): they
must be logging-only, off by default, and compute what they claim.

Run from the repo root with
    python -m pytest tests/test_cf_diag.py -q
    python tests/test_cf_diag.py
Two tests need real JAX / gymnasium and skip themselves without them.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parent.parent
for _p in ("serl_launcher", "examples"):
    sys.path.insert(0, str(REPO / _p))


def test_off_by_default_and_flag_only_when_set():
    from serl_launcher.aqi.config import AQIConfig
    assert AQIConfig().quality_cf_diag is False
    import run_campaign as rc
    assert "--quality_cf_diag" in rc._aqi_flags(
        SimpleNamespace(quality_cf_diag=True))
    assert not any("cf_diag" in f for f in rc._aqi_flags(SimpleNamespace()))


def test_note_ground_truth_only_when_enabled():
    from serl_launcher.aqi.runtime import AQIRuntime
    gt = {"gt_lat_err_mm": 3.5, "gt_height_mm": 1.0, "gt_jammed": True,
          "z": 0.2}
    for on in (False, True):
        stub = SimpleNamespace(cfg=SimpleNamespace(quality_cf_diag=on),
                               _pending_step_record={"step": 4})
        AQIRuntime.note_ground_truth(stub, dict(gt, z_task=0.4))
        rec = stub._pending_step_record
        assert ("gt_lat_err_mm" in rec) == on
        assert rec.get("op_z_task") == (0.4 if on else None)
        assert "z" not in rec                     # only the listed fields
    stub = SimpleNamespace(cfg=SimpleNamespace(quality_cf_diag=True),
                           _pending_step_record=None)
    AQIRuntime.note_ground_truth(stub, gt)        # nothing pending: no-op


def test_wrapper_reports_ground_truth_only_when_asked():
    import gymnasium
    if not hasattr(gymnasium, "make"):
        print("SKIP (no gymnasium)")
        return
    from serl_launcher.aqi.operator_gate import (MonitoringSignal,
                                                 SyntheticOperatorWrapper)
    from serl_launcher.aqi.operator_model import (SyntheticOperator,
                                                  make_operator_theta)
    from serl_launcher.aqi.toy_insertion import (ToyInsertionEnv,
                                                 ToyInsertionOracle)
    for on in (False, True):
        orc = ToyInsertionOracle()
        op = SyntheticOperator(orc, make_operator_theta("intermediate"),
                               rng=np.random.default_rng(0))
        env = SyntheticOperatorWrapper(
            ToyInsertionEnv(seed=0), op, MonitoringSignal(orc),
            rng=np.random.default_rng(1), log_ground_truth=on)
        env.reset()
        obs, _r, _te, _tr, info = env.step(np.zeros(2))
        syn = info["synthetic_operator"]
        assert ("gt_lat_err_mm" in syn) == on
        if on:      # the toy's hole is at x = 0: the error is |x| after the step
            assert abs(syn["gt_lat_err_mm"] - abs(obs["state"][0])) < 1e-9
            assert syn["gt_height_mm"] == float(obs["state"][1])


def test_counterfactual_diagnostics_values():
    import jax
    if not hasattr(jax, "random"):
        print("SKIP (no jax)")
        return
    import flax
    import jax.numpy as jnp
    from serl_launcher.aqi.uncertainty import counterfactual_diagnostics

    class FakeAgent(flax.struct.PyTreeNode):
        """Q_k(s, a) = s0 - ||a - w_k||^2; policy mode 0, samples N(0, 0.1)."""
        w: jnp.ndarray                                    # (K, A)

        def forward_critic(self, obs, actions, rng=None, train=False):
            s0 = obs["state"][:, 0, 0]
            return s0[None, :] - jnp.sum(
                (actions[None] - self.w[:, None, :]) ** 2, axis=-1)

        def sample_actions(self, observations, seed=None, argmax=False):
            mu = jnp.zeros((observations["state"].shape[0], self.w.shape[1]))
            return mu if argmax else mu + 0.1 * jax.random.normal(seed, mu.shape)

    K, A, B, N = 4, 3, 45, 6              # B > max_batch: chunking + padding
    agent = FakeAgent(w=jnp.linspace(-0.2, 0.2, K * A).reshape(K, A))
    rng = np.random.default_rng(0)
    obs = {"state": jnp.asarray(rng.normal(size=(B, 1, 5)), jnp.float32)}
    a_h = rng.uniform(-1, 1, (B, A))
    a_r = rng.uniform(-1, 1, (B, A))
    d = counterfactual_diagnostics(agent, obs, a_h, a_r, seed=3,
                                   n_samples=N, max_batch=16)
    assert d["qh"].shape == (K, B) and d["qr"].shape == (K, B)
    assert d["qmu"].shape == (K, B) and d["vk"].shape == (K, B)
    assert d["qs"].shape == (N, B) and d["a_mu"].shape == (B, A)
    s0 = np.asarray(obs["state"])[:, 0, 0]
    w = np.asarray(agent.w)
    q = lambda a: s0[None] - ((a[None] - w[:, None]) ** 2).sum(-1)
    assert np.allclose(d["qh"], q(a_h), atol=1e-4)
    assert np.allclose(d["qr"], q(a_r), atol=1e-4)
    assert np.allclose(d["qmu"], q(np.zeros((B, A))), atol=1e-4)
    assert np.allclose(d["a_mu"], 0.0)
    # vk (mean over samples per member) and qs (mean over members per
    # sample) average the same N x K values
    assert np.allclose(d["vk"].mean(0), d["qs"].mean(0), atol=1e-4)
    assert counterfactual_diagnostics(agent, obs, a_h[:0], a_r[:0], 0) == {}


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
