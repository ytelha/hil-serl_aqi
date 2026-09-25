"""Tests for HiRIL's JAX-free parts: HiRILConfig and the TDRD replay
priorities (serl_launcher/hiril/{config,priorities}.py). numpy only.

Run from the repo root with
    python -m pytest tests/test_hiril_priorities.py -q
    python tests/test_hiril_priorities.py
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "serl_launcher"))

from serl_launcher.hiril.config import CLI_FIELDS, HiRILConfig  # noqa: E402
from serl_launcher.hiril.priorities import RiskAwarePriorities   # noqa: E402


def test_config_defaults_follow_the_paper():
    c = HiRILConfig()
    # Table A1 / A2
    assert (c.k_tau, c.k_theta, c.kappa) == (8, 8, 1.0)
    assert (c.critic_lr, c.actor_lr, c.policy_delay) == (5e-4, 2e-4, 2)
    assert (c.explore_init, c.explore_final) == (1.0, 0.05)
    assert (c.per_alpha, c.per_beta, c.per_eps) == (1.0, 1.0, 1e-3)
    assert c.kl_weight == 0.0 and c.du_mode == "dual"
    assert HiRILConfig.from_dict(c.as_dict()) == c
    assert set(CLI_FIELDS) <= set(c.as_dict())
    for bad in (dict(du_mode="x"), dict(k_tau=0), dict(calib_quantile=1.0),
                dict(du_k_tau=1, du_k_theta=1)):
        try:
            HiRILConfig(**bad)
        except AssertionError:
            continue
        raise AssertionError(f"accepted {bad}")


def test_new_transitions_get_max_priority_plus_bonus():
    pr = RiskAwarePriorities(10, alpha=1.0, beta=1.0, eps=1e-3)
    pr.set_new([0, 1], 0.0)                     # RL experience
    pr.set_new(2, 2.0)                          # human, RD bonus exp(.)=2
    assert np.allclose(pr.priority[:3], [1.001, 1.001, 3.001])
    pr.update([0], [5.0])                       # learner TD refresh
    assert pr.max_abs_td == 5.0
    pr.set_new(3, 0.0)                          # later arrivals start at max
    assert np.isclose(pr.priority[3], 5.001)
    assert np.isclose(pr.priority[0], 5.001)
    pr.update([1], [np.nan])                    # non-finite -> max priority
    assert np.isclose(pr.abs_td[1], 5.0)


def test_sampling_follows_priorities_and_is_weights():
    pr = RiskAwarePriorities(4, alpha=1.0, beta=1.0, eps=1e-3)
    pr.set_new([0, 1, 2, 3], 0.0)
    pr.update([0, 1, 2, 3], [0.0, 1.0, 3.0, 0.0])
    valid = np.array([True, True, True, False])  # slot 3 is padding
    p = pr.probabilities(4, valid)
    assert p[3] == 0.0 and np.isclose(p.sum(), 1.0)
    assert np.isclose(p[2] / p[1], 3.001 / 1.001)
    rng = np.random.default_rng(0)
    idx, w = pr.sample(4, 20000, rng, valid)
    assert 3 not in idx
    freq = np.bincount(idx, minlength=4) / len(idx)
    assert np.allclose(freq[:3], p[:3], atol=0.01)
    # beta = 1: w ~ 1 / P(i), normalised by the batch max
    expect = (4 * p[idx]) ** -1.0
    assert np.allclose(w, expect / expect.max(), rtol=1e-5)


def test_rd_bonus_rewards_risk_reduction():
    pr = RiskAwarePriorities(1)
    assert pr.rd_bonus(0.5, 0.2) > 1.0 > pr.rd_bonus(0.2, 0.5)
    assert np.isclose(pr.rd_bonus(0.3, 0.3), 1.0)
    assert np.isclose(pr.rd_bonus(1e6, 0.0, clip=10.0), np.exp(10.0))


def test_stats_report_human_share():
    pr = RiskAwarePriorities(4)
    pr.set_new([0, 1], 0.0)
    pr.set_new([2, 3], 3.0)
    st = pr.stats(4, human=np.array([0, 0, 1, 1], np.float32))
    assert st["human_frac"] == 0.5 and st["human_bonus_mean"] == 3.0
    assert st["human_priority_share"] > 0.5


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
