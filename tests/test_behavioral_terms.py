"""Tests for AQIConfig.quality_behavioral_terms (the behavioral score's term
ablation) in serl_launcher/aqi/quality.py.

Run from the repo root with
    python -m pytest tests/test_behavioral_terms.py -q
    python tests/test_behavioral_terms.py
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "serl_launcher"))

from serl_launcher.aqi.config import AQIConfig, BEHAVIORAL_TERMS   # noqa: E402
from serl_launcher.aqi.normalization import RunningZScore          # noqa: E402
from serl_launcher.aqi.quality import QualityScorer                # noqa: E402


def _stream(n=60, seed=0):
    r = np.random.default_rng(seed)
    return [(float(r.integers(5, 80)), float(r.normal()), float(abs(r.normal())))
            for _ in range(n)]


def test_default_is_the_original_three_term_score():
    """Default terms reproduce z(d_value) - z(t_int) - z(f_peak) exactly."""
    assert AQIConfig().quality_behavioral_terms == BEHAVIORAL_TERMS
    sc = QualityScorer(AQIConfig(mode="quality"))
    zt, zd, zf = RunningZScore(), RunningZScore(), RunningZScore()
    for t, d, f in _stream():
        ref = zd.z(d) - zt.z(t) - zf.z(f)
        assert sc.behavioral_q(t, d, f) == ref


def test_subset_drops_a_term():
    sc = QualityScorer(AQIConfig(mode="quality",
                                 quality_behavioral_terms=("d_value", "f_peak")))
    zd, zf = RunningZScore(), RunningZScore()
    for t, d, f in _stream(seed=1):
        # t_int never enters: any takeover length gives the same score
        assert sc.behavioral_q(t, d, f) == zd.z(d) - zf.z(f)


def test_invalid_terms_rejected():
    for bad in ((), ("t_int", "nonsense")):
        try:
            AQIConfig(mode="quality", quality_behavioral_terms=bad)
        except AssertionError:
            continue
        raise AssertionError(f"accepted {bad}")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
