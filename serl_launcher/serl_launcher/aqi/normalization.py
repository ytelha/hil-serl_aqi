"""Online normalization utilities for the active query gate (S5.2).

The acquisition function combines heterogeneous signals (Q-value spread,
probabilities, Newtons, distances), so each term is standardized with a
running z-score, z(x) = (x - mean) / std, before being weighted.
"""

import numpy as np


class RunningZScore:
    """Numerically stable Welford estimator of mean/std with z() transform."""

    def __init__(self, eps: float = 1e-6, min_count: int = 10, center: bool = True):
        self.eps = eps
        self.min_count = min_count
        self.center = center
        self.count = 0
        self._mean = 0.0
        self._m2 = 0.0

    def update(self, x: float) -> None:
        x = float(x)
        if not np.isfinite(x):
            return
        self.count += 1
        delta = x - self._mean
        self._mean += delta / self.count
        self._m2 += delta * (x - self._mean)

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def std(self) -> float:
        if self.count < 2:
            return 1.0
        return float(np.sqrt(max(self._m2 / (self.count - 1), self.eps ** 2)))

    def z(self, x: float, update: bool = True) -> float:
        """Standardize x. Returns 0 until `min_count` samples are seen."""
        if update:
            self.update(x)
        if self.count < self.min_count:
            return 0.0
        mu = self._mean if self.center else 0.0
        return float((x - mu) / (self.std + self.eps))

    def state_dict(self) -> dict:
        return {"count": self.count, "mean": self._mean, "m2": self._m2}

    def load_state_dict(self, d: dict) -> None:
        self.count = int(d["count"])
        self._mean = float(d["mean"])
        self._m2 = float(d["m2"])


class RunningVectorZScore:
    """Per-dimension running standardization for feature vectors phi(s)."""

    def __init__(self, dim: int, eps: float = 1e-6, min_count: int = 10):
        self.dim = dim
        self.eps = eps
        self.min_count = min_count
        self.count = 0
        self._mean = np.zeros(dim, dtype=np.float64)
        self._m2 = np.zeros(dim, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64).reshape(self.dim)
        if not np.all(np.isfinite(x)):
            return
        self.count += 1
        delta = x - self._mean
        self._mean += delta / self.count
        self._m2 += delta * (x - self._mean)

    @property
    def std(self) -> np.ndarray:
        if self.count < 2:
            return np.ones(self.dim)
        return np.sqrt(np.maximum(self._m2 / (self.count - 1), self.eps ** 2))

    def transform(self, x: np.ndarray, update: bool = False) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(self.dim)
        if update:
            self.update(x)
        if self.count < self.min_count:
            return x - self._mean
        return (x - self._mean) / (self.std + self.eps)
