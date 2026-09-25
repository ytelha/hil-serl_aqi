"""Out-of-distribution / novelty trigger term (S5.4).

    OOD(s_t) = (1/k) * sum_{j=1..k} || phi(s_t) - phi(s_j) ||,
    s_j in D_success

phi(s) is a per-dimension standardized proprioceptive feature vector by
default (pose, velocity, force, torque as flattened by SERLObsWrapper).
D_success holds states from human demonstrations and from episodes that
ended in success, in a fixed-capacity ring buffer.

The implementation is dependency-free (numpy only) and computes exact k-NN
mean distance with chunking; with the default capacity (2e4) and one query
per control step this costs well under a millisecond.
"""

import numpy as np

from serl_launcher.aqi.normalization import RunningVectorZScore


class KnnOODScorer:
    def __init__(self, dim: int, k: int = 10, capacity: int = 20000,
                 min_ref: int = 200):
        self.dim = dim
        self.k = k
        self.capacity = capacity
        self.min_ref = min_ref
        self._buf = np.zeros((capacity, dim), dtype=np.float32)
        self._size = 0
        self._ptr = 0
        self._norm = RunningVectorZScore(dim)

    def __len__(self) -> int:
        return self._size

    def add(self, feats: np.ndarray) -> None:
        """Add one (dim,) or many (N, dim) reference feature vectors."""
        feats = np.asarray(feats, dtype=np.float32)
        if feats.ndim == 1:
            feats = feats[None]
        for f in feats:
            if not np.all(np.isfinite(f)):
                continue
            self._norm.update(f)
            self._buf[self._ptr] = f
            self._ptr = (self._ptr + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)

    def score(self, feat: np.ndarray) -> float:
        """Mean distance to the k nearest reference states (0 if too few)."""
        if self._size < max(self.min_ref, self.k):
            return 0.0
        feat = np.asarray(feat, dtype=np.float32).reshape(self.dim)
        ref = self._buf[: self._size]
        z_ref = (ref - self._norm._mean) / (self._norm.std + 1e-6)
        z_q = self._norm.transform(feat)
        d2 = np.sum((z_ref - z_q[None]) ** 2, axis=1)
        k = min(self.k, self._size)
        nn = np.partition(d2, k - 1)[:k]
        return float(np.mean(np.sqrt(np.maximum(nn, 0.0))))
