"""Operator-consistency weight for the weighted imitation loss (C8).

SiLRI (Zhao et al.) treats the human as a noisy policy with a state-dependent
spread sigma_beta(s) and trusts the human less where that spread is large.
This module estimates sigma_beta non-parametrically from the operator's own
earlier takeovers and turns it into a multiplicative BC weight:

    sigma_beta(s) = sqrt( mean_dims Var_j[ a_H(s_j) ] ),
                    s_j = nearest earlier takeover state of each of the k
                          earlier segments closest to s
    c(s)          = 1 / (1 + (sigma_beta(s) / sigma_ref)^2)

Design notes:
  - At most ONE neighbour per earlier segment. Consecutive takeover states
    of one trajectory are nearest neighbours of each other and their
    actions are smooth, so an unrestricted kNN would call any operator
    "consistent". The spread is therefore measured ACROSS takeovers.
  - References are only earlier segments of the same run (one run = one
    operator in every campaign), added after the episode is scored, so a
    segment is never compared with itself.
  - State distance uses the per-dimension running standardisation of the
    references (as the OOD scorer, aqi/ood.py).
  - Not ready (fewer than min_ref reference steps or fewer than k earlier
    segments): sigma_beta = NaN and c = 1, i.e. no down-weighting.

numpy only; the query cost is one (n x N) distance matrix per segment.
"""

from typing import Optional, Tuple

import numpy as np

from serl_launcher.aqi.normalization import RunningVectorZScore, RunningZScore


class OperatorConsistency:
    def __init__(self, state_dim: int, k: int = 5, capacity: int = 20000,
                 min_ref: int = 200, sigma_ref: Optional[float] = None):
        self.state_dim = int(state_dim)
        self.k = max(int(k), 1)
        self.capacity = int(capacity)
        self.min_ref = int(min_ref)
        self.sigma_ref_fixed = sigma_ref
        self._S = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self._A = None                      # allocated on the first add
        self._seg = np.full(self.capacity, -1, dtype=np.int64)
        self._size = 0
        self._ptr = 0
        self._next_seg = 0
        self._norm = RunningVectorZScore(self.state_dim)
        self._sigma_scale = RunningZScore(center=False)
        self._groups = None                 # cached per-segment ordering

    def __len__(self) -> int:
        return self._size

    @property
    def n_segments(self) -> int:
        if self._size == 0:
            return 0
        return int(np.unique(self._seg[: self._size]).size)

    @property
    def ready(self) -> bool:
        return (self._size >= max(self.min_ref, 1)
                and self.n_segments >= self.k)

    @property
    def sigma_ref(self) -> float:
        if self.sigma_ref_fixed is not None:
            return float(self.sigma_ref_fixed)
        return max(self._sigma_scale.mean, 1e-6)

    # ------------------------------------------------------------------
    def add_segment(self, states: np.ndarray, actions: np.ndarray) -> None:
        """Store one takeover's (state, a_H) pairs under a new segment id."""
        states = np.asarray(states, dtype=np.float32).reshape(-1, self.state_dim)
        actions = np.asarray(actions, dtype=np.float32).reshape(len(states), -1)
        if len(states) == 0:
            return
        if self._A is None:
            self._A = np.zeros((self.capacity, actions.shape[1]),
                               dtype=np.float32)
        sid = self._next_seg
        self._next_seg += 1
        for s, a in zip(states, actions):
            if not (np.all(np.isfinite(s)) and np.all(np.isfinite(a))):
                continue
            self._norm.update(s)
            self._S[self._ptr] = s
            self._A[self._ptr] = a
            self._seg[self._ptr] = sid
            self._ptr = (self._ptr + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)
        self._groups = None

    def _grouping(self):
        """Reference rows ordered by segment id + each segment's [start, end)."""
        if self._groups is None:
            seg = self._seg[: self._size]
            order = np.argsort(seg, kind="stable")
            seg_sorted = seg[order]
            starts = np.flatnonzero(
                np.r_[True, seg_sorted[1:] != seg_sorted[:-1]])
            ends = np.r_[starts[1:], len(seg_sorted)]
            self._groups = (order, starts, ends)
        return self._groups

    def sigma_beta(self, states: np.ndarray) -> np.ndarray:
        """sigma_beta per query state, shape (n,); NaN while not ready."""
        states = np.asarray(states, dtype=np.float64).reshape(-1, self.state_dim)
        out = np.full(len(states), np.nan)
        if len(states) == 0 or not self.ready:
            return out
        order, starts, ends = self._grouping()
        mu = self._norm._mean
        sd = self._norm.std + 1e-6
        z_ref = (self._S[: self._size][order].astype(np.float64) - mu) / sd
        acts = self._A[: self._size][order].astype(np.float64)
        z_q = (states - mu) / sd
        d2 = ((z_q ** 2).sum(1)[:, None] + (z_ref ** 2).sum(1)[None]
              - 2.0 * z_q @ z_ref.T)
        seg_min = np.minimum.reduceat(d2, starts, axis=1)       # (n, S)
        k = min(self.k, seg_min.shape[1])
        picks = np.argpartition(seg_min, k - 1, axis=1)[:, :k]
        ddof = 1 if k > 1 else 0
        for i in range(len(states)):
            rows = [starts[g] + int(np.argmin(d2[i, starts[g]:ends[g]]))
                    for g in picks[i]]
            var = np.var(acts[rows], axis=0, ddof=ddof)
            out[i] = float(np.sqrt(np.mean(var)))
        return out

    def weights(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(sigma_beta, c) for a segment's states. The running sigma_ref is
        updated with the finite sigma_beta values BEFORE c is computed."""
        sig = self.sigma_beta(states)
        finite = np.isfinite(sig)
        for s in sig[finite]:
            self._sigma_scale.update(float(s))
        c = np.ones_like(sig)
        c[finite] = 1.0 / (1.0 + (sig[finite] / self.sigma_ref) ** 2)
        return sig, c
