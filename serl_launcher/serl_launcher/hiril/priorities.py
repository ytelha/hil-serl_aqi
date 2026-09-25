"""HiRIL's TDRD replay priorities (paper Eq. 26-28), numpy only.

    p_J(i) = k_i / sum_j k_j                                           (26)
    k_i    = |delta_TD_i| + eps                          RL experience (27)
    k_i^H  = |delta_TD_i| + eps + exp(DU(s, a_RL) - DU(s, a_H))  human (28)

delta_TD is the critic's (mean-Q) TD error of the transition, refreshed by
the learner after every update of a batch that contained it; a new
transition starts at the largest |delta_TD| seen so far (standard PER
"max priority" so it is replayed at least once). The risk-difference (RD)
bonus exp(DU_RL - DU_H) is fixed at collection time: the actor stores it
with each human transition (0 for RL transitions). It is > 1 when the
human action was LESS risky than the policy's own action, so corrections
that reduce risk are replayed more.

Table A2 values: priority factor alpha = 1, sample factor beta = 1,
eps = 1e-3. With alpha = 1 the priority is k_i itself, as in Eq. 26.
Importance-sampling weights w_i = (N * P(i))^-beta / max_batch w correct
the critic loss for the non-uniform sampling.
"""

import numpy as np


class RiskAwarePriorities:
    """Priority bookkeeping for a fixed-capacity ring buffer.

    Slots that are not valid for sampling (e.g. frame-stack padding in
    MemoryEfficientReplayBuffer) are excluded through the ``valid`` mask
    passed to ``probabilities`` / ``sample``.
    """

    def __init__(self, capacity: int, alpha: float = 1.0, beta: float = 1.0,
                 eps: float = 1e-3):
        self.alpha, self.beta, self.eps = float(alpha), float(beta), float(eps)
        self.abs_td = np.zeros(capacity, dtype=np.float64)
        self.bonus = np.zeros(capacity, dtype=np.float64)
        self.priority = np.zeros(capacity, dtype=np.float64)
        self.max_abs_td = 1.0

    def _refresh(self, idx):
        self.priority[idx] = (self.abs_td[idx] + self.eps
                              + self.bonus[idx]) ** self.alpha

    def set_new(self, idx, bonus) -> None:
        """New transition(s) at ``idx``: max-priority TD term + RD bonus."""
        idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
        self.abs_td[idx] = self.max_abs_td
        self.bonus[idx] = np.broadcast_to(np.asarray(bonus, dtype=np.float64),
                                          idx.shape)
        self._refresh(idx)

    def update(self, idx, abs_td) -> None:
        """Refresh the TD term of sampled transitions (learner, per batch)."""
        idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
        abs_td = np.abs(np.asarray(abs_td, dtype=np.float64)).reshape(-1)
        abs_td = np.where(np.isfinite(abs_td), abs_td, self.max_abs_td)
        self.abs_td[idx] = abs_td
        if abs_td.size:
            self.max_abs_td = max(self.max_abs_td, float(abs_td.max()))
        self._refresh(idx)

    def probabilities(self, n: int, valid=None) -> np.ndarray:
        p = self.priority[:n].copy()
        if valid is not None:
            p *= np.asarray(valid[:n], dtype=np.float64)
        total = p.sum()
        if not np.isfinite(total) or total <= 0.0:
            p = (np.ones(n) if valid is None
                 else np.asarray(valid[:n], dtype=np.float64))
            total = p.sum()
        return p / total

    def sample(self, n: int, batch_size: int, rng, valid=None):
        """(indices, importance-sampling weights) for one batch."""
        prob = self.probabilities(n, valid)
        idx = rng.choice(n, size=batch_size, replace=True, p=prob)
        w = (n * np.maximum(prob[idx], 1e-12)) ** (-self.beta)
        return idx, (w / w.max()).astype(np.float32)

    def rd_bonus(self, du_rl: float, du_h: float, clip: float = 10.0) -> float:
        """Eq. 28's human term exp(DU(s, a_RL) - DU(s, a_H)), clipped."""
        return float(np.exp(np.clip(du_rl - du_h, -clip, clip)))

    def stats(self, n: int, human=None) -> dict:
        if n == 0:
            return {}
        out = {"priority_mean": float(self.priority[:n].mean()),
               "priority_max": float(self.priority[:n].max()),
               "max_abs_td": float(self.max_abs_td)}
        if human is not None:
            h = np.asarray(human[:n]) > 0.5
            out["human_frac"] = float(h.mean())
            if h.any():
                out["human_bonus_mean"] = float(self.bonus[:n][h].mean())
                out["human_priority_share"] = float(
                    self.priority[:n][h].sum() / max(self.priority[:n].sum(),
                                                     1e-12))
        return out
