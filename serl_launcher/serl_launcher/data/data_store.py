from threading import Lock
from typing import Union, Iterable

import gymnasium as gym
import jax
import numpy as np
from flax.core import frozen_dict
from serl_launcher.data.replay_buffer import ReplayBuffer
from serl_launcher.data.memory_efficient_replay_buffer import (
    MemoryEfficientReplayBuffer,
)

from agentlace.data.data_store import DataStoreBase


class ReplayBufferDataStore(ReplayBuffer, DataStoreBase):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
    ):
        ReplayBuffer.__init__(self, observation_space, action_space, capacity)
        DataStoreBase.__init__(self, capacity)
        self._lock = Lock()

    # ensure thread safety
    def insert(self, *args, **kwargs):
        with self._lock:
            super(ReplayBufferDataStore, self).insert(*args, **kwargs)

    # ensure thread safety
    def sample(self, *args, **kwargs):
        with self._lock:
            return super(ReplayBufferDataStore, self).sample(*args, **kwargs)

    # NOTE: method for DataStoreBase
    def latest_data_id(self):
        return self._insert_index

    # NOTE: method for DataStoreBase
    def get_latest_data(self, from_id: int):
        raise NotImplementedError  # TODO


class MemoryEfficientReplayBufferDataStore(MemoryEfficientReplayBuffer, DataStoreBase):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        image_keys: Iterable[str] = ("image",),
        **kwargs,
    ):
        MemoryEfficientReplayBuffer.__init__(
            self, observation_space, action_space, capacity, pixel_keys=image_keys, **kwargs
        )
        DataStoreBase.__init__(self, capacity)
        self._lock = Lock()

    # ensure thread safety
    def insert(self, *args, **kwargs):
        with self._lock:
            super(MemoryEfficientReplayBufferDataStore, self).insert(*args, **kwargs)

    # ensure thread safety
    def sample(self, *args, **kwargs):
        with self._lock:
            return super(MemoryEfficientReplayBufferDataStore, self).sample(
                *args, **kwargs
            )

    # NOTE: method for DataStoreBase
    def latest_data_id(self):
        return self._insert_index

    # NOTE: method for DataStoreBase
    def get_latest_data(self, from_id: int):
        raise NotImplementedError  # TODO


class QualityAwareReplayBufferDataStore(MemoryEfficientReplayBufferDataStore):
    """AQI-HIL-SERL quality-weighted replay (proposal S5.6).

    Priority of transition i:
        p_i = (eps + |delta_i| + eta * max(q_int_i, 0)) ** alpha
    |delta_i| is initialized optimistically to `per_delta_init` (the PER
    max-priority convention) and can be refreshed by the learner through
    update_priorities() when `use_td_priority` is enabled in AQIConfig;
    by default the priority is driven by the intervention quality q_int
    attached to each transition by the actor (S5.5, Option B).

    Transitions without AQI metadata (e.g. pre-recorded demonstrations)
    receive q_int = 0, quality_weight = demo_weight, intervention_mask =
    demo_mask, so demos keep the baseline (eps + delta_init)^alpha priority
    and, by default, do not enter the imitation loss (RLPD semantics).
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        image_keys: Iterable[str] = ("image",),
        per_eps: float = 0.05,
        per_eta: float = 1.0,
        per_alpha: float = 0.6,
        per_delta_init: float = 1.0,
        priority_sampling: bool = True,
        demo_weight: float = 0.0,
        demo_mask: float = 0.0,
        per_standardize: bool = False,
        per_scale_decay: float = 0.0,
        per_refresh_period: int = 0,
        per_is_beta: float = 0.0,
        **kwargs,
    ):
        kwargs.setdefault("include_aqi_fields", True)
        super().__init__(
            observation_space, action_space, capacity,
            image_keys=image_keys, **kwargs,
        )
        self.per_eps = float(per_eps)
        self.per_eta = float(per_eta)
        self.per_alpha = float(per_alpha)
        self.per_delta_init = float(per_delta_init)
        self.priority_sampling = bool(priority_sampling)
        self._demo_weight = float(demo_weight)
        self._demo_mask = float(demo_mask)
        self._per_standardize = bool(per_standardize)
        self._per_scale_decay = float(per_scale_decay)
        self._per_refresh_period = int(per_refresh_period)
        self._per_is_beta = float(per_is_beta)
        # standardization scale for POSITIVE q_int: cumulative mean by default,
        # or a forgetting EMA when per_scale_decay > 0 (keeps `per_eta` acting on
        # a ~unit-scale signal even as the critic drifts).
        self._q_scale_sum = 0.0
        self._q_scale_n = 0
        self._q_scale_ema = None
        self._since_refresh = 0
        self._last_sampled = None          # indices from the last sample() call
        self._priorities = np.zeros(capacity, dtype=np.float64)
        self._abs_td = np.full(capacity, self.per_delta_init, dtype=np.float64)

    @property
    def _q_scale(self) -> float:
        if self._per_scale_decay > 0.0:
            return max(self._q_scale_ema, 1e-8) if self._q_scale_ema is not None \
                else 1.0
        if self._q_scale_n == 0:
            return 1.0
        return max(self._q_scale_sum / self._q_scale_n, 1e-8)

    def _observe_q(self, q: float) -> None:
        if not self._per_standardize or q <= 0.0:
            return
        if self._per_scale_decay > 0.0:
            d = self._per_scale_decay
            self._q_scale_ema = (q if self._q_scale_ema is None
                                 else (1.0 - d) * self._q_scale_ema + d * q)
        else:
            self._q_scale_sum += q
            self._q_scale_n += 1

    # ------------------------------------------------------------------
    def _priority(self, q_int: float, abs_td: float) -> float:
        q = max(float(q_int), 0.0)
        if self._per_standardize:
            q = q / self._q_scale
        return (self.per_eps + abs_td + self.per_eta * q) ** self.per_alpha

    def _refresh_priorities(self) -> None:
        """Recompute every stored priority against the CURRENT q-scale, so old
        transitions priced under a stale EMA scale stay consistent with new
        ones (periodic priority refresh)."""
        n = len(self)
        if n == 0:
            return
        q = np.clip(self.dataset_dict["q_int"][:n].astype(np.float64), 0.0, None)
        if self._per_standardize:
            q = q / self._q_scale
        self._priorities[:n] = (self.per_eps + self._abs_td[:n]
                                + self.per_eta * q) ** self.per_alpha
        self._since_refresh = 0

    def _is_weights(self, idx: np.ndarray, n: int) -> np.ndarray:
        """Importance-sampling weights w_i = (N * P(i))^(-beta) / max_j w_j for
        the sampled indices (Schaul et al.); ones if priorities are degenerate."""
        p = self._priorities[:n] * self._is_correct_index[:n]
        total = p.sum()
        if total <= 0.0 or not np.isfinite(total):
            return np.ones(len(idx), dtype=np.float32)
        prob = np.maximum(self._priorities[idx] / total, 1e-12)
        w = (n * prob) ** (-self._per_is_beta)
        w = w / w.max()
        return w.astype(np.float32)

    def insert(self, data_dict):
        with self._lock:
            data_dict = dict(data_dict)
            data_dict.setdefault("q_int", np.float32(0.0))
            data_dict.setdefault("quality_weight",
                                 np.float32(self._demo_weight))
            data_dict.setdefault("intervention_mask",
                                 np.float32(self._demo_mask))
            self._observe_q(float(data_dict["q_int"]))
            pre = self._insert_index
            MemoryEfficientReplayBuffer.insert(self, data_dict)
            post = self._insert_index
            n_written = (post - pre) % self._capacity
            p = self._priority(data_dict["q_int"], self.per_delta_init)
            for i in range(n_written):
                idx = (pre + i) % self._capacity
                self._priorities[idx] = p
                self._abs_td[idx] = self.per_delta_init
            if self._per_refresh_period > 0:
                self._since_refresh += n_written
                if self._since_refresh >= self._per_refresh_period:
                    self._refresh_priorities()

    # called from within sample() -> already holds self._lock
    def _sample_indices(self, batch_size: int) -> np.ndarray:
        if not self.priority_sampling:
            idx = super()._sample_indices(batch_size)
            self._last_sampled = idx
            return idx
        n = len(self)
        p = self._priorities[:n] * self._is_correct_index[:n]
        total = p.sum()
        if total <= 0.0 or not np.isfinite(total):
            idx = super()._sample_indices(batch_size)
            self._last_sampled = idx
            return idx
        p = p / total
        if hasattr(self.np_random, "choice"):
            idx = self.np_random.choice(n, size=batch_size, replace=True, p=p)
        else:
            idx = np.random.choice(n, size=batch_size, replace=True, p=p)
        self._last_sampled = idx
        return idx

    def sample(self, *args, **kwargs):
        # single lock; MemoryEfficientReplayBuffer.sample (grandparent, unlocked)
        # calls our _sample_indices, which stashes the drawn indices so we can
        # attach the importance-sampling weights for the prioritised draw.
        with self._lock:
            batch = MemoryEfficientReplayBuffer.sample(self, *args, **kwargs)
            if self._per_is_beta > 0.0 and self._last_sampled is not None:
                b = batch.unfreeze()
                b["is_weight"] = self._is_weights(self._last_sampled, len(self))
                batch = frozen_dict.freeze(b)
            return batch

    # ------------------------------------------------------------------
    def update_priorities(self, indices, abs_td) -> None:
        """Optional TD refresh (learner side; AQIConfig.use_td_priority)."""
        with self._lock:
            indices = np.asarray(indices, dtype=np.int64)
            abs_td = np.abs(np.asarray(abs_td, dtype=np.float64))
            for idx, td in zip(indices, abs_td):
                if 0 <= idx < self._capacity:
                    self._abs_td[idx] = td
                    q = float(self.dataset_dict["q_int"][idx])
                    self._priorities[idx] = self._priority(q, td)

    def priority_stats(self) -> dict:
        with self._lock:
            n = len(self)
            if n == 0:
                return dict(priority_mean=0.0, priority_max=0.0,
                            q_int_mean=0.0, intervention_frac=0.0)
            return dict(
                priority_mean=float(self._priorities[:n].mean()),
                priority_max=float(self._priorities[:n].max()),
                q_int_mean=float(self.dataset_dict["q_int"][:n].mean()),
                intervention_frac=float(
                    self.dataset_dict["intervention_mask"][:n].mean()),
            )


def populate_data_store(
    data_store: DataStoreBase,
    demos_path: str,
):
    """
    Utility function to populate demonstrations data into data_store.
    :return data_store
    """
    import pickle as pkl
    import numpy as np
    from copy import deepcopy

    for demo_path in demos_path:
        with open(demo_path, "rb") as f:
            demo = pkl.load(f)
            for transition in demo:
                data_store.insert(transition)
        print(f"Loaded {len(data_store)} transitions.")
    return data_store


def populate_data_store_with_z_axis_only(
    data_store: DataStoreBase,
    demos_path: str,
):
    """
    Utility function to populate demonstrations data into data_store.
    This will remove the x and y cartesian coordinates from the state.
    :return data_store
    """
    import pickle as pkl
    import numpy as np
    from copy import deepcopy

    for demo_path in demos_path:
        with open(demo_path, "rb") as f:
            demo = pkl.load(f)
            for transition in demo:
                tmp = deepcopy(transition)
                tmp["observations"]["state"] = np.concatenate(
                    (
                        tmp["observations"]["state"][:, :4],
                        tmp["observations"]["state"][:, 6][None, ...],
                        tmp["observations"]["state"][:, 10:],
                    ),
                    axis=-1,
                )
                tmp["next_observations"]["state"] = np.concatenate(
                    (
                        tmp["next_observations"]["state"][:, :4],
                        tmp["next_observations"]["state"][:, 6][None, ...],
                        tmp["next_observations"]["state"][:, 10:],
                    ),
                    axis=-1,
                )
                data_store.insert(tmp)
        print(f"Loaded {len(data_store)} transitions.")
    return data_store
