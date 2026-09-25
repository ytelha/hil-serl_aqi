import collections
from typing import Any, Iterator, Optional, Sequence, Tuple, Union

import gymnasium as gym
import jax
import numpy as np
from serl_launcher.data.dataset import Dataset, DatasetDict


def _init_replay_dict(
    obs_space: gym.Space, capacity: int
) -> Union[np.ndarray, DatasetDict]:
    if isinstance(obs_space, gym.spaces.Box):
        return np.empty((capacity, *obs_space.shape), dtype=obs_space.dtype)
    elif isinstance(obs_space, gym.spaces.Dict):
        data_dict = {}
        for k, v in obs_space.spaces.items():
            data_dict[k] = _init_replay_dict(v, capacity)
        return data_dict
    else:
        raise TypeError()


def _insert_recursively(
    dataset_dict: DatasetDict, data_dict: DatasetDict, insert_index: int
):
    if isinstance(dataset_dict, np.ndarray):
        dataset_dict[insert_index] = data_dict
    elif isinstance(dataset_dict, dict):
        for k in dataset_dict.keys():
            _insert_recursively(dataset_dict[k], data_dict[k], insert_index)
    else:
        raise TypeError()


class ReplayBuffer(Dataset):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        next_observation_space: Optional[gym.Space] = None,
        include_next_actions: Optional[bool] = False,
        include_label: Optional[bool] = False,
        include_grasp_penalty: Optional[bool] = False,
        include_aqi_fields: Optional[bool] = False,
        bootstrap_ensemble_size: int = 0,
        bootstrap_p: float = 1.0,
    ):
        if next_observation_space is None:
            next_observation_space = observation_space

        observation_data = _init_replay_dict(observation_space, capacity)
        next_observation_data = _init_replay_dict(next_observation_space, capacity)
        dataset_dict = dict(
            observations=observation_data,
            next_observations=next_observation_data,
            actions=np.empty((capacity, *action_space.shape), dtype=action_space.dtype),
            rewards=np.empty((capacity,), dtype=np.float32),
            masks=np.empty((capacity,), dtype=np.float32),
            dones=np.empty((capacity,), dtype=bool),
        )

        if include_next_actions:
            dataset_dict['next_actions'] = np.empty((capacity, *action_space.shape), dtype=action_space.dtype)
            dataset_dict['next_intvn'] = np.empty((capacity,), dtype=bool)
            
        if include_label:
            dataset_dict['labels'] = np.empty((capacity,), dtype=int)
        
        if include_grasp_penalty:
            dataset_dict['grasp_penalty'] = np.empty((capacity,), dtype=np.float32)

        if bootstrap_ensemble_size and bootstrap_p < 1.0:
            # Bootstrapped ensemble training: a PERSISTENT Bernoulli(p)
            # mask per (transition, ensemble member), drawn once at
            # insertion and stored with the transition, so each critic
            # member trains on its own fixed subsample of the data
            # (Osband-style bootstrap; resampling per update would only
            # add loss noise, not data diversity). At least one member is
            # always active per transition.
            dataset_dict['ensemble_mask'] = np.empty(
                (capacity, bootstrap_ensemble_size), dtype=np.float32)

        if include_aqi_fields:
            # AQI-HIL-SERL (proposal S5.5/S5.6): per-transition intervention
            # quality q_int, imitation weight w_i, and a mask marking
            # intervention transitions. Autonomous transitions carry zeros.
            dataset_dict['q_int'] = np.empty((capacity,), dtype=np.float32)
            dataset_dict['quality_weight'] = np.empty((capacity,), dtype=np.float32)
            dataset_dict['intervention_mask'] = np.empty((capacity,), dtype=np.float32)

        super().__init__(dataset_dict)

        self._bootstrap_K = (bootstrap_ensemble_size
                             if bootstrap_p < 1.0 else 0)
        self._bootstrap_p = float(bootstrap_p)
        self._bootstrap_rng = np.random.default_rng()
        self._size = 0
        self._capacity = capacity
        self._insert_index = 0

    def __len__(self) -> int:
        return self._size

    def _draw_bootstrap_mask(self) -> np.ndarray:
        m = (self._bootstrap_rng.random(self._bootstrap_K)
             < self._bootstrap_p).astype(np.float32)
        if not m.any():  # guarantee the transition trains someone
            m[self._bootstrap_rng.integers(self._bootstrap_K)] = 1.0
        return m

    def insert(self, data_dict: DatasetDict):
        if self._bootstrap_K and "ensemble_mask" not in data_dict:
            data_dict = {**data_dict,
                         "ensemble_mask": self._draw_bootstrap_mask()}
        _insert_recursively(self.dataset_dict, data_dict, self._insert_index)

        self._insert_index = (self._insert_index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def get_iterator(self, queue_size: int = 2, sample_args: dict = {}, device=None):
        # See https://flax.readthedocs.io/en/latest/_modules/flax/jax_utils.html#prefetch_to_device
        # queue_size = 2 should be ok for one GPU.
        queue = collections.deque()

        def enqueue(n):
            for _ in range(n):
                data = self.sample(**sample_args)
                queue.append(jax.device_put(data, device=device))

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)

    def download(self, from_idx: int, to_idx: int):
        indices = np.arange(from_idx, to_idx)
        data_dict = self.sample(batch_size=len(indices), indx=indices)
        return to_idx, data_dict

    def get_download_iterator(self):
        last_idx = 0
        while True:
            if last_idx >= self._size:
                raise RuntimeError(f"last_idx {last_idx} >= self._size {self._size}")
            last_idx, batch = self.download(last_idx, self._size)
            yield batch
