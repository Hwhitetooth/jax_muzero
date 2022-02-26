from typing import Any

import numpy as np
import tree


class UniformBuffer(object):
    def __init__(self, min_size: int, max_size: int, traj_len: int):
        self._min_size = min_size
        self._max_size = max_size
        self._traj_len = traj_len
        self._timestep_storage = None
        self._n = 0
        self._idx = 0

    def extend(self, timesteps: Any):
        if self._timestep_storage is None:
            sample_timestep = tree.map_structure(lambda t: t[0], timesteps)
            self._timestep_storage = self._preallocate(sample_timestep)
        num_steps = timesteps.observation.shape[0]
        indices = np.arange(self._idx, self._idx + num_steps) % self._max_size
        tree.map_structure(lambda a, x: assign(a, indices, x), self._timestep_storage, timesteps)
        self._idx = (self._idx + num_steps) % self._max_size
        self._n = min(self._n + num_steps, self._max_size)

    def sample(self, batch_size: int):
        if batch_size + self._traj_len > self._n:
            return None
        start_indices = np.random.choice(self._n - self._traj_len, batch_size, replace=False)
        all_indices = start_indices[:, None] + np.arange(self._traj_len + 1)[None]
        base_idx = 0 if self._n < self._max_size else self._idx
        all_indices = (all_indices + base_idx) % self._max_size
        trajectories = tree.map_structure(lambda a: a[all_indices], self._timestep_storage)
        return trajectories

    def full(self):
        return self._n == self._max_size

    def ready(self):
        return self._n >= self._min_size

    @property
    def size(self):
        return self._n

    def _preallocate(self, item):
        return tree.map_structure(lambda t: np.empty((self._max_size,) + t.shape, t.dtype), item)


def assign(a, i, x):
    a[i] = x
