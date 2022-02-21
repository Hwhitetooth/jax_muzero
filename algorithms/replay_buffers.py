import numpy as np
import snappy


from algorithms import utils
from algorithms.types import ActorOutput


class UniformBuffer(object):
    def __init__(self, min_size, max_size, compress=False):
        self._min_size = min_size
        self._max_size = max_size
        self._buf = [None for _ in range(max_size)]
        self._n = 0
        self._idx = 0
        if compress:
            self._encode = lambda ts: ts._replace(observation=compress_array(ts.observation))
            self._decode = lambda ts: ts._replace(observation=uncompress_array(ts.observation))
        else:
            self._encode = lambda ts: ts
            self._decode = lambda ts: ts

    def append(self, trajectory: ActorOutput):
        if self._n < self._max_size:
            self._n += 1
        self._buf[self._idx] = self._encode(trajectory)
        self._idx = (self._idx + 1) % self._max_size

    def extend(self, trajectories: ActorOutput):
        unpacked_trajectories = utils.unpack_namedtuple_np(trajectories)
        for trajectory in unpacked_trajectories:
            self.append(trajectory)
        return len(unpacked_trajectories)

    def _get(self, indices):
        batch = [self._decode(self._buf[i]) for i in indices]
        return utils.pack_namedtuple_np(batch, axis=0)

    def sample(self, batch_size):
        if batch_size > self._n:
            return None
        batch_indices = np.random.choice(self._n, batch_size, replace=False)
        return self._get(batch_indices)

    def full(self):
        return self._n == self._max_size

    def ready(self):
        return self._n >= self._min_size

    @property
    def size(self):
        return self._n


def compress_array(array: np.ndarray):
    """Compresses a numpy array with snappy."""
    return snappy.compress(array), array.shape, array.dtype


def uncompress_array(compressed) -> np.ndarray:
    """Uncompresses a numpy array with snappy given its shape and dtype."""
    compressed_array, shape, dtype = compressed
    byte_string = snappy.uncompress(compressed_array)
    return np.frombuffer(byte_string, dtype=dtype).reshape(shape)
