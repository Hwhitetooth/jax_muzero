from .vec_env import VecEnvWrapper
import numpy as np
from gym import spaces


class VecFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack):
        self.venv = venv
        self.nstack = nstack
        wos = venv.observation_space  # wrapped ob space
        low = np.repeat(wos.low, self.nstack, axis=-1)
        high = np.repeat(wos.high, self.nstack, axis=-1)
        self.num_channels = wos.shape[-1]
        self.stackedobs = np.zeros((venv.num_envs,) + low.shape, low.dtype)
        self.news = np.zeros((venv.num_envs,), dtype=np.bool)
        observation_space = spaces.Box(low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        self.stackedobs = np.roll(self.stackedobs, shift=-self.num_channels, axis=-1)
        for (i, new) in enumerate(self.news):
            if new:
                self.stackedobs[i] = 0
        obs, rews, self.news, infos = self.venv.step_wait()
        self.stackedobs[..., -self.num_channels:] = obs
        return self.stackedobs, rews, self.news, infos

    def reset(self):
        obs = self.venv.reset()
        self.stackedobs[...] = 0
        self.stackedobs[..., -self.num_channels:] = obs
        self.news = np.zeros_like(self.news)
        return self.stackedobs
