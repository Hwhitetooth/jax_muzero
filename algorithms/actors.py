"""Actors for generating trajectories."""
from typing import Optional

import chex
import jax
import numpy as np

from algorithms.types import ActorOutput, Params
from algorithms import utils


class Actor(object):
    def __init__(self, envs, agent, num_steps: int):
        self._envs = envs
        self._agent = agent
        self._num_steps = num_steps
        num_envs = self._envs.num_envs
        self._timestep = ActorOutput(
            action_tm1=np.zeros((num_envs,), dtype=np.int32),
            reward=np.zeros((num_envs,), dtype=np.float32),
            observation=self._envs.reset(),
            first=np.ones((num_envs,), dtype=np.float32),
            last=np.zeros((num_envs,), dtype=np.float32),
        )
        self._trajectories = [self._timestep]

    def rollout(self, rng_key: chex.PRNGKey, params: Params, random: bool, temperature: Optional[float] = None):
        epinfos = []
        while len(self._trajectories) <= self._num_steps:
            if random:
                action = np.array([self._envs.action_space.sample() for _ in range(self._envs.num_envs)])
            else:
                rng_key, action, agent_out = self._agent.step(
                    rng_key, params, jax.device_put(self._timestep), temperature, False)
                action = jax.device_get(action)
            observation, reward, done, info = self._envs.step(action)
            self._timestep = ActorOutput(
                action_tm1=action,
                reward=reward,
                observation=observation,
                first=self._timestep.last,  # If the previous timestep is the last, this is the first.
                last=done.astype(np.float32),
            )
            self._trajectories.append(self._timestep)
            for i in info:
                maybeepinfo = i.get('episode')
                if maybeepinfo:
                    epinfos.append(maybeepinfo)
            trajectories = utils.pack_namedtuple_np(self._trajectories, axis=1)
        self._trajectories = self._trajectories[1:]
        return rng_key, trajectories, epinfos


class EvaluateActor(object):
    def __init__(self, envs, agent):
        self._envs = envs
        self._agent = agent

    def evaluate(self, rng_key: chex.PRNGKey, params):
        num_envs = self._envs.num_envs
        timestep = ActorOutput(
            action_tm1=np.zeros((num_envs,), dtype=np.int32),
            reward=np.zeros((num_envs,), dtype=np.float32),
            observation=self._envs.reset(),
            first=np.ones((num_envs,), dtype=np.float32),
            last=np.zeros((num_envs,), dtype=np.float32),
        )
        epinfos = [None] * num_envs
        count = 0
        while count < num_envs:
            rng_key, action, agent_out = self._agent.step(
                rng_key, params, jax.device_put(timestep), 1., True)
            action = jax.device_get(action)
            observation, reward, done, info = self._envs.step(action)
            timestep = ActorOutput(
                action_tm1=action,
                reward=reward,
                observation=observation,
                first=timestep.last,  # If the previous timestep is the last, this is the first.
                last=done.astype(np.float32),
            )
            for k, i in enumerate(info):
                maybeepinfo = i.get('episode')
                if maybeepinfo and epinfos[k] is None:
                    epinfos[k] = maybeepinfo
                    count += 1
        return rng_key, epinfos
