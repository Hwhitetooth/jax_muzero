"""Actors for generating trajectories."""
from typing import Optional

import chex
import jax
import numpy as np

from algorithms.types import ActorOutput, Params


class Actor(object):
    def __init__(self, envs, agent):
        self._envs = envs
        self._agent_step = jax.jit(agent.batch_step)
        num_envs = self._envs.num_envs
        self._timestep = ActorOutput(
            action_tm1=np.zeros((num_envs,), dtype=np.int32),
            reward=np.zeros((num_envs,), dtype=np.float32),
            observation=self._envs.reset(),
            first=np.ones((num_envs,), dtype=np.float32),
            last=np.zeros((num_envs,), dtype=np.float32),
        )

    def initial_timestep(self):
        return self._timestep

    def step(self, rng_key: chex.PRNGKey, params: Params, random: bool, temperature: Optional[float] = None):
        if random:
            action = np.array([self._envs.action_space.sample() for _ in range(self._envs.num_envs)])
        else:
            rng_key, action, agent_out = self._agent_step(
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
        epinfos = []
        for i in info:
            maybeepinfo = i.get('episode')
            if maybeepinfo:
                epinfos.append(maybeepinfo)
        return rng_key, self._timestep, epinfos


class EvaluateActor(object):
    def __init__(self, envs, agent):
        self._envs = envs
        self._agent_step = jax.jit(agent.batch_step)

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
            rng_key, action, agent_out = self._agent_step(
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
