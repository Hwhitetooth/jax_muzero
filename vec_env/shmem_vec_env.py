"""
An interface for asynchronous vectorized environments.
"""

import logging
import multiprocessing as mp
import numpy as np
from .vec_env import VecEnv, CloudpickleWrapper
import ctypes

from .util import dict_to_obs, obs_space_info, obs_to_dict

_NP_TO_CT = {np.float32: ctypes.c_float,
             np.int64: ctypes.c_int64,
             np.int32: ctypes.c_int32,
             np.int8: ctypes.c_int8,
             np.uint8: ctypes.c_char,
             np.bool: ctypes.c_bool}


def alloc_shmem(ctx, shape, dtype):
    return ctx.RawArray(_NP_TO_CT[dtype], int(np.prod(shape)))


class ShmemVecEnv(VecEnv):
    """
    Optimized version of SubprocVecEnv that uses shared variables to communicate observations.
    """

    def __init__(self, env_fns, spaces=None, context='spawn'):
        """
        If you don't specify observation_space, we'll have to create a dummy
        environment to get it.
        """
        ctx = mp.get_context(context)
        if spaces:
            observation_space, action_space = spaces
        else:
            logging.warning('Creating dummy env object to get spaces')
            dummy = env_fns[0]()
            observation_space, action_space = dummy.observation_space, dummy.action_space
            dummy.close()
            del dummy
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)
        self.obs_keys, self.obs_shapes, self.obs_dtypes = obs_space_info(observation_space)
        raw_obs_buf = {
            k: alloc_shmem(ctx, (len(env_fns),) + self.obs_shapes[k], self.obs_dtypes[k].type)
            for k in self.obs_keys
        }
        self.obs_buf = {
            k: np.frombuffer(raw_obs_buf[k], dtype=self.obs_dtypes[k].type).reshape((len(env_fns),) + self.obs_shapes[k])
            for k in self.obs_keys
        }
        raw_act_buf = alloc_shmem(ctx, (len(env_fns),) + action_space.shape, action_space.dtype.type)
        self.act_buf = np.frombuffer(raw_act_buf, dtype=action_space.dtype.type).reshape((len(env_fns),))
        raw_rew_buf = alloc_shmem(ctx, (len(env_fns),), np.float32)
        self.rew_buf = np.frombuffer(raw_rew_buf, dtype=np.float32).reshape((len(env_fns),))
        raw_done_buf = alloc_shmem(ctx, (len(env_fns),), np.bool)
        self.done_buf = np.frombuffer(raw_done_buf, dtype=np.bool).reshape((len(env_fns),))
        self.parent_pipes = []
        self.procs = []
        for idx, env_fn in enumerate(env_fns):
            wrapped_fn = CloudpickleWrapper(env_fn)
            parent_pipe, child_pipe = ctx.Pipe()
            proc = ctx.Process(
                target=_subproc_worker,
                args=(idx, child_pipe, parent_pipe, wrapped_fn,
                      raw_obs_buf, self.obs_keys, self.obs_shapes, self.obs_dtypes,
                      raw_act_buf, action_space.dtype.type,
                      raw_rew_buf, np.float32,
                      raw_done_buf, np.bool))
            proc.daemon = True
            self.procs.append(proc)
            self.parent_pipes.append(parent_pipe)
            proc.start()
            child_pipe.close()
        self.waiting_step = False
        self.viewer = None

    def reset(self):
        if self.waiting_step:
            logging.warning('Called reset() while waiting for the step to complete')
            self.step_wait()
        for pipe in self.parent_pipes:
            pipe.send('reset')
        for pipe in self.parent_pipes:
            pipe.recv()
        return self._decode_obses()

    def step_async(self, actions):
        assert len(actions) == len(self.parent_pipes)
        np.copyto(dst=self.act_buf, src=actions.astype(self.act_buf.dtype))
        for pipe, act in zip(self.parent_pipes, actions):
            pipe.send('step')
        self.waiting_step = True

    def step_wait(self):
        infos = [pipe.recv() for pipe in self.parent_pipes]
        self.waiting_step = False
        return self._decode_obses(), np.copy(self.rew_buf), np.copy(self.done_buf), infos

    def close_extras(self):
        if self.waiting_step:
            self.step_wait()
        for pipe in self.parent_pipes:
            pipe.send('close')
        for pipe in self.parent_pipes:
            pipe.recv()
            pipe.close()
        for proc in self.procs:
            proc.join()

    def get_images(self, mode='human'):
        for pipe in self.parent_pipes:
            pipe.send('render')
        return [pipe.recv() for pipe in self.parent_pipes]

    def _decode_obses(self):
        result = {}
        for k in self.obs_keys:
            result[k] = np.copy(self.obs_buf[k])
        return dict_to_obs(result)


def _subproc_worker(
        idx, pipe, parent_pipe, env_fn_wrapper,
        raw_obs_buf, obs_keys, obs_shapes, obs_dtypes,
        raw_act_buf, act_dtype,
        raw_rew_buf, rew_dtype,
        raw_done_buf, done_dtype):
    """
    Control a single environment instance using IPC and
    shared memory.
    """
    obs_buf = {
        k: np.frombuffer(raw_obs_buf[k], dtype=obs_dtypes[k].type).reshape((-1,) + obs_shapes[k])
        for k in obs_keys
    }
    act_buf = np.frombuffer(raw_act_buf, dtype=act_dtype).reshape((-1,))
    rew_buf = np.frombuffer(raw_rew_buf, dtype=rew_dtype).reshape((-1,))
    done_buf = np.frombuffer(raw_done_buf, dtype=done_dtype).reshape((-1,))

    def _write_obs(maybe_dict_obs):
        flatdict = obs_to_dict(maybe_dict_obs)
        for k in obs_buf.keys():
            np.copyto(dst=obs_buf[k][idx], src=flatdict[k])

    parent_pipe.close()
    env = env_fn_wrapper.x()
    done = True
    try:
        while True:
            cmd = pipe.recv()
            if cmd == 'reset':
                _write_obs(env.reset())
                done = False
                pipe.send(None)
            elif cmd == 'step':
                if done:
                    obs = env.reset()
                    reward = 0.
                    done = False
                    info = {}
                else:
                    obs, reward, done, info = env.step(act_buf[idx])
                _write_obs(obs)
                np.copyto(dst=rew_buf[idx:idx+1], src=reward)
                np.copyto(dst=done_buf[idx:idx+1], src=done)
                pipe.send(info)
            elif cmd == 'render':
                pipe.send(env.render(mode='rgb_array'))
            elif cmd == 'close':
                pipe.send(None)
                break
            else:
                raise RuntimeError('Got unrecognized cmd %s' % cmd)
    except KeyboardInterrupt:
        print('ShmemVecEnv worker: got KeyboardInterrupt')
    finally:
        env.close()
