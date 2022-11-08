"""MuZero: a MCTS agent that plans with a learned value-equivalent model."""
import logging
import time

import chex
import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from ray import tune
import rlax

from algorithms import actors
from algorithms import agents
from algorithms import replay_buffers as replay
from algorithms import utils
from algorithms.types import ActorOutput, Params
from environments import atari
import vec_env



def generate_update_fn(agent: agents.Agent, opt_update, unroll_steps: int, td_steps: int, discount_factor: float,
                       value_coef: float, policy_coef: float):
    def loss(params: Params, target_params: Params, trajectory: ActorOutput, rng_key: chex.PRNGKey):
        # 1. Make predictions. Unroll the model from the first state.
        timestep = jax.tree_map(lambda t: t[:1], trajectory)
        learner_root = agent.root_unroll(params, timestep)
        learner_root = jax.tree_map(lambda t: t[0], learner_root)

        # Fill the actions after the absorbing state with random actions.
        unroll_trajectory = jax.tree_map(lambda t: t[:unroll_steps + 1], trajectory)
        random_action_mask = jnp.cumprod(1. - unroll_trajectory.first[1:]) == 0.
        action_sequence = unroll_trajectory.action_tm1[1:]
        num_actions = learner_root.logits.shape[-1]
        random_actions = jax.random.choice(rng_key, num_actions, action_sequence.shape, replace=True)
        simulate_action_sequence = jax.lax.select(random_action_mask, random_actions, action_sequence)

        model_out = agent.model_unroll(params, learner_root.state, simulate_action_sequence)

        num_bins = learner_root.reward_logits.shape[-1]

        # 2. Construct targets.
        ## 2.1 Reward.
        rewards = trajectory.reward[1:]
        reward_target = jax.lax.select(
            random_action_mask, jnp.zeros_like(rewards[:unroll_steps]), rewards[:unroll_steps])
        reward_target = agent.value_transform(reward_target)
        reward_logits_target = utils.scalar_to_two_hot(reward_target, num_bins)

        ## 2.2 Policy
        target_roots = agent.root_unroll(target_params, trajectory)
        search_roots = jax.tree_map(lambda t: t[:unroll_steps + 1], target_roots)
        rng_key, search_key = jax.random.split(rng_key)
        search_keys = jax.random.split(search_key, search_roots.state.shape[0])
        target_trees = jax.vmap(agent.mcts, (0, None, 0, None))(search_keys, target_params, search_roots, False)
        # The target distribution always uses a temperature of 1.
        policy_target = jax.vmap(agent.act_prob, (0, None))(target_trees.visit_count[:, 0], 1.)
        # Set the policy targets for the absorbing state and the states after to uniform random.
        uniform_policy = jnp.ones_like(policy_target) / num_actions
        random_policy_mask = jnp.cumprod(1. - unroll_trajectory.last) == 0.
        random_policy_mask = jnp.broadcast_to(random_policy_mask[:, None], policy_target.shape)
        policy_target = jax.lax.select(random_policy_mask, uniform_policy, policy_target)
        policy_target = jax.lax.stop_gradient(policy_target)

        ## 2.3 Value
        discounts = (1. - trajectory.last[1:]) * discount_factor

        def n_step_return(i):
            # According to the EfficientZero source code, it is unnecessary to use the search value for bootstrapping.
            # See: https://github.com/YeWR/EfficientZero/blob/main/main.py#L41
            #   and https://github.com/YeWR/EfficientZero/blob/main/core/reanalyze_worker.py#L325
            bootstrap_value = jax.tree_map(lambda t: t[i + td_steps], target_roots.value)
            _rewards = jnp.concatenate([rewards[i:i + td_steps], bootstrap_value[None]], axis=0)
            _discounts = jnp.concatenate([jnp.ones((1,)), jnp.cumprod(discounts[i:i + td_steps])], axis=0)
            return jnp.sum(_rewards * _discounts)

        returns = []
        for i in range(unroll_steps + 1):
            returns.append(n_step_return(i))
        returns = jnp.stack(returns)
        # Set the value targets for the absorbing state and the states after to 0.
        zero_return_mask = jnp.cumprod(1. - unroll_trajectory.last) == 0.
        value_target = jax.lax.select(zero_return_mask, jnp.zeros_like(returns), returns)
        value_target = agent.value_transform(value_target)
        value_logits_target = utils.scalar_to_two_hot(value_target, num_bins)
        value_logits_target = jax.lax.stop_gradient(value_logits_target)

        # 3. Compute the losses.
        _batch_categorical_cross_entropy = jax.vmap(rlax.categorical_cross_entropy)
        reward_loss = jnp.mean(_batch_categorical_cross_entropy(reward_logits_target, model_out.reward_logits))
        value_logits = jnp.concatenate([learner_root.value_logits[None], model_out.value_logits], axis=0)
        value_loss = jnp.mean(_batch_categorical_cross_entropy(value_logits_target, value_logits))
        logits = jnp.concatenate([learner_root.logits[None], model_out.logits], axis=0)
        policy_loss = jnp.mean(_batch_categorical_cross_entropy(policy_target, logits))
        total_loss = reward_loss + value_coef * value_loss + policy_coef * policy_loss
        policy_target_entropy = jax.vmap(lambda p: distrax.Categorical(probs=p).entropy())(policy_target)
        log = {
            'reward_target': reward_target,
            'reward_prediction': model_out.reward,
            'value_target': value_target,
            'value_prediction': utils.logits_to_scalar(value_logits),
            'policy_entropy': -rlax.entropy_loss(logits, jnp.ones(logits.shape[:-1])),
            'policy_target_entropy': policy_target_entropy,
            'reward_loss': reward_loss,
            'value_loss': value_loss,
            'policy_loss': policy_loss,
            'total_loss': total_loss,
        }
        return total_loss, log

    def batch_loss(params: Params, target_params: Params, trajectories: ActorOutput, rng_key: chex.PRNGKey):
        batch_size = trajectories.observation.shape[0]
        rng_keys = jax.random.split(rng_key, batch_size)
        losses, log = jax.vmap(loss, (None, None, 0, 0))(params, target_params, trajectories, rng_keys)
        log.update({
            'reward_target_mean': jnp.mean(log['reward_target']),
            'reward_target_std': jnp.std(log['reward_target']),
            'reward_prediction_mean': jnp.mean(log['reward_prediction']),
            'reward_prediction_std': jnp.std(log['reward_prediction']),
            'value_target_mean': jnp.mean(log['value_target']),
            'value_target_std': jnp.std(log['value_target']),
            'value_prediction_mean': jnp.mean(log['value_prediction']),
            'value_prediction_std': jnp.std(log['value_prediction']),
            'policy_entropy': jnp.mean(log['policy_entropy']),
            'policy_target_entropy': jnp.mean(log['policy_target_entropy']),
            'reward_loss': jnp.mean(log['reward_loss']),
            'value_loss': jnp.mean(log['value_loss']),
            'policy_loss': jnp.mean(log['policy_loss']),
            'total_loss': jnp.mean(log['total_loss']),
        })
        log.pop('reward_target')
        log.pop('reward_prediction')
        log.pop('value_target')
        log.pop('value_prediction')
        return jnp.mean(losses), log

    def update(rng_key: chex.PRNGKey, params: Params, target_params: Params, opt_state, trajectories: ActorOutput):
        grads, log = jax.grad(batch_loss, has_aux=True)(params, target_params, trajectories, rng_key)
        grads = jax.lax.pmean(grads, axis_name='i')
        updates, opt_state = opt_update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        log.update({
            'grad_norm': optax.global_norm(grads),
            'update_norm': optax.global_norm(updates),
            'param_norm': optax.global_norm(params),
        })
        return params, opt_state, log

    return update


class Experiment(tune.Trainable):
    def setup(self, config):
        self._config = config
        platform = jax.lib.xla_bridge.get_backend().platform
        self._num_devices = jax.lib.xla_bridge.device_count()
        logging.warning("Running on %s %s(s)", self._num_devices, platform)

        seed = config['seed']
        env_id = config['env_id'] + 'NoFrameskip-v4'
        self._envs = atari.make_vec_env(
            env_id,
            num_env=config['num_envs'],
            seed=seed,
            env_kwargs=config['env_kwargs'],
        )
        self._envs = vec_env.VecFrameStack(self._envs, 4)
        self._evaluate_envs = atari.make_vec_env(
            env_id=env_id,
            num_env=config['evaluate_episodes'],
            seed=config['seed'],
            env_kwargs=config['env_kwargs'],
            wrapper_kwargs={'episode_life': False},
        )
        self._evaluate_envs = vec_env.VecFrameStack(self._evaluate_envs, 4)
        self._agent = agents.Agent(
            self._envs.observation_space,
            self._envs.action_space,
            num_bins=config['num_bins'],
            channels=config['channels'],
            use_v2=config['use_resnet_v2'],
            output_init_scale=config['output_init_scale'],
            discount_factor=config['discount_factor'],
            num_simulations=config['num_simulations'],
            max_search_depth=config['max_search_depth'],
            mcts_c1=config['mcts_c1'],
            mcts_c2=config['mcts_c2'],
            alpha=config['alpha'],
            exploration_prob=config['exploration_prob'],
            q_normalize_epsilon=config['q_normalize_epsilon'],
            child_select_epsilon=config['child_select_epsilon'],
        )
        self._actor = actors.Actor(self._envs, self._agent)
        self._evaluate_actor = actors.EvaluateActor(self._evaluate_envs, self._agent)

        if config['temperature_scheduling'] == 'staircase':
            def temperature_fn(num_frames: int):
                frac = num_frames / config['total_frames']
                if frac < 0.5:
                    return 1.
                elif frac < 0.75:
                    return 0.5
                else:
                    return 0.25
        elif config['temperature_scheduling'] == 'constant':
            def temperature_fn(num_frames: int):
                return 1.
        else:
            raise KeyError

        self._temperature_fn = temperature_fn

        self._rng_key = jax.random.PRNGKey(seed)
        self._rng_key, init_key = jax.random.split(self._rng_key)
        self._params = self._agent.init(init_key)
        self._target_params = self._params
        self._target_update_interval = config['target_update_interval']

        # Only apply weight decay to the weights in Dense layers and Conv layers.
        # Do NOT apply to the biases and the scales and offsets in normalization layers.
        weight_decay_mask = Params(
            encoder=hk.data_structures.map(
                lambda module_name, name, value: True if name == 'w' else False, self._params.encoder),
            transition=hk.data_structures.map(
                lambda module_name, name, value: True if name == 'w' else False, self._params.transition),
            prediction=hk.data_structures.map(
                lambda module_name, name, value: True if name == 'w' else False, self._params.prediction),
        )
        learning_rate = optax.warmup_exponential_decay_schedule(
            init_value=0.,
            peak_value=config['learning_rate'],
            warmup_steps=config['warmup_steps'],
            transition_steps=100_000,
            decay_rate=config['learning_rate_decay'],
            staircase=True,
        )
        # Apply the decoupled weight decay. Ref: https://arxiv.org/abs/1711.05101.
        self._opt = optax.adamw(
            learning_rate=learning_rate,
            weight_decay=config['weight_decay'],
            mask=weight_decay_mask,
        )
        if config['max_grad_norm']:
            self._opt = optax.chain(
                optax.clip_by_global_norm(config['max_grad_norm']),
                self._opt,
            )
        self._opt_state = self._opt.init(self._params)

        self._params = jax.device_put_replicated(self._params, jax.local_devices())
        self._target_params = self._params
        self._opt_state = jax.device_put_replicated(self._opt_state, jax.local_devices())

        self._update_fn = generate_update_fn(
            self._agent,
            self._opt.update,
            unroll_steps=config['unroll_steps'],
            td_steps=config['td_steps'],
            discount_factor=config['discount_factor'],
            value_coef=config['value_coef'],
            policy_coef=config['policy_coef'],
        )
        self._update_fn = jax.pmap(self._update_fn, axis_name='i')

        self._replay_buffer = replay.UniformBuffer(
            min_size=config['replay_min_size'],
            max_size=config['replay_max_size'],
            traj_len=config['unroll_steps'] + config['td_steps'],
        )
        self._batch_size = config['batch_size']
        assert self._batch_size % self._num_devices == 0

        self._log_interval = config['log_interval']
        self._num_frames = 0
        self._total_frames = config['total_frames']
        self._num_updates = 0

        init_timestep = self._actor.initial_timestep()
        self._replay_buffer.extend(init_timestep)
        self._num_frames += init_timestep.observation.shape[0]
        act_params = jax.tree_map(lambda t: t[0], self._params)
        while not self._replay_buffer.ready():
            self._rng_key, timesteps, epinfos = self._actor.step(self._rng_key, act_params, random=True)
            self._replay_buffer.extend(timesteps)
            self._num_frames += timesteps.observation.shape[0]
        self._trajectories = [
            self._replay_buffer.sample(self._batch_size // self._num_devices)
            for _ in range(self._num_devices)
        ]
        self._trajectories = jax.device_put_sharded(self._trajectories, jax.local_devices())

    def step(self):
        t0 = time.time()
        for _ in range(self._log_interval):
            # There are essentially 3 operations in each iteration: sampling, training, and acting.
            # The typical order of execution is: acting -> sampling -> training. But this order does not allow any
            # parallelization. Here we use a different order: training -> sampling -> acting. Due to the asynchronous
            # dispatching of JAX, the call to self._update_fn returns immediately before the computation completes.
            # This enables overlap between training, which only needs the GPU, and sampling, which only needs the CPU.
            # Acting needs both the CPU, the GPU, and synchronization between these two, so it is like a barrier and
            # cannot overlap with either of the other two operations.
            self._rng_key, update_key = jax.random.split(self._rng_key)
            update_keys = jax.random.split(update_key, self._num_devices)
            self._params, self._opt_state, log = self._update_fn(
                update_keys, self._params, self._target_params, self._opt_state, self._trajectories)
            self._num_updates += 1
            if self._num_updates % self._target_update_interval == 0:
                self._target_params = self._params

            self._trajectories = [
                self._replay_buffer.sample(self._batch_size // self._num_devices)
                for _ in range(self._num_devices)
            ]
            self._trajectories = jax.device_put_sharded(self._trajectories, jax.local_devices())

            if self._num_frames < self._total_frames:
                act_params = jax.tree_map(lambda t: t[0], self._params)
                temperature = self._temperature_fn(self._num_frames)
                self._rng_key, timesteps, epinfos = self._actor.step(
                    self._rng_key, act_params, random=False, temperature=temperature)
                self._replay_buffer.extend(timesteps)
                self._num_frames += timesteps.observation.shape[0]

        act_params = jax.tree_map(lambda t: t[0], self._params)
        self._rng_key, epinfos = self._evaluate_actor.evaluate(self._rng_key, act_params)

        log = jax.tree_map(lambda t: t[0], log)
        log = jax.device_get(log)
        log.update({
            'ups': self._log_interval / (time.time() - t0),
            'num_frames': self._num_frames,
            'num_updates': self._num_updates,
            'episode_return': np.mean([epinfo['r'] for epinfo in epinfos]),
            'episode_length': np.mean([epinfo['l'] for epinfo in epinfos]),
        })
        return log

    def cleanup(self):
        self._envs.close()
        self._evaluate_envs.close()


if __name__ == '__main__':
    config = {
        'env_id': 'Qbert',
        'env_kwargs': {},
        'seed': 42,
        'num_envs': 1,
        'unroll_steps': 5,
        'td_steps': 5,
        'max_search_depth': None,

        'channels': 64,
        'num_bins': 601,
        'use_resnet_v2': True,
        'output_init_scale': 0.,
        'discount_factor': 0.997 ** 4,
        'mcts_c1': 1.25,
        'mcts_c2': 19625,
        'alpha': 0.3,
        'exploration_prob': 0.25,
        'temperature_scheduling': 'staircase',
        'q_normalize_epsilon': 0.01,
        'child_select_epsilon': 1E-6,
        'num_simulations': 50,

        'replay_min_size': 2_000,
        'replay_max_size': 100_000,
        'batch_size': 256,

        'value_coef': 0.25,
        'policy_coef': 1.,
        'max_grad_norm': 5.,
        'learning_rate': 7E-4,
        'warmup_steps': 1_000,
        'learning_rate_decay': 0.1,
        'weight_decay': 1E-4,
        'target_update_interval': 200,

        'evaluate_episodes': 32,
        'log_interval': 4_000,
        'total_frames': 100_000,
    }
    analysis = tune.run(
        Experiment,
        config=config,
        stop={
            'num_updates': 120_000,
        },
        resources_per_trial={
            'gpu': 4,
        },
    )
