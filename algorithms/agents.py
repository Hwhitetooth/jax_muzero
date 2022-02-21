import chex
import gym
import haiku as hk
import jax
import jax.numpy as jnp
import rlax

from algorithms import haiku_nets as nets
from algorithms import utils
from algorithms.types import ActorOutput, AgentOutput, Params, Tree


class Agent(object):
    """A MCTS agent."""

    def __init__(self, observation_space: gym.spaces.Box, action_space: gym.spaces.Discrete, num_bins: int,
                 channels: int, use_v2: bool, output_init_scale: float, discount_factor: float, num_simulations: int,
                 max_search_depth: int, mcts_c1: float, mcts_c2: float, alpha: float, exploration_prob: float,
                 q_normalize_epsilon: float, child_select_epsilon: float):
        self._observation_space = observation_space
        self._action_space = action_space
        self._discount_factor = discount_factor
        self._num_simulations = num_simulations
        self._max_search_depth = num_simulations if max_search_depth is None else max_search_depth
        self._mcts_c1 = mcts_c1
        self._mcts_c2 = mcts_c2
        self._alpha = alpha
        self._exploration_prob = exploration_prob
        self._q_normalize_epsilon = q_normalize_epsilon
        self._child_select_epsilon = child_select_epsilon
        self.value_transform = utils.value_transform
        self.inv_value_transform = utils.inv_value_transform
        self._encode_fn = hk.without_apply_rng(hk.transform(
            lambda observations: nets.EZStateEncoder(channels, use_v2)(observations)))
        num_actions = self._action_space.n
        self._predict_fn = hk.without_apply_rng(hk.transform(
            lambda states: nets.EZPrediction(num_actions, num_bins, output_init_scale, use_v2)(states)))
        self._transit_fn = hk.without_apply_rng(hk.transform(
            lambda action, state: nets.EZTransition(use_v2)(action, state)))
        self.step = jax.jit(self._batch_step)

    def init(self, rng_key: chex.PRNGKey):
        encoder_key, prediction_key, transition_key = jax.random.split(rng_key, 3)
        dummy_observation = self._observation_space.sample()
        encoder_params = self._encode_fn.init(encoder_key, dummy_observation)
        dummy_state = self._encode_fn.apply(encoder_params, dummy_observation)
        prediction_params = self._predict_fn.init(prediction_key, dummy_state)
        dummy_action = jnp.zeros((self._action_space.n,))
        transition_params = self._transit_fn.init(transition_key, dummy_action, dummy_state)
        params = Params(encoder=encoder_params, prediction=prediction_params, transition=transition_params)
        return params

    def _batch_step(self, rng_key: chex.PRNGKey, params: Params, timesteps: ActorOutput, temperature: float,
                    is_eval: bool):
        batch_size = timesteps.reward.shape[0]
        rng_key, step_key = jax.random.split(rng_key)
        step_keys = jax.random.split(step_key, batch_size)
        batch_root_step = jax.vmap(self._root_step, (0, None, 0, None, None))
        actions, agent_out = batch_root_step(step_keys, params, timesteps, temperature, is_eval)
        return rng_key, actions, agent_out

    def _root_step(self, rng_key: chex.PRNGKey, params: Params, timesteps: ActorOutput, temperature: float,
                   is_eval: bool):
        """The input `timesteps` is assumed to be [input_dim]."""
        trajectories = jax.tree_map(lambda t: t[None], timesteps)  # Add a dummy time dimension.
        agent_out = self.root_unroll(params, trajectories)
        agent_out = jax.tree_map(lambda t: t.squeeze(axis=0), agent_out)  # Squeeze the dummy time dimension.
        search_key, sample_key, greedy_key = jax.random.split(rng_key, 3)
        tree = self.mcts(search_key, params, agent_out, is_eval)
        act_prob = self.act_prob(tree.visit_count[0], temperature)
        sampled_action = rlax.categorical_sample(sample_key, act_prob)
        greedy_actions = (tree.visit_count[0] == tree.visit_count[0].max()).astype(jnp.float32)
        greedy_prob = greedy_actions / greedy_actions.sum()
        greedy_action = rlax.categorical_sample(greedy_key, greedy_prob)
        # Choose the greedy action during evaluation.
        action = jax.lax.select(is_eval, greedy_action, sampled_action)
        return action, agent_out

    def root_unroll(self, params: Params, trajectory: ActorOutput):
        """The input `trajectory` is assumed to be [T, input_dim]."""
        state = self._encode_fn.apply(params.encoder, trajectory.observation)  # [T, S]
        logits, reward_logits, value_logits = self._predict_fn.apply(params.prediction, state)
        reward = utils.logits_to_scalar(reward_logits)
        reward = self.inv_value_transform(reward)
        value = utils.logits_to_scalar(value_logits)
        value = self.inv_value_transform(value)
        return AgentOutput(
            state=state,
            logits=logits,
            reward_logits=reward_logits,
            reward=reward,
            value_logits=value_logits,
            value=value,
        )

    def model_step(self, params: Params, state: chex.Array, action: chex.Array):
        """The input `state` and `action` are assumed to be [S] and []."""
        one_hot_action = hk.one_hot(action, self._action_space.n)
        next_state = self._transit_fn.apply(params.transition, one_hot_action, state)
        next_state = utils.scale_gradient(next_state, 0.5)
        logits, reward_logits, value_logits = self._predict_fn.apply(params.prediction, next_state)
        reward = utils.logits_to_scalar(reward_logits)
        reward = self.inv_value_transform(reward)
        value = utils.logits_to_scalar(value_logits)
        value = self.inv_value_transform(value)
        return AgentOutput(
            state=next_state,
            logits=logits,
            reward_logits=reward_logits,
            reward=reward,
            value_logits=value_logits,
            value=value,
        )

    def model_unroll(self, params: Params, state: chex.Array, action_sequence: chex.Array):
        """The input `state` and `action` are assumed to be [S] and [T]."""
        def fn(state: chex.Array, action: chex.Array):
            one_hot_action = hk.one_hot(action, self._action_space.n)
            next_state = self._transit_fn.apply(params.transition, one_hot_action, state)
            next_state = utils.scale_gradient(next_state, 0.5)
            return next_state, next_state

        _, state_sequence = jax.lax.scan(fn, state, action_sequence)
        logits, reward_logits, value_logits = self._predict_fn.apply(params.prediction, state_sequence)
        reward = utils.logits_to_scalar(reward_logits)
        reward = self.inv_value_transform(reward)
        value = utils.logits_to_scalar(value_logits)
        value = self.inv_value_transform(value)
        return AgentOutput(
            state=state_sequence,
            logits=logits,
            reward_logits=reward_logits,
            reward=reward,
            value_logits=value_logits,
            value=value,
        )

    def init_tree(self, rng_key: chex.PRNGKey, root: AgentOutput, is_eval: bool):
        num_nodes = self._num_simulations + 1
        num_actions = self._action_space.n
        state = jnp.zeros((num_nodes,) + root.state.shape)
        logits = jnp.zeros((num_nodes,) + root.logits.shape)
        prob = jnp.zeros((num_nodes,) + root.logits.shape)
        reward_logits = jnp.zeros((num_nodes,) + root.reward_logits.shape)
        reward = jnp.zeros((num_nodes,) + root.reward.shape)
        value_logits = jnp.zeros((num_nodes,) + root.value_logits.shape)
        value = jnp.zeros((num_nodes,) + root.value.shape)
        action_value = jnp.zeros((num_nodes, num_actions) + root.value.shape)
        depth = jnp.zeros((num_nodes,), dtype=jnp.int32)
        parent = jnp.zeros((num_nodes,), dtype=jnp.int32)
        parent_action = jnp.zeros((num_nodes,), dtype=jnp.int32)
        child = jnp.zeros((num_nodes, num_actions), dtype=jnp.int32)
        visit_count = jnp.zeros((num_nodes, num_actions), dtype=jnp.int32)
        state = state.at[0].set(root.state)
        logits = logits.at[0].set(root.logits)
        noise = jax.random.dirichlet(rng_key, jnp.full((num_actions,), self._alpha))
        # Do not apply the Dirichlet noise during evaluation.
        exploration_prob = jax.lax.select(is_eval, 0., self._exploration_prob)
        root_prob = jax.nn.softmax(root.logits) * (1 - exploration_prob) + exploration_prob * noise
        prob = prob.at[0].set(root_prob)
        reward_logits = reward_logits.at[0].set(root.reward_logits)
        reward = reward.at[0].set(root.reward)
        value_logits = value_logits.at[0].set(root.value_logits)
        value = value.at[0].set(root.value)
        parent = parent.at[0].set(-1)
        parent_action = parent_action.at[0].set(-1)
        tree = Tree(
            state=state,
            logits=logits,
            prob=prob,
            reward_logits=reward_logits,
            reward=reward,
            value_logits=value_logits,
            value=value,
            action_value=action_value,
            depth=depth,
            parent=parent,
            parent_action=parent_action,
            child=child,
            visit_count=visit_count,
        )
        return tree

    def mcts(self, rng_key: chex.PRNGKey, params: Params, root: AgentOutput, is_eval: bool):
        num_actions = self._action_space.n
        max_search_depth = self._max_search_depth
        c1 = self._mcts_c1
        c2 = self._mcts_c2
        discount_factor = self._discount_factor

        def simulate(rng_key: chex.PRNGKey, tree: Tree):
            # First compute the minimum and the maximum action-value in the current tree.
            # Note that these statistics are hard to maintain incrementally because they are non-monotonic.
            is_valid = jnp.clip(tree.visit_count, 0, 1)
            action_value = tree.action_value
            q_min = jnp.min(jnp.where(is_valid, action_value, jnp.full_like(action_value, jnp.inf)))
            q_max = jnp.max(jnp.where(is_valid, action_value, jnp.full_like(action_value, -jnp.inf)))
            q_min = jax.lax.select(is_valid.sum() == 0, 0., q_min)
            q_max = jax.lax.select(is_valid.sum() == 0, 0., q_max)

            def _select_action(rng_key: chex.PRNGKey, t, q_mean):
                # Assign an estimated value to the unvisited nodes.
                # See Eq. (8) in https://arxiv.org/pdf/2111.00210.pdf
                # and https://github.com/YeWR/EfficientZero/blob/main/core/ctree/cnode.cpp#L96.
                q = action_value[t]
                q = jax.lax.select(tree.visit_count[t] > 0, q, jnp.full_like(q, q_mean))
                # Normalize the action-values of the current node so that they are in [0, 1].
                # This is required for the pUCT rule.
                # See Eq. (5) in https://www.nature.com/articles/s41586-020-03051-4.pdf
                q = (q - q_min) / jnp.maximum(q_max - q_min, self._q_normalize_epsilon)
                p = tree.prob[t]
                n = tree.visit_count[t]
                # The action scores are computed by the pUCT rule.
                # See Eq. (2) in https://www.nature.com/articles/s41586-020-03051-4.pdf.
                score = q + p * jnp.sqrt(n.sum()) / (1 + n) * (c1 + jnp.log((n.sum() + c2 + 1) / c2))
                best_actions = score >= score.max() - self._child_select_epsilon
                tie_breaking_prob = best_actions / best_actions.sum()
                return jax.random.choice(rng_key, num_actions, p=tie_breaking_prob)

            def _cond(loop_state):
                rng_key, p, a, q_mean = loop_state
                return jnp.logical_and(tree.depth[p] + 1 < max_search_depth, tree.visit_count[p, a] > 0)

            def _body(loop_state):
                rng_key, p, a, q_mean = loop_state
                p = tree.child[p, a]
                is_valid_child = jnp.clip(tree.visit_count[p], 0, 1)
                q_mean = (q_mean + jnp.sum(tree.action_value[p] * is_valid_child)) / (jnp.sum(is_valid_child) + 1)
                rng_key, sub_key = jax.random.split(rng_key)
                a = _select_action(sub_key, p, q_mean)
                return rng_key, p, a, q_mean

            is_valid_child = jnp.clip(tree.visit_count[0], 0, 1)
            q_mean = jnp.sum(tree.action_value[0] * is_valid_child) / jnp.maximum(jnp.sum(is_valid_child), 1)
            rng_key, sub_key = jax.random.split(rng_key)
            a = _select_action(sub_key, 0, q_mean)
            _, p, a, _ = jax.lax.while_loop(
                _cond,
                _body,
                (rng_key, 0, a, q_mean),
            )
            return p, a

        def expand(tree: Tree, p, a, c):
            p_state = tree.state[p]
            model_out = self.model_step(params, p_state, a)
            tree = tree._replace(
                state=tree.state.at[c].set(model_out.state),
                logits=tree.logits.at[c].set(model_out.logits),
                prob=tree.prob.at[c].set(jax.nn.softmax(model_out.logits)),
                reward_logits=tree.reward_logits.at[c].set(model_out.reward_logits),
                reward=tree.reward.at[c].set(model_out.reward),
                value_logits=tree.value_logits.at[c].set(model_out.value_logits),
                value=tree.value.at[c].set(model_out.value),
                depth=tree.depth.at[c].set(tree.depth[p] + 1),
                parent=tree.parent.at[c].set(p),
                parent_action=tree.parent_action.at[c].set(a),
                child=tree.child.at[p, a].set(c),
            )
            return tree

        def backup(tree: Tree, c):
            def _update(tree, c, g):
                g = tree.reward[c] + discount_factor * g
                p = tree.parent[c]
                a = tree.parent_action[c]
                new_n = tree.visit_count[p, a] + 1
                new_q = (tree.action_value[p, a] * tree.visit_count[p, a] + g) / new_n
                tree = tree._replace(
                    visit_count=tree.visit_count.at[p, a].add(1),
                    action_value=tree.action_value.at[p, a].set(new_q),
                )
                return tree, p, g

            tree, _, _ = jax.lax.while_loop(
                lambda t: t[1] > 0,
                lambda t: _update(t[0], t[1], t[2]),
                (tree, c, tree.value[c]),
            )
            return tree

        def body_fn(sim, loop_state):
            rng_key, tree = loop_state
            rng_key, simulate_key = jax.random.split(rng_key)
            p, a = simulate(simulate_key, tree)
            c = sim + 1
            tree = expand(tree, p, a, c)
            tree = backup(tree, c)
            return rng_key, tree

        rng_key, init_key = jax.random.split(rng_key)
        tree = self.init_tree(init_key, root, is_eval)
        rng_key, tree = jax.lax.fori_loop(
            0, self._num_simulations, body_fn, (rng_key, tree))

        return tree

    def act_prob(self, visit_count: chex.Array, temperature: float):
        """Compute the final policy recommended by MCTS for acting."""
        unnormalized = jnp.power(visit_count, 1. / temperature)
        act_prob = unnormalized / unnormalized.sum(axis=-1, keepdims=True)
        return act_prob

    def value(self, value: chex.Array, action_value: chex.Array, visit_count: chex.Array):
        """Compute the improved value estimation recommended by MCTS."""
        total_value = value + jnp.sum(action_value * visit_count, axis=-1)
        total_count = 1 + visit_count.sum(axis=-1)
        return total_value / total_count
