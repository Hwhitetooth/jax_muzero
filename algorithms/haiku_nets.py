"""Haiku neural network modules."""
from typing import Tuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp


class ResidualConvBlockV1(hk.Module):
    """A v1 residual convolutional block."""
    def __init__(self, channels: int, stride: int, use_projection: bool, name='residual_conv_block_v1'):
        super(ResidualConvBlockV1, self).__init__(name=name)
        self._use_projection = use_projection
        if use_projection:
            self._proj_conv = hk.Conv2D(channels, kernel_shape=3, stride=stride, padding='SAME', with_bias=False)
            self._proj_ln = hk.LayerNorm(axis=(-3, -2, -1), create_scale=True, create_offset=True)
        self._conv_0 = hk.Conv2D(channels, kernel_shape=3, stride=stride, padding='SAME', with_bias=False)
        self._ln_0 = hk.LayerNorm(axis=(-3, -2, -1), create_scale=True, create_offset=True)
        self._conv_1 = hk.Conv2D(channels, kernel_shape=3, stride=1, padding='SAME', with_bias=False)
        self._ln_1 = hk.LayerNorm(axis=(-3, -2, -1), create_scale=True, create_offset=True)

    def __call__(self, x):
        # NOTE: Replacing BatchNorm with LayerNorm is totally fine for RL.
        #   See https://arxiv.org/pdf/2104.06294.pdf Appendix A for an example.
        shortcut = out = x

        if self._use_projection:
            shortcut = self._proj_conv(shortcut)
            shortcut = self._proj_ln(shortcut)

        out = hk.Sequential([
            self._conv_0,
            self._ln_0,
            jax.nn.relu,
            self._conv_1,
            self._ln_1,
        ])(out)

        return jax.nn.relu(shortcut + out)


class ResidualConvBlockV2(hk.Module):
    """A v2 residual convolutional block."""
    def __init__(self, channels, stride: int, use_projection: bool, name='residual_conv_block_v2'):
        super(ResidualConvBlockV2, self).__init__(name=name)
        self._use_projection = use_projection
        if use_projection:
            self._proj_conv = hk.Conv2D(channels, kernel_shape=3, stride=stride, padding='SAME', with_bias=False)
        self._conv_0 = hk.Conv2D(channels, kernel_shape=3, stride=stride, padding='SAME', with_bias=False)
        self._ln_0 = hk.LayerNorm(axis=(-3, -2, -1), create_scale=True, create_offset=True)
        self._conv_1 = hk.Conv2D(channels, kernel_shape=3, stride=1, padding='SAME', with_bias=False)
        self._ln_1 = hk.LayerNorm(axis=(-3, -2, -1), create_scale=True, create_offset=True)

    def __call__(self, x):
        # NOTE: Replacing BatchNorm with LayerNorm is totally fine for RL.
        #   See https://arxiv.org/pdf/2104.06294.pdf Appendix A for an example.
        shortcut = out = x
        out = self._ln_0(out)
        out = jax.nn.relu(out)
        if self._use_projection:
            shortcut = self._proj_conv(out)
        out = hk.Sequential([
            self._conv_0,
            self._ln_1,
            jax.nn.relu,
            self._conv_1,
        ])(out)
        return shortcut + out


class EZStateEncoder(hk.Module):
    """EfficientZero encoder architecture."""
    def __init__(self, channels, use_v2, name='ez_state_encoder'):
        super(EZStateEncoder, self).__init__(name=name)
        self._channels = channels
        self._use_v2 = use_v2

    def __call__(self, observations: chex.Array) -> chex.Array:
        ResBlock = ResidualConvBlockV2 if self._use_v2 else ResidualConvBlockV1
        torso = [
            lambda x: x / 255.,
            hk.Conv2D(self._channels // 2, kernel_shape=3, stride=2, padding='SAME', with_bias=False),
        ]
        if not self._use_v2:
            torso.extend([
                hk.LayerNorm(axis=(-3, -2, -1), create_scale=True, create_offset=True),
                jax.nn.relu,
            ])
        torso.append(ResBlock(self._channels // 2, stride=1, use_projection=False))
        torso.append(ResBlock(self._channels, stride=2, use_projection=True))
        torso.extend([
            ResBlock(self._channels, stride=1, use_projection=False),
            hk.AvgPool(window_shape=(3, 3, 1), strides=(2, 2, 1), padding='SAME'),
            ResBlock(self._channels, stride=1, use_projection=False),
            hk.AvgPool(window_shape=(3, 3, 1), strides=(2, 2, 1), padding='SAME'),
            ResBlock(self._channels, stride=1, use_projection=False),
        ])
        return hk.Sequential(torso)(observations)


class EZPrediction(hk.Module):
    def __init__(self, num_actions, num_bins, output_init_scale, use_v2, name='ez_prediction'):
        super(EZPrediction, self).__init__(name=name)
        self._num_actions = num_actions
        self._num_bins = num_bins
        self._output_init_scale = output_init_scale
        self._use_v2 = use_v2

    def __call__(self, states: chex.Array) -> Tuple[chex.Array, chex.Array, chex.Array]:
        # NOTE: Somehow reward predictions do not have a residual block before the 1x1 convolutions.
        ResBlock = ResidualConvBlockV2 if self._use_v2 else ResidualConvBlockV1
        output_init = hk.initializers.VarianceScaling(scale=self._output_init_scale)
        reward_head = []
        if self._use_v2:
            reward_head.extend([
                hk.LayerNorm(axis=(-3, -2, -1), create_scale=True, create_offset=True),
                jax.nn.relu,
            ])
        reward_head.extend([
            hk.Conv2D(16, kernel_shape=1, stride=1, padding='SAME', with_bias=False),
            hk.LayerNorm(axis=(-3, -2, -1), create_scale=True, create_offset=True),
            jax.nn.relu,
            hk.Flatten(-3),
            hk.Linear(32, with_bias=False),
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
            jax.nn.relu,
            hk.Linear(self._num_bins, w_init=output_init),
        ])
        reward_logits = hk.Sequential(reward_head)(states)
        out = ResBlock(states.shape[-1], stride=1, use_projection=False)(states)
        value_head = []
        if self._use_v2:
            value_head.extend([
                hk.LayerNorm(axis=(-3, -2, -1), create_scale=True, create_offset=True),
                jax.nn.relu,
            ])
        value_head.extend([
            hk.Conv2D(16, kernel_shape=1, stride=1, padding='SAME', with_bias=False),
            hk.LayerNorm(axis=(-3, -2, -1), create_scale=True, create_offset=True),
            jax.nn.relu,
            hk.Flatten(-3),
            hk.Linear(32, with_bias=False),
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
            jax.nn.relu,
            hk.Linear(self._num_bins, w_init=output_init),
        ])
        value_logits = hk.Sequential(value_head)(out)
        logits_head = []
        if self._use_v2:
            logits_head.extend([
                hk.LayerNorm(axis=(-3, -2, -1), create_scale=True, create_offset=True),
                jax.nn.relu,
            ])
        logits_head.extend([
            hk.Conv2D(16, kernel_shape=1, stride=1, padding='SAME', with_bias=False),
            hk.LayerNorm(axis=(-3, -2, -1), create_scale=True, create_offset=True),
            jax.nn.relu,
            hk.Flatten(-3),
            hk.Linear(32, with_bias=False),
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
            jax.nn.relu,
            hk.Linear(self._num_actions, w_init=output_init),
        ])
        logits = hk.Sequential(logits_head)(out)
        return logits, reward_logits, value_logits


class EZTransition(hk.Module):
    """EfficientZero transition architecture."""
    def __init__(self, use_v2, name='ez_transition'):
        super(EZTransition, self).__init__(name=name)
        self._use_v2 = use_v2

    def __call__(self, inputs, prev_state) -> chex.Array:
        channels = prev_state.shape[-1]
        ResBlock = ResidualConvBlockV2 if self._use_v2 else ResidualConvBlockV1
        shortcut = prev_state
        if self._use_v2:
            prev_state = hk.LayerNorm(axis=(-3, -2, -1), create_scale=True, create_offset=True)(prev_state)
            prev_state = jax.nn.relu(prev_state)
        inputs = inputs[None, None, :]
        action_one_hot = jnp.broadcast_to(inputs, prev_state.shape[:-1] + inputs.shape[-1:])
        x_and_h = jnp.concatenate([prev_state, action_one_hot], axis=-1)
        out = hk.Conv2D(channels, kernel_shape=3, stride=1, padding='SAME', with_bias=False)(x_and_h)
        if self._use_v2:
            out += shortcut
        else:
            out = hk.LayerNorm(axis=(-3, -2, -1), create_scale=True, create_offset=True)(out)
            out = jax.nn.relu(out + shortcut)
        out = ResBlock(channels, stride=1, use_projection=False)(out)
        return out
