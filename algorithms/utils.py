import chex
import jax
import jax.numpy as jnp
import numpy as np
import tree as tree_util


def pack_namedtuple_jnp(xs, axis=0):
    return jax.tree_map(lambda *xs: jnp.stack(xs, axis=axis), *xs)


def pack_namedtuple_np(xs, axis=0):
    return tree_util.map_structure(lambda *xs: np.stack(xs, axis=axis), *xs)


def unpack_namedtuple_jnp(structure, axis=0):
    transposed = jax.tree_map(lambda t: jnp.moveaxis(t, axis, 0), structure)
    flat = jax.tree_flatten(transposed)
    unpacked = list(map(lambda xs: jax.tree_unflatten(structure, xs), zip(*flat)))
    return unpacked


def unpack_namedtuple_np(structure, axis=0):
    transposed = tree_util.map_structure(lambda t: np.moveaxis(t, axis, 0), structure)
    flat = tree_util.flatten(transposed)
    unpacked = list(map(lambda xs: tree_util.unflatten_as(structure, xs), zip(*flat)))
    return unpacked


def scale_gradient(g, scale: float):
    return g * scale + jax.lax.stop_gradient(g) * (1. - scale)


def weighted_mean(x: chex.Array, w: chex.Array):
    return jnp.sum(x * w) / jnp.maximum(w.sum(), 1.)


def weighted_std(x: chex.Array, w: chex.Array):
    mean = weighted_mean(x, w)
    return jnp.sqrt(jnp.sum((x - mean) ** 2 * w) / jnp.maximum(w.sum(), 1.))


def scalar_to_two_hot(x: chex.Array, num_bins: int):
    """A categorical representation of real values. Ref: https://www.nature.com/articles/s41586-020-03051-4.pdf."""
    max_val = (num_bins - 1) // 2
    x = jnp.clip(x, -max_val, max_val)
    x_low = jnp.floor(x).astype(jnp.int32)
    x_high = jnp.ceil(x).astype(jnp.int32)
    p_high = x - x_low
    p_low = 1. - p_high
    idx_low = x_low + max_val
    idx_high = x_high + max_val
    cat_low = jax.nn.one_hot(idx_low, num_bins) * p_low[..., None]
    cat_high = jax.nn.one_hot(idx_high, num_bins) * p_high[..., None]
    return cat_low + cat_high


def logits_to_scalar(logits: chex.Array):
    """The inverse of the scalar_to_two_hot function above."""
    num_bins = logits.shape[-1]
    max_val = (num_bins - 1) // 2
    x = jnp.sum((jnp.arange(num_bins) - max_val) * jax.nn.softmax(logits), axis=-1)
    return x


def value_transform(x: chex.Array, epsilon: float = 1E-3):
    """A non-linear value transformation for variance reduction. Ref: https://arxiv.org/abs/1805.11593."""
    return jnp.sign(x) * (jnp.sqrt(jnp.abs(x) + 1) - 1) + epsilon * x


def inv_value_transform(x: chex.Array, epsilon: float = 1E-3):
    """The inverse of the non-linear value transformation above."""
    return jnp.sign(x) * (((jnp.sqrt(1 + 4 * epsilon * (jnp.abs(x) + 1 + epsilon)) - 1) / (2 * epsilon)) ** 2 - 1)
