"""Utility functions used for training models."""

import functools
from collections.abc import Callable, Mapping

import jax
import jax.numpy as jnp

from quadruped_mjx_rl.domain_randomization import DomainRandomizationFn
from quadruped_mjx_rl.environments.physics_pipeline import Env, PipelineModel
from quadruped_mjx_rl.environments.wrappers import wrap_for_training
from quadruped_mjx_rl.types import Params, PRNGKey

PMAP_AXIS_NAME = "i"


def unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)


def strip_weak_type(tree):
    # brax user code is sometimes ambiguous about weak_type.
    # to avoid extra jit recompilations, we strip all weak types from user input
    def f(leaf):
        leaf = jnp.asarray(leaf)
        return leaf.astype(leaf.dtype)

    return jax.tree_util.tree_map(f, tree)


def maybe_wrap_env(
    env: Env,
    wrap_env: bool,
    num_envs: int,
    episode_length: int | None,
    action_repeat: int,
    device_count: int,
    key_env: PRNGKey,
    wrap_env_fn: Callable | None = None,
    randomization_fn: DomainRandomizationFn | None = None,
    vision: bool = False,
):
    """Wraps the environment for training/eval if wrap_env is True."""
    if not wrap_env:
        return env
    if episode_length is None:
        raise ValueError("episode_length must be specified in train")
    v_randomization_fn = None
    randomization_batch_size = num_envs // device_count
    if randomization_fn is not None:
        # randomization_batch_size = num_envs // device_count
        # all devices get the same randomization rng
        # randomization_rng = jax.random.split(key_env, randomization_batch_size)
        v_randomization_fn = functools.partial(
            randomization_fn, rng_key=key_env, num_worlds=randomization_batch_size
        )
    wrap = wrap_env_fn or wrap_for_training
    return wrap(
        env,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=v_randomization_fn,
        vision=vision,
        num_vision_envs=randomization_batch_size,
    )


def random_translate_pixels(
    obs: Mapping[str, jax.Array], key: PRNGKey
) -> Mapping[str, jax.Array]:
    """Apply random translations to B x T x ... pixel observations.

    The same shift is applied across the unroll_length (T) dimension.

    Args:
      obs: a dictionary of observations
      key: a PRNGKey

    Returns:
      A dictionary of observations with translated pixels
    """

    @jax.vmap
    def rt_all_views(ub_obs: Mapping[str, jax.Array], key: PRNGKey) -> Mapping[str, jax.Array]:
        # Expects a dictionary of unbatched observations.
        def rt_view(img: jax.Array, padding: int, key: PRNGKey) -> jax.Array:  # TxHxWxC
            # Randomly translates a set of pixel inputs.
            # Adapted from https://github.com/ikostrikov/jaxrl/blob/main/jaxrl/agents/drq/augmentations.py
            crop_from = jax.random.randint(key, (2,), 0, 2 * padding + 1)
            zero = jnp.zeros((1,), dtype=jnp.int32)
            crop_from = jnp.concatenate([zero, crop_from, zero])
            padded_img = jnp.pad(
                img,
                ((0, 0), (padding, padding), (padding, padding), (0, 0)),
                mode="edge",
            )
            return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)

        out = {}
        for k_view, v_view in ub_obs.items():
            if k_view.startswith("pixels/"):
                key, key_shift = jax.random.split(key)
                out[k_view] = rt_view(v_view, 4, key_shift)
        return {**ub_obs, **out}

    bdim = next(iter(obs.items()), None)[1].shape[0]
    keys = jax.random.split(key, bdim)
    obs = rt_all_views(obs, keys)
    return obs


def remove_pixels(
    obs: jnp.ndarray | Mapping[str, jax.Array],
) -> jnp.ndarray | Mapping[str, jax.Array]:
    """Removes pixel observations from the observation dict."""
    if not isinstance(obs, Mapping):
        return obs
    return {k: v for k, v in obs.items() if not k.startswith("pixels/")}


def param_size(params: Params) -> int:
    return jax.tree.reduce(lambda c, x: c + x.size, params, 0)
