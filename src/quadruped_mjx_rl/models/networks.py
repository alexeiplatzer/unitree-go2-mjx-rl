"""Utility functions for instantiating neural networks."""

# Typing
from dataclasses import dataclass
from collections.abc import Callable, Mapping, Collection
from quadruped_mjx_rl.types import (
    ObservationSize,
    PreprocessObservationFn,
    identity_observation_preprocessor,
)

# Math
import jax
import jax.numpy as jnp
from flax import linen
from quadruped_mjx_rl import running_statistics


@dataclass
class FeedForwardNetwork:
    init: Callable
    apply: Callable


def _get_obs_state_size(obs_size: ObservationSize, obs_key: str) -> int:
    obs_size = obs_size[obs_key] if isinstance(obs_size, Mapping) else obs_size
    return jax.tree_util.tree_flatten(obs_size)[0][-1]


def normalizer_select(
    processor_params: running_statistics.RunningStatisticsState, obs_key: str
) -> running_statistics.RunningStatisticsState:
    return running_statistics.RunningStatisticsState(
        count=processor_params.count,
        mean=processor_params.mean[obs_key],
        summed_variance=processor_params.summed_variance[obs_key],
        std=processor_params.std[obs_key],
    )


def make_network(
    module: linen.Module,
    obs_size: ObservationSize,
    preprocess_observations_fn: PreprocessObservationFn = identity_observation_preprocessor,
    squeeze_output: bool = False,
):
    def apply(processor_params, params, obs):
        obs = preprocess_observations_fn(obs, processor_params)
        out = module.apply(params, obs)
        if squeeze_output:
            return jnp.squeeze(out, axis=-1)
        return out

    dummy_obs = jax.tree_util.tree_map(
        lambda x: jnp.zeros((1,) + x),
        obs_size,
    )
    return FeedForwardNetwork(init=lambda key: module.init(key, dummy_obs), apply=apply)
