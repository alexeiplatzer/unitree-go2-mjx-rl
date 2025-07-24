"""Utility functions for instantiating neural networks."""

# Typing
from dataclasses import dataclass
from collections.abc import Callable, Mapping, Collection, Sequence
from quadruped_mjx_rl.types import (
    Observation,
    ObservationSize,
    PreprocessObservationFn,
    PreprocessorParams,
    Params,
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


def preprocess_by_key(
    *,
    preprocess_obs_fn: PreprocessObservationFn,
    processor_params: PreprocessorParams,
    obs: Observation,
    preprocess_obs_keys: Collection[str] = (),
):
    """Preprocesses the specified observations only, returns the same structure."""
    if not isinstance(obs, Mapping):
        return preprocess_obs_fn(obs, processor_params)
    return {
        obs_key: obs[obs_key] if obs_key not in preprocess_obs_keys
        else preprocess_obs_fn(obs[obs_key], normalizer_select(processor_params, obs_key))
        for obs_key in obs
    }


def make_network(
    module: linen.Module,
    obs_size: ObservationSize,
    preprocess_observations_fn: PreprocessObservationFn = identity_observation_preprocessor,
    preprocess_obs_keys: Collection[str] = (),
    apply_to_obs_keys: Sequence[str] = ("state",),
    squeeze_output: bool = False,
):
    def to_inputs(obs: Observation) -> list[jax.Array]:
        if isinstance(obs, Mapping):
            return [obs[k] for k in apply_to_obs_keys]
        else:
            return [obs]

    def apply(processor_params: PreprocessorParams, params: Params, obs: Observation):
        obs = preprocess_by_key(
            preprocess_obs_fn=preprocess_observations_fn,
            processor_params=processor_params,
            obs=obs,
            preprocess_obs_keys=preprocess_obs_keys,
        )
        inputs = to_inputs(obs)
        out = module.apply(params, *inputs)
        if squeeze_output:
            return jnp.squeeze(out, axis=-1)
        return out

    dummy_obs = jax.tree_util.tree_map(
        lambda x: jnp.zeros((1,) + x) if isinstance(x, tuple) else jnp.zeros((1, x)),
        obs_size,
    )
    dummy_inputs = to_inputs(dummy_obs)
    return FeedForwardNetwork(init=lambda key: module.init(key, *dummy_inputs), apply=apply)
