import dataclasses
from collections.abc import Callable, Mapping

import jax
import jax.numpy as jnp
from brax.training import types
from brax.training.acme import running_statistics
from flax import linen


@dataclasses.dataclass
class FeedForwardNetwork:
    init: Callable
    apply: Callable


def _get_obs_state_size(obs_size: types.ObservationSize, obs_key: str) -> int:
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
    obs_size: types.ObservationSize,
    preprocess_observations_fn: types.PreprocessObservationFn = (
            types.identity_observation_preprocessor
    ),
    obs_keys: str | tuple[str, ...] = "state",
    squeeze_output: bool = False,
):
    def preprocess_by_key(obs, processor_params, obs_key=obs_keys):
        return preprocess_observations_fn(
            obs[obs_key], normalizer_select(processor_params, obs_key)
        )

    def preprocess_multiple_obs(obs, processor_params):
        obs = [
            preprocess_by_key(obs, processor_params, obs_key)
            if obs_key != "latent" else obs[obs_key]
            for obs_key in obs_keys
        ]
        return jnp.concatenate(obs, axis=-1)

    if isinstance(obs_size, Mapping):
        if isinstance(obs_keys, tuple):
            preprocess_observations = preprocess_multiple_obs
            obs_size = sum((_get_obs_state_size(obs_size, obs_key) for obs_key in obs_keys))
        else:
            preprocess_observations = preprocess_by_key
            obs_size = _get_obs_state_size(obs_size, obs_keys)
    else:
        preprocess_observations = preprocess_observations_fn
        obs_size = _get_obs_state_size(obs_size, obs_keys)

    def apply(processor_params, params, obs):
        obs = preprocess_observations(obs, processor_params)
        out = module.apply(params, obs)
        if squeeze_output:
            return jnp.squeeze(out, axis=-1)
        return out

    dummy_obs = jnp.zeros((1, obs_size))
    return FeedForwardNetwork(init=lambda key: module.init(key, dummy_obs), apply=apply)
