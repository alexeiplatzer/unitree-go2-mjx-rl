"""Utility functions for instantiating neural networks."""

import functools
import logging
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import TypeVar

import jax
import jax.numpy as jnp
from flax import linen

from quadruped_mjx_rl import running_statistics
from quadruped_mjx_rl.models import AgentParams, FeedForwardNetwork, RecurrentNetwork
from quadruped_mjx_rl.models.distributions import ParametricDistribution
from quadruped_mjx_rl.types import (
    Action,
    Extra,
    Observation,
    ObservationSize,
    PRNGKey,
)
from quadruped_mjx_rl.models.types import (
    identity_observation_preprocessor,
    Params,
    Policy,
    PreprocessObservationFn,
    PreprocessorParams,
    RecurrentAgentState,
    RecurrentCarry,
)


def normalizer_select(
    processor_params: running_statistics.RunningStatisticsState, obs_key: str
) -> running_statistics.RunningStatisticsState:
    return running_statistics.RunningStatisticsState(
        count=processor_params.count,
        mean=processor_params.mean[obs_key],
        summed_variance=processor_params.summed_variance[obs_key],
        std=processor_params.std[obs_key],
    )


def preprocess_obs_by_key(
    *,
    preprocess_obs_fn: PreprocessObservationFn,
    processor_params: PreprocessorParams,
    obs: Observation,
    preprocess_obs_keys: Collection[str] = (),
) -> Observation:
    """Preprocesses the specified observations only, returns the same structure."""
    if not isinstance(obs, Mapping):
        return preprocess_obs_fn(obs, processor_params)
    return {
        obs_key: (
            obs[obs_key]
            if obs_key not in preprocess_obs_keys
            else preprocess_obs_fn(obs[obs_key], normalizer_select(processor_params, obs_key))
        )
        for obs_key in obs
    }


WrappedFunOutput = TypeVar("WrappedFunOutput")


def wrap_network_input(
    fun: Callable[..., WrappedFunOutput],
    apply_to_obs_keys: Sequence[str] = ("proprioceptive",),
    concatenate_inputs: bool = True,
) -> Callable[..., WrappedFunOutput]:
    """Allows the function to be only applied to the selected obs keys,
    concatenated optionally."""

    def wrap(first_arg, obs, *args, **kwargs):
        if isinstance(obs, Mapping):
            obs_list = [obs[k] for k in apply_to_obs_keys]
            if concatenate_inputs:
                input_vector = jnp.concatenate(obs_list, axis=-1)
                return fun(first_arg, input_vector, *args, **kwargs)
            else:
                return fun(first_arg, *obs_list, *args, **kwargs)
        else:
            return fun(first_arg, obs, *args, **kwargs)

    return wrap


def wrap_apply(
    apply_fun: Callable[..., WrappedFunOutput],
    preprocess_observation_fn: PreprocessObservationFn = identity_observation_preprocessor,
    preprocess_obs_keys: Collection[str] = (),
    squeeze_output: bool = False,
) -> Callable[..., WrappedFunOutput]:
    def wrap(
        preprocessor_params: PreprocessorParams,
        params: Params,
        obs: Observation,
        *args,
        **kwargs,
    ):
        obs = preprocess_obs_by_key(
            preprocess_obs_fn=preprocess_observation_fn,
            processor_params=preprocessor_params,
            obs=obs,
            preprocess_obs_keys=preprocess_obs_keys,
        )

        output = apply_fun(params, obs, *args, **kwargs)

        if squeeze_output:
            if not isinstance(output, tuple):
                return jnp.squeeze(output, axis=-1)
            else:
                first_output, *rest = output
                return jnp.squeeze(first_output, axis=-1), *rest
        return output

    return wrap


def wrap_init(
    apply_fun: Callable[..., WrappedFunOutput], obs_size: ObservationSize
) -> Callable[..., WrappedFunOutput]:
    logging.info(f"observation size: {obs_size}")

    def wrap(first_arg, *args, **kwargs):
        dummy_obs = jax.tree_util.tree_map(
            lambda x: jnp.zeros((1,) + x) if isinstance(x, tuple) else jnp.zeros((1, x)),
            obs_size,
            is_leaf=lambda x: isinstance(x, tuple) and all(isinstance(e, int) for e in x),
        )
        return apply_fun(first_arg, dummy_obs, *args, **kwargs)

    return wrap


def wrap_apply_recurrent_buffers(
    apply_fun: Callable[
        [
            Params,
            PreprocessorParams,
            Observation,
            jax.Array,
            RecurrentCarry,
            jax.Array,
            jax.Array,
            PRNGKey,
        ],
        tuple[jax.Array, tuple[jax.Array, jax.Array], jax.Array, jax.Array],
    ],
) -> Callable[
    [
        Params,
        PreprocessorParams,
        Observation,
        jax.Array,
        RecurrentAgentState,
        PRNGKey,
    ],
    tuple[jax.Array, RecurrentAgentState],
]:
    def wrap(
        params: Params,
        preprocessor_params: PreprocessorParams,
        obs: Observation,
        done: jax.Array,
        recurrent_agent_state: RecurrentAgentState,
        key: PRNGKey,
    ) -> tuple[jax.Array, RecurrentAgentState]:
        output, recurrent_carry, recurrent_buffer, done_buffer = apply_fun(
            params,
            preprocessor_params,
            obs,
            done,
            recurrent_agent_state.recurrent_carry,
            recurrent_agent_state.recurrent_buffer,
            recurrent_agent_state.done_buffer,
            key,
        )
        return output, RecurrentAgentState(recurrent_carry, recurrent_buffer, done_buffer)

    return wrap


def make_network(
    module: linen.Module,
    obs_size: ObservationSize,
    preprocess_observations_fn: PreprocessObservationFn = identity_observation_preprocessor,
    preprocess_obs_keys: Collection[str] = (),
    apply_to_obs_keys: Sequence[str] = ("proprioceptive",),
    concatenate_inputs: bool = True,
    squeeze_output: bool = False,
    recurrent: bool = False,
):
    """
    Creates a network that selects and preprocesses the specified observations.
    """
    wrap_input = functools.partial(
        wrap_network_input,
        apply_to_obs_keys=apply_to_obs_keys,
        concatenate_inputs=concatenate_inputs,
    )

    apply = wrap_apply(
        wrap_input(module.apply),
        preprocess_observation_fn=preprocess_observations_fn,
        preprocess_obs_keys=preprocess_obs_keys,
        squeeze_output=squeeze_output,
    )

    init = wrap_init(wrap_input(module.init), obs_size=obs_size)

    if recurrent:
        apply_differentiable = wrap_apply_recurrent_buffers(apply)
        apply_encode = wrap_apply(
            wrap_input(functools.partial(module.apply, method=module.encode)),
            preprocess_observation_fn=preprocess_observations_fn,
            preprocess_obs_keys=preprocess_obs_keys,
            squeeze_output=squeeze_output,
        )
        init_carry = wrap_input(module.initialize_carry)
        return RecurrentNetwork(
            init=init,
            init_carry=init_carry,
            apply_differentiable=apply_differentiable,
            apply=apply_encode,
        )
    else:
        return FeedForwardNetwork(init=init, apply=apply)


def policy_factory(
    policy_apply: Callable,
    parametric_action_distribution: ParametricDistribution,
    params: AgentParams,
    deterministic: bool = False,
) -> Policy:
    def process_logits(logits: jax.Array, sample_key: PRNGKey) -> tuple[Action, Extra]:
        if deterministic:
            return parametric_action_distribution.mode(logits), {}
        raw_actions = parametric_action_distribution.sample_no_postprocessing(
            logits, sample_key
        )
        log_prob = parametric_action_distribution.log_prob(logits, raw_actions)
        postprocessed_actions = parametric_action_distribution.postprocess(raw_actions)
        return postprocessed_actions, {
            "log_prob": log_prob,
            "raw_action": raw_actions,
        }

    return lambda obs, rng: process_logits(policy_apply(params, obs), rng)
