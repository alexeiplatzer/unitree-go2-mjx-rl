"""Utility functions for instantiating neural networks."""

import logging
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

import jax
import jax.numpy as jnp
from flax import linen
from flax.struct import dataclass as flax_dataclass

from quadruped_mjx_rl import running_statistics, types
from quadruped_mjx_rl.models.architectures.configs_base import AgentModel
from quadruped_mjx_rl.models.distributions import ParametricDistribution
from quadruped_mjx_rl.types import (
    Action,
    Extra,
    identity_observation_preprocessor,
    Observation,
    ObservationSize,
    Params,
    Policy,
    PreprocessObservationFn,
    PreprocessorParams,
    PRNGKey,
    RecurrentHiddenState,
    RecurrentPolicy,
)


@flax_dataclass
class RecurrentAgentState:
    recurrent_carry: RecurrentHiddenState
    recurrent_buffer: jax.Array
    done_buffer: jax.Array


@dataclass
class FeedForwardNetwork:
    init: Callable[[PRNGKey], Params]
    apply: Callable[[Params, PreprocessorParams, Observation], jax.Array]


@dataclass
class RecurrentNetwork:
    init: Callable[[PRNGKey], Params]
    init_carry: Callable[[PRNGKey, int], RecurrentHiddenState]
    apply: Callable[
        [
            Params,
            PreprocessorParams,
            Observation,
            jax.Array,
            RecurrentAgentState,
            PRNGKey,
        ],
        tuple[jax.Array, RecurrentAgentState],
    ]


AgentNetworkParams = TypeVar("AgentNetworkParams")


@flax_dataclass
class AgentParams(Generic[AgentNetworkParams]):
    preprocessor_params: PreprocessorParams
    network_params: AgentNetworkParams

    def restore_params(
        self,
        restore_params: "AgentParams[AgentNetworkParams]",
        restore_value: bool = False,
    ) -> "AgentParams[AgentNetworkParams]":
        pass


class PolicyFactory(Protocol[AgentNetworkParams]):
    def __call__(
        self,
        params: AgentParams[AgentNetworkParams],
        deterministic: bool,
    ) -> Policy:
        pass


def policy_factory(
    policy_apply: Callable,
    parametric_action_distribution: ParametricDistribution,
    params: AgentParams,
    deterministic: bool = False,
    recurrent: bool = False,
) -> Policy | RecurrentPolicy:
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

    def recurrent_policy(
        observation: Observation,
        done: jax.Array,
        sample_key: PRNGKey,
        recurrent_state: RecurrentHiddenState,
    ) -> tuple[Action, RecurrentHiddenState, Extra]:
        policy_logits, next_recurrent_state = policy_apply(
            params, observation, done, recurrent_state
        )
        action, extra = process_logits(policy_logits, sample_key)
        return action, next_recurrent_state, extra

    if recurrent:
        return recurrent_policy
    else:
        return lambda obs, rng: process_logits(policy_apply(params, obs), rng)


class NetworkFactory(Protocol[AgentNetworkParams]):
    def __call__(
        self,
        observation_size: ObservationSize,
        action_size: int,
        preprocess_observations_fn: PreprocessObservationFn = identity_observation_preprocessor,
    ) -> AgentModel[AgentNetworkParams]:
        pass


WrappedFunOutput = TypeVar("WrappedFunOutput")


def recurry(fun: Callable[..., WrappedFunOutput]) -> Callable[..., WrappedFunOutput]:
    return lambda *args, **kwargs: lambda first_arg: fun(first_arg, *args, **kwargs)


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


@recurry
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


@recurry
def wrap_apply_with_preprocessor(
    apply_fun: Callable[..., WrappedFunOutput],
    preprocess_observation_fn: PreprocessObservationFn = identity_observation_preprocessor,
    preprocess_obs_keys: Collection[str] = (),
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
        return apply_fun(params, obs, *args, **kwargs)

    return wrap


@recurry
def maybe_squeeze_output(output, squeeze_output: bool = False):
    if squeeze_output:
        return jnp.squeeze(output, axis=-1)
    else:
        return output


def pipeline(f1, f2):
    def fun(*args, **kwargs):
        return f2(f1(*args, **kwargs))

    return fun


def pipeline_first_output(f1, f2):
    def fun(*args, **kwargs):
        first, *rest = f1(*args, **kwargs)
        return f2(first), *rest

    return fun


@recurry
def wrap_apply_to_dummy_obs(
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


def wrap_apply_recurrent(
    apply_fun: Callable[
        [
            Params,
            PreprocessorParams,
            Observation,
            jax.Array,
            RecurrentHiddenState,
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
    wrap_input = wrap_network_input(apply_to_obs_keys, concatenate_inputs)

    apply = wrap_apply_with_preprocessor(preprocess_observations_fn, preprocess_obs_keys)(
        wrap_input(module.apply)
    )

    wrap_dummy_obs = wrap_apply_to_dummy_obs(obs_size)
    init = wrap_dummy_obs(wrap_input(module.init))

    maybe_squeeze = maybe_squeeze_output(squeeze_output)

    if recurrent:
        apply = wrap_apply_recurrent(apply)
        apply = pipeline_first_output(apply, maybe_squeeze)
        init_carry = wrap_input(module.initialize_carry)
        return RecurrentNetwork(init=init, apply=apply, init_carry=init_carry)
    else:
        apply = pipeline(apply, maybe_squeeze)
        return FeedForwardNetwork(init=init, apply=apply)
