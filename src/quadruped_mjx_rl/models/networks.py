"""Utility functions for instantiating neural networks."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar, Protocol
import logging

import jax
import jax.numpy as jnp
from flax import linen
from flax.struct import dataclass as flax_dataclass

from quadruped_mjx_rl import running_statistics
from quadruped_mjx_rl.models.distributions import ParametricDistribution
from quadruped_mjx_rl.types import (
    identity_observation_preprocessor,
    Observation,
    ObservationSize,
    Params,
    PreprocessObservationFn,
    PreprocessorParams,
    PRNGKey,
    Policy,
    Action,
    Extra,
)


@dataclass
class FeedForwardNetwork:
    init: Callable
    apply: Callable


@dataclass
class RecurrentNetwork:
    init: Callable
    init_carry: Callable
    apply: Callable


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


class ComponentNetworkArchitecture(ABC, Generic[AgentNetworkParams]):
    @abstractmethod
    def agent_params_class(self):
        pass

    @abstractmethod
    def initialize(self, rng: PRNGKey) -> AgentNetworkParams:
        pass

    @abstractmethod
    def policy_apply(self, params: AgentNetworkParams, observation: Observation) -> jax.Array:
        """Gets the logits for applying the network's policy with the provided params to the
        provided observations"""
        pass

    @abstractmethod
    def policy_metafactory(self) -> tuple[PolicyFactory[AgentNetworkParams], ...]:
        pass


def policy_factory(
    policy_apply,
    parametric_action_distribution: ParametricDistribution,
    params: AgentParams,
    deterministic: bool = False,
):
    def policy(
        observation: Observation,
        sample_key: PRNGKey,
    ) -> tuple[Action, Extra]:
        policy_logits = policy_apply(params, observation)
        if deterministic:
            return parametric_action_distribution.mode(policy_logits), {}
        raw_actions = parametric_action_distribution.sample_no_postprocessing(
            policy_logits, sample_key
        )
        log_prob = parametric_action_distribution.log_prob(policy_logits, raw_actions)
        postprocessed_actions = parametric_action_distribution.postprocess(raw_actions)
        return postprocessed_actions, {
            "log_prob": log_prob,
            "raw_action": raw_actions,
        }

    return policy


class NetworkFactory(Protocol[AgentNetworkParams]):
    def __call__(
        self,
        observation_size: ObservationSize,
        action_size: int,
        preprocess_observations_fn: PreprocessObservationFn = identity_observation_preprocessor,
    ) -> ComponentNetworkArchitecture[AgentNetworkParams]:
        pass


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
        obs_key: (
            obs[obs_key]
            if obs_key not in preprocess_obs_keys
            else preprocess_obs_fn(obs[obs_key], normalizer_select(processor_params, obs_key))
        )
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
    """
    Creates a feedforward network that selects and preprocesses the specified observations.
    """
    def to_input_vector(obs: Observation) -> jax.Array:
        """Selects the needed observations and concatenates them into a single input vector."""
        if isinstance(obs, Mapping):
            return jnp.concatenate([obs[k] for k in apply_to_obs_keys], axis=-1)
        else:
            return obs

    def apply(
        processor_params: PreprocessorParams, params: Params, obs: Observation
    ) -> jax.Array:
        obs = preprocess_by_key(
            preprocess_obs_fn=preprocess_observations_fn,
            processor_params=processor_params,
            obs=obs,
            preprocess_obs_keys=preprocess_obs_keys,
        )
        input_vector = to_input_vector(obs)
        out = module.apply(params, input_vector)
        if squeeze_output:
            return jnp.squeeze(out, axis=-1)
        return out

    logging.info(f"observation size: {obs_size}")

    dummy_obs = jax.tree_util.tree_map(
        lambda x: jnp.zeros((1,) + x) if isinstance(x, tuple) else jnp.zeros((1, x)),
        obs_size,
        is_leaf=lambda x: isinstance(x, tuple),
    )
    dummy_input = to_input_vector(dummy_obs)
    return FeedForwardNetwork(init=lambda key: module.init(key, dummy_input), apply=apply)
