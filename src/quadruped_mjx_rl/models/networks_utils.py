"""Utility functions for instantiating neural networks."""

import functools
import logging
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import TypeVar

import jax
import jax.numpy as jnp
from flax import linen

from quadruped_mjx_rl.environments import is_obs_key_vision
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
) -> Observation:
    """Preprocesses the specified observations only, returns the same structure."""
    if not isinstance(obs, Mapping):
        return preprocess_obs_fn(obs, processor_params)
    return {
        obs_key: (
            obs[obs_key]
            if is_obs_key_vision(obs_key)  # Do not preprocess visual observations
            else preprocess_obs_fn(obs[obs_key], normalizer_select(processor_params, obs_key))
        )
        for obs_key in obs
    }

def make_dummy_obs(obs_size: ObservationSize) -> Observation:
    return jax.tree_util.tree_map(
        lambda x: jnp.zeros((1,) + x) if isinstance(x, tuple) else jnp.zeros((1, x)),
        obs_size,
        is_leaf=lambda x: isinstance(x, tuple),
    )


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
