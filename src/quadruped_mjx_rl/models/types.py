from dataclasses import dataclass
from typing import Any, Callable, Generic, Protocol, TypeVar

import jax
from flax.struct import dataclass as flax_dataclass

from quadruped_mjx_rl.models.architectures.configs_base import ComponentNetworksArchitecture
from quadruped_mjx_rl.running_statistics import RunningStatisticsState
from quadruped_mjx_rl.types import (
    Action,
    Extra,
    Observation,
    ObservationSize,
    PRNGKey,
)

Params = Any
PreprocessorParams = RunningStatisticsState
PolicyParams = tuple[PreprocessorParams, Params]
RecurrentCarry = tuple[jax.Array, jax.Array] | None


class Policy(Protocol):
    def __call__(
        self,
        observation: Observation,
        key: PRNGKey,
    ) -> tuple[Action, Extra]:
        pass


class RecurrentPolicyMaker(Protocol):
    def __call__(
        self,
        observation: Observation,
        done: jax.Array,
        key: PRNGKey,
        recurrent_carry: RecurrentCarry,
    ) -> tuple[Policy, RecurrentCarry]:
        pass


class InitCarryFn(Protocol):
    def __call__(self, rng: PRNGKey) -> RecurrentCarry:
        pass


class PreprocessObservationFn(Protocol):
    def __call__(
        self,
        observation: Observation,
        preprocessor_params: PreprocessorParams,
    ) -> jax.Array:
        pass


def identity_observation_preprocessor(
    observation: Observation, preprocessor_params: PreprocessorParams
):
    del preprocessor_params
    return observation


@flax_dataclass
class RecurrentAgentState:
    recurrent_carry: RecurrentCarry
    recurrent_buffer: jax.Array
    done_buffer: jax.Array


@dataclass
class FeedForwardNetwork:
    init: Callable[[PRNGKey], Params]
    apply: Callable[[Params, PreprocessorParams, Observation], jax.Array]


@dataclass
class RecurrentNetwork:
    init: Callable[[PRNGKey], Params]
    init_carry: Callable[[PRNGKey, int], RecurrentCarry]
    apply: Callable[
        [
            Params,
            PreprocessorParams,
            Observation,
            jax.Array,
            RecurrentCarry,
            PRNGKey,
        ],
        tuple[jax.Array, RecurrentCarry],
    ]
    apply_differentiable: Callable[
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


class NetworkFactory(Protocol[AgentNetworkParams]):
    def __call__(
        self,
        observation_size: ObservationSize,
        action_size: int,
        preprocess_observations_fn: PreprocessObservationFn = identity_observation_preprocessor,
    ) -> ComponentNetworksArchitecture[AgentNetworkParams]:
        pass
