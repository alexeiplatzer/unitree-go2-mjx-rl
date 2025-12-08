from typing import Any, Generic, Protocol, TypeVar

import jax
from flax.struct import dataclass as flax_dataclass

from quadruped_mjx_rl.running_statistics import RunningStatisticsState
from quadruped_mjx_rl.types import (
    Action,
    Extra,
    Observation,
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
        sample_key: PRNGKey,
    ) -> tuple[Action, Extra]:
        pass


class PolicyWithLatents(Protocol):
    def __call__(
        self,
        observation: Observation,
        latent_encoding: jax.Array,
        sample_key: PRNGKey,
    ) -> tuple[Action, Extra]:
        pass


class RecurrentEncoder(Protocol):
    def __call__(
        self,
        observation: Observation,
        done: jax.Array,
        key: PRNGKey,
        recurrent_carry: RecurrentCarry,
    ) -> tuple[jax.Array, RecurrentCarry]:
        pass


class InitCarryFn(Protocol):
    def __call__(self, rng: PRNGKey) -> RecurrentCarry:
        pass


class PreprocessObservationFn(Protocol):
    def __call__(
        self,
        observation: jax.Array,
        preprocessor_params: PreprocessorParams,
    ) -> jax.Array:
        pass


def identity_observation_preprocessor(
    observation: jax.Array, preprocessor_params: PreprocessorParams
):
    del preprocessor_params
    return observation


@flax_dataclass
class RecurrentAgentState:
    recurrent_carry: RecurrentCarry
    recurrent_buffer: jax.Array
    done_buffer: jax.Array


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
