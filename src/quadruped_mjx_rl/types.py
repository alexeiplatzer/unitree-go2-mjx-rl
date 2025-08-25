"""Brax training types."""

from collections.abc import Mapping
from typing import Any, NamedTuple, Protocol, Tuple, TypeVar

from jax.numpy import ndarray

from quadruped_mjx_rl.running_statistics import NestedArray, RunningStatisticsState

Params = Any
PRNGKey = ndarray
Metrics = Mapping[str, ndarray]
Observation = ndarray | Mapping[str, ndarray]
ObservationSize = int | Mapping[str, tuple[int, ...] | int]
Action = ndarray
Extra = Mapping[str, Any]
PreprocessorParams = RunningStatisticsState
PolicyParams = Tuple[PreprocessorParams, Params]
NetworkType = TypeVar("NetworkType")


class Transition(NamedTuple):
    """Container for a transition."""

    observation: NestedArray
    action: NestedArray
    reward: NestedArray
    discount: NestedArray
    next_observation: NestedArray
    extras: NestedArray = ()


class Policy(Protocol):
    def __call__(
        self,
        observation: Observation,
        key: PRNGKey,
    ) -> Tuple[Action, Extra]:
        pass


class PreprocessObservationFn(Protocol):
    def __call__(
        self,
        observation: Observation,
        preprocessor_params: PreprocessorParams,
    ) -> ndarray:
        pass


def identity_observation_preprocessor(
    observation: Observation, preprocessor_params: PreprocessorParams
):
    del preprocessor_params
    return observation


class NetworkFactory(Protocol[NetworkType]):
    def __call__(
        self,
        observation_size: ObservationSize,
        action_size: int,
        preprocess_observations_fn: PreprocessObservationFn = identity_observation_preprocessor,
    ) -> NetworkType:
        pass
