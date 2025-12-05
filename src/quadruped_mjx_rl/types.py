"""Brax training types."""

from collections.abc import Mapping
from typing import Any, NamedTuple

import jax

from quadruped_mjx_rl.running_statistics import (
    NestedArray,
)

# Reinforcement learning types
Observation = jax.Array | Mapping[str, jax.Array]
ObservationSize = int | Mapping[str, tuple[int, ...] | int]
Action = jax.Array
Metrics = Mapping[str, jax.Array] | Mapping[str, float]

# Utility types
PRNGKey = jax.Array
ArraySize = int | tuple[int, ...]
Extra = Mapping[str, Any]


class Transition(NamedTuple):
    """Container for a transition."""

    observation: NestedArray
    action: NestedArray
    reward: NestedArray
    discount: NestedArray
    next_observation: NestedArray
    extras: NestedArray = ()
