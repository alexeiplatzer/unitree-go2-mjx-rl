# Typing
from typing import Any
from abc import ABC, abstractmethod
from collections.abc import Sequence
from quadruped_mjx_rl.types import Observation, ObservationSize

# Supporting
from quadruped_mjx_rl.environments.rendering import render_array

# Math
import jax
from flax.struct import dataclass as flax_dataclass, field as flax_field
import numpy as np

# Sim
from quadruped_mjx_rl.environments.utils import System, Base
from quadruped_mjx_rl.environments.pipeline_utils import (
    PipelineState,
    init as pipeline_init,
    step as pipeline_step,
)


@flax_dataclass
class State(Base):
    """Environment state for training and inference."""

    pipeline_state: PipelineState
    obs: Observation
    reward: jax.Array
    done: jax.Array
    metrics: dict[str, jax.Array] = flax_field(default_factory=dict)
    info: dict[str, Any] = flax_field(default_factory=dict)


class Env(ABC):
    """Interface for driving training and inference."""

    @abstractmethod
    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""

    @abstractmethod
    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""

    @property
    @abstractmethod
    def observation_size(self) -> ObservationSize:
        """The size of the observation vector returned in step and reset."""

    @property
    @abstractmethod
    def action_size(self) -> int:
        """The size of the action vector expected by step."""

    @property
    def unwrapped(self) -> 'Env':
        return self


class Wrapper(Env):
    """Wraps an environment to allow modular transformations."""

    def __init__(self, env: Env):
        self.env = env

    def reset(self, rng: jax.Array) -> State:
        return self.env.reset(rng)

    def step(self, state: State, action: jax.Array) -> State:
        return self.env.step(state, action)

    @property
    def observation_size(self) -> ObservationSize:
        return self.env.observation_size

    @property
    def action_size(self) -> int:
        return self.env.action_size

    @property
    def unwrapped(self) -> Env:
        return self.env.unwrapped

    def __getattr__(self, name):
        if name == '__setstate__':
            raise AttributeError(name)
        return getattr(self.env, name)


class PipelineEnv(Env):
    """API for driving a brax system for training and inference."""

    def __init__(
        self,
        sys: System,
        n_frames: int = 1,
    ):
        """Initializes PipelineEnv.

        Args:
            sys: system defining the kinematic tree and other properties
            n_frames: the number of times to step the physics pipeline for each
                environment step
    """
        self.sys = sys
        self._n_frames = n_frames

    @abstractmethod
    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""

    @abstractmethod
    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""

    def pipeline_init(
        self,
        q: jax.Array,
        qd: jax.Array,
        act: jax.Array | None = None,
        ctrl: jax.Array | None = None,
    ) -> PipelineState:
        """Initializes the pipeline state."""
        return pipeline_init(self.sys, q, qd, act, ctrl)

    def pipeline_step(self, pipeline_state: Any, action: jax.Array) -> PipelineState:
        """Takes a physics step using the physics pipeline."""

        def f(state, _):
            return (
                pipeline_step(self.sys, state, action),
                None,
            )

        return jax.lax.scan(f, pipeline_state, (), self._n_frames)[0]

    @property
    def dt(self) -> jax.Array:
        """The timestep used for each env step."""
        return self.sys.opt.timestep * self._n_frames

    @property
    def observation_size(self) -> ObservationSize:
        rng = jax.random.PRNGKey(0)
        reset_state = self.unwrapped.reset(rng)
        obs = reset_state.obs
        if isinstance(obs, jax.Array):
            return obs.shape[-1]
        return jax.tree_util.tree_map(lambda x: x.shape, obs)

    @property
    def action_size(self) -> int:
        return self.sys.act_size()

    def render(
        self,
        trajectory: list[PipelineState],
        height: int = 240,
        width: int = 320,
        camera: str | None = None,
    ) -> Sequence[np.ndarray]:
        """Renders a trajectory using the MuJoCo renderer."""
        return render_array(self.sys, trajectory, height, width, camera)
