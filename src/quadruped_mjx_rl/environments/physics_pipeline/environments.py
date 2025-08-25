
from abc import ABC, abstractmethod

import jax
import mujoco
from flax.struct import dataclass as flax_dataclass
from mujoco import mjx

from quadruped_mjx_rl.environments.physics_pipeline.base import Base
from quadruped_mjx_rl.types import Observation, ObservationSize


@flax_dataclass
class State(Base):
    """Environment state for training and inference."""

    pipeline_state: mjx.Data
    obs: Observation
    reward: jax.Array
    done: jax.Array
    metrics: dict[str, jax.Array]
    info: dict[str, ...]


class Env(ABC):
    """Interface for a Reinforcement Learning environment."""

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
    @abstractmethod
    def env_model(self) -> mujoco.MjModel:
        """Mujoco model of the environment."""

    @property
    @abstractmethod
    def pipeline_model(self) -> mjx.Model:
        """Mjx model of the environment."""

    @property
    def unwrapped(self) -> "Env":
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
    def env_model(self) -> mujoco.MjModel:
        return self.env.env_model

    @property
    def pipeline_model(self) -> mjx.Model:
        return self.env.pipeline_model

    @property
    def unwrapped(self) -> Env:
        return self.env.unwrapped

    def __getattr__(self, name):
        if name == "__setstate__":
            raise AttributeError(name)
        return getattr(self.env, name)
