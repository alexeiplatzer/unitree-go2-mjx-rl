# Typing
from abc import ABC, abstractmethod
from collections.abc import Sequence
from quadruped_mjx_rl.types import Observation, ObservationSize

# Supporting
from quadruped_mjx_rl.environments.utils.rendering import render_array

# Math
import jax
from flax.struct import dataclass as flax_dataclass
import numpy as np

# Sim
import mujoco
from mujoco import mjx
from quadruped_mjx_rl.environments.utils import Base
from quadruped_mjx_rl.environments.physics_pipeline import pipeline_init, pipeline_n_steps


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
    def env_model(self) -> mujoco.MjModel:
        return self.env.env_model

    @property
    def pipeline_model(self) -> mjx.Model:
        return self.env.pipeline_model

    @property
    def unwrapped(self) -> Env:
        return self.env.unwrapped

    def __getattr__(self, name):
        if name == '__setstate__':
            raise AttributeError(name)
        return getattr(self.env, name)


class PipelineEnv(Env):
    """API for driving an mjx-based environment for training and inference."""

    def __init__(
        self,
        env_model: mujoco.MjModel,
        sim_dt: float = 0.004,
        ctrl_dt: float = 0.02,
    ):
        """Initializes PipelineEnv.

        Args:
            env_model: a model describing the physical environment
            sim_dt: the timestep of the physical simulation
            ctrl_dt: the time interval between reinforcement learning steps
        """
        self._env_model = env_model
        self._env_model.opt.timestep = sim_dt
        self._pipeline_model = mjx.put_model(self._env_model)
        self._sim_dt = sim_dt
        self._ctrl_dt = ctrl_dt
        self._n_frames = int(ctrl_dt / sim_dt)

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
    ) -> mjx.Data:
        """Initializes the pipeline state."""
        return pipeline_init(self._pipeline_model, q, qd, act, ctrl)

    def pipeline_step(self, pipeline_state: mjx.Data, action: jax.Array) -> mjx.Data:
        """Takes a physics step using the physics pipeline."""
        return pipeline_n_steps(self._pipeline_model, pipeline_state, action, self._n_frames)

    @property
    def dt(self) -> jax.Array:
        """The timestep used for each env step."""
        return self._pipeline_model.opt.timestep * self._n_frames

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
        return self._pipeline_model.nu

    @property
    def env_model(self) -> mujoco.MjModel:
        return self._env_model

    @property
    def pipeline_model(self) -> mjx.Model:
        return self._pipeline_model

    def render(
        self,
        trajectory: list[mjx.Data],
        height: int = 240,
        width: int = 320,
        camera: str | None = None,
    ) -> Sequence[np.ndarray]:
        """Renders a trajectory using the MuJoCo renderer."""
        return render_array(self._env_model, trajectory, height, width, camera)
