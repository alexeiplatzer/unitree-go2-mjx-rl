from abc import abstractmethod
from collections.abc import Sequence
import jax
import numpy as np
from quadruped_mjx_rl.environments.physics_pipeline import (
    EnvModel,
    PipelineModel,
    PipelineState,
    Env,
    pipeline_step,
    pipeline_init,
    make_pipeline_model,
    State,
    render_array
)
from quadruped_mjx_rl.types import ObservationSize


def pipeline_n_steps(
    pipeline_model: PipelineModel, pipeline_state: PipelineState, act: jax.Array, n_steps: int = 1
) -> PipelineState:
    """Performs n sequential pipeline steps"""
    return jax.lax.scan(
        (lambda x, _: (pipeline_step(pipeline_model, x, act), None)),
        pipeline_state,
        (),
        n_steps,
    )[0]


class PipelineEnv(Env):
    """API for driving an mjx-based environment for training and inference."""

    def __init__(
        self,
        env_model: EnvModel,
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
        self._pipeline_model = make_pipeline_model(self._env_model)
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
    ) -> PipelineState:
        """Initializes the pipeline state."""
        return pipeline_init(self._pipeline_model, q, qd, act, ctrl)

    def pipeline_step(self, pipeline_state: PipelineState, action: jax.Array) -> PipelineState:
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
    def env_model(self) -> EnvModel:
        return self._env_model

    @property
    def pipeline_model(self) -> PipelineModel:
        return self._pipeline_model

    def render(
        self,
        trajectory: list[PipelineState],
        height: int = 240,
        width: int = 320,
        camera: str | None = None,
    ) -> Sequence[np.ndarray]:
        """Renders a trajectory using the MuJoCo renderer."""
        return render_array(self._env_model, trajectory, height, width, camera)