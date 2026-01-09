import functools
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import jax
import mediapy as media
import numpy as np
from etils.epath import PathLike
from jax import numpy as jnp

from quadruped_mjx_rl import running_statistics
from quadruped_mjx_rl.config_utils import Configuration, register_config_base_class
from quadruped_mjx_rl.environments import (
    PipelineEnv,
    QuadrupedJoystickBaseEnv,
    is_obs_key_vision,
)
from quadruped_mjx_rl.environments.wrappers import _vmap_wrap_with_randomization
from quadruped_mjx_rl.environments.wrappers import EpisodeWrapper
from quadruped_mjx_rl.models import get_networks_factory, ModelConfig
from quadruped_mjx_rl.models.architectures import (
    TeacherStudentAgentParams,
    TeacherStudentNetworks,
    TeacherStudentRecurrentNetworks,
)
from quadruped_mjx_rl.models.types import Params
from quadruped_mjx_rl.domain_randomization import DomainRandomizationConfig


@dataclass
class RenderConfig(Configuration):
    episode_length: int = 4096
    render_every: int = 2
    seed: int = 0
    n_steps: int = 512
    cameras: list[str] = field(default_factory=lambda: ["track"])
    command: dict[str, float] = field(
        default_factory=lambda: {
            "x_vel": 0.5,
            "y_vel": 0.0,
            "ang_vel": 0.0,
        }
    )

    @classmethod
    def config_base_class_key(cls) -> str:
        return "render"


register_config_base_class(RenderConfig)


RolloutRenderer = Callable[[RenderConfig], tuple[Sequence[np.ndarray], float]]

PolicyRenderingFn = Callable[[Sequence[np.ndarray], float], None]


def show_video(frames: Sequence[np.ndarray], fps: float):
    media.show_video(frames, fps=fps, codec="gif")


def save_video(frames: Sequence[np.ndarray], fps: float, save_path: PathLike):
    media.write_video(save_path, frames, fps=fps, codec="gif")


def render_policy_rollout(
    env: PipelineEnv,
    model_config: ModelConfig,
    model_params: Params,
    render_config: RenderConfig,
    normalize_observations: bool = False,
    domain_rand_config: DomainRandomizationConfig | None = None,
) -> tuple[
    dict[str, tuple[dict[str, Sequence[np.ndarray]], dict[str, Sequence[np.ndarray]] | None]],
    float,
]:
    model_class = type(model_config).get_model_class()
    if not isinstance(model_params, model_class.agent_params_class()):
        raise ValueError(
            f"The provided params have the wrong type: {type(model_params)}"
            f" - while {model_class.agent_params_class()} were expected!"
        )
    network_factory = get_networks_factory(model_config)
    preprocess_fn = running_statistics.normalize if normalize_observations else lambda x, y: x
    network = network_factory(
        observation_size={
            "proprioceptive": 1,
            "proprioceptive_history": 1,
            "environment_privileged": 1,
            "pixels/terrain/depth": 1,
            "pixels/frontal_ego/rgb": 1,
            "pixels/frontal_ego/rgb_adjusted": 1,
            "privileged_terrain_map": 1,
        },
        action_size=env.action_size,
        vision_obs_period=None,  # TODO: adapt
        preprocess_observations_fn=preprocess_fn,
    )
    unroll_factories = {
        "training_acting_policy": network.make_acting_unroll_fn(
            model_params, deterministic=True, accumulate_pipeline_states=True
        )
    }
    if isinstance(network, TeacherStudentRecurrentNetworks):
        assert isinstance(model_params, TeacherStudentAgentParams)
        unroll_factories["student_policy"] = network.make_student_unroll_fn(
            model_params, deterministic=True, accumulate_pipeline_states=True
        )
    elif isinstance(network, TeacherStudentNetworks):
        assert isinstance(model_params, TeacherStudentAgentParams)
        unroll_factories["student_policy"] = network.make_student_unroll_fn(
            model_params, deterministic=True, accumulate_pipeline_states=True,
        )
    if hasattr(network, "vision") and getattr(network, "vision"):
        env = _vmap_wrap_with_randomization(
            env,
            vision=True,
            worlds_random_key=jax.random.key(render_config.seed),
            randomization_config=domain_rand_config,
        )
    elif domain_rand_config is not None:
        env = _vmap_wrap_with_randomization(
            env,
            vision=False,
            worlds_random_key=jax.random.key(render_config.seed),
            randomization_config=domain_rand_config,
        )
    demo_env = EpisodeWrapper(
        env,
        episode_length=render_config.episode_length,
        action_repeat=1,
    )

    render_fn = functools.partial(
        render_rollout,
        env=demo_env,
        render_config=render_config,
    )
    fps = 1.0 / (env.dt.item() * render_config.render_every)
    return {
        name: render_fn(unroll_factory=unroll_factory)
        for name, unroll_factory in unroll_factories.items()
    }, fps


def render_rollout(
    env: PipelineEnv,
    unroll_factory: Callable,
    render_config: RenderConfig,
    return_vision_obs: bool = False,
) -> tuple[dict[str, Sequence[np.ndarray]], dict[str, Sequence[np.ndarray]] | None]:
    reset_fn = jax.jit(env.reset)
    unroll_fn = jax.jit(
        functools.partial(unroll_factory, env=env, unroll_length=render_config.n_steps)
    )
    command = jnp.array(
        [
            render_config.command["x_vel"],
            render_config.command["y_vel"],
            render_config.command["ang_vel"],
        ]
    )
    render_every = render_config.render_every

    reset_key, unroll_key = jax.random.split(jax.random.PRNGKey(render_config.seed), 2)
    state = reset_fn(jax.random.split(reset_key, 1))
    if isinstance(env.unwrapped, QuadrupedJoystickBaseEnv):
        state.info["command"] = jnp.expand_dims(command, 0)
    rollout = [jax.tree_util.tree_map(lambda x: x[0], state.pipeline_state)]

    _, transitions = unroll_fn(
        env_state=state,
        key=unroll_key,
    )
    rollout += [
        jax.tree_util.tree_map(
            lambda x: x[i * render_every, 0], transitions.extras["pipeline_states"]
        )
        for i in range(1, render_config.n_steps // render_every)
    ]
    rendering = {camera: env.render(rollout, camera=camera) for camera in render_config.cameras}
    if return_vision_obs:
        vision_obs = {
            vision_key: vision_image
            for vision_key, vision_image in transitions.observations.items()
            if is_obs_key_vision(vision_key)
        }
    else:
        vision_obs = None
    return rendering, vision_obs
