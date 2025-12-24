import functools
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import jax
import mediapy as media
import numpy as np
from etils.epath import PathLike
from jax import numpy as jnp
import functools
from typing import Protocol

from etils.epath import PathLike

from quadruped_mjx_rl import running_statistics
from quadruped_mjx_rl.models import io
from quadruped_mjx_rl.models.architectures import (
    TeacherStudentAgentParams,
    ActorCriticConfig,
    ModelConfig,
    TeacherStudentConfig,
    TeacherStudentVisionConfig,
    TeacherStudentRecurrentConfig,
    TeacherStudentNetworks,
    TeacherStudentRecurrentNetworks,
)
from quadruped_mjx_rl.models.architectures.configs_base import ComponentNetworksArchitecture
from quadruped_mjx_rl.models.types import (
    AgentNetworkParams,
    identity_observation_preprocessor,
    PreprocessObservationFn,
)
from quadruped_mjx_rl.types import ObservationSize
from quadruped_mjx_rl.config_utils import Configuration, register_config_base_class
from quadruped_mjx_rl.environments import PipelineEnv
from quadruped_mjx_rl.environments.wrappers import EpisodeWrapper
from quadruped_mjx_rl.models import ModelConfig, get_networks_factory
from quadruped_mjx_rl.models.types import Params


# from quadruped_mjx_rl.environments.wrappers import MadronaWrapper

# TODO this all probably outdated, see todos


@dataclass
class RenderConfig(Configuration):
    episode_length: int = 1000
    render_every: int = 2
    seed: int = 0
    n_steps: int = 500
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
    # animation_save_path: PathLike | dict[str, PathLike] | None,
    # video_maker: PolicyRenderingFn = show_video,
    # vision: bool = False,
) -> tuple[Sequence[np.ndarray], float]:
    model_class = type(model_config).get_model_class()
    if not isinstance(model_params, model_class.agent_params_class()):
        raise ValueError(
            f"The provided params have the wrong type: {type(model_params)}"
            f" - while {model_class.agent_params_class()} were expected!"
        )
    network_factory = get_networks_factory(model_config)
    preprocess_fn = (
        running_statistics.normalize
        if normalize_observations
        else lambda x, y: x
    )
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
        preprocess_observations_fn=preprocess_fn,
    )
    # TODO: finish implementing
    unroll_factories = {"training_acting_policy": network.make_unroll_fn(model_params)}
    if isinstance(network, TeacherStudentRecurrentNetworks):
        pass
    elif isinstance(network, TeacherStudentNetworks):
        unroll_fn = network.make_unroll_fn(model_params)
        unroll_factories = {
            "teacher": network.get_acting_policy_factory(),
            "student": network.get_student_policy_factory(),
        }
    demo_env = EpisodeWrapper(
        env,
        episode_length=render_config.episode_length,
        action_repeat=1,
    )
    # TODO update
    # if vision:
    #     demo_env = MadronaWrapper(demo_env, num_worlds=1)

    render_fn = functools.partial(
        render_rollout,
        env=demo_env,
        render_config=render_config,
    )
    if isinstance(ppo_inference_fn, dict):
        for name, inference_fn in ppo_inference_fn.items():
            frames, fps = render_fn(
                inference_fn=inference_fn,
            )
            video_maker(frames, fps)
    else:
        frames, fps = render_fn(
            inference_fn=ppo_inference_fn,
        )
        video_maker(frames, fps)


def render_rollout(
    env: PipelineEnv,
    inference_fn: Callable,
    render_config: RenderConfig,
) -> tuple[Sequence[np.ndarray], float]:
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    act_fn = jax.jit(inference_fn)
    command = jnp.array(
        [
            render_config.command["x_vel"],
            render_config.command["y_vel"],
            render_config.command["ang_vel"],
        ]
    )
    render_every = render_config.render_every

    rng = jax.random.key(render_config.seed)
    state = reset_fn(rng)
    state.info["command"] = command
    rollout = [state.pipeline_state]

    for i in range(render_config.n_steps):
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = act_fn(state.obs, act_rng)
        state = step_fn(state, ctrl)
        if i % render_every == 0:
            rollout.append(state.pipeline_state)

    frames = env.render(rollout, camera="track")
    fps = 1.0 / (env.dt.item() * render_every)
    return frames, fps
