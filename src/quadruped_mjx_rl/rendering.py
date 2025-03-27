# Import all necessary packages

# Supporting
from typing import Protocol
from etils.epath import PathLike
from collections.abc import Callable
from functools import partial
from dataclasses import dataclass, field
from typing import TypeVar, Generic

# Math
import jax
import jax.numpy as jp
import mediapy as media
import numpy as np

# Brax
from brax import envs
from brax.envs.base import PipelineEnv

from .environments import EnvironmentConfig
from .robots import RobotConfig
from .models import ModelConfig, load_inference_fn


@dataclass
class RenderConfig:
    name: str = "default"
    episode_length: int = 1000
    render_every: int = 2
    seed: int = 0
    n_steps: int = 500
    command: dict[str, float] = field(default_factory=lambda: {
        "x_vel": 0.5,
        "y_vel": 0.0,
        "ang_vel": 0.0,
    })


name_to_rendering_config = {
    "default": RenderConfig,
}

EnvType = TypeVar("EnvType", bound=PipelineEnv)


def render(
    # environment: type[EnvType],
    # env_config: EnvironmentConfig[EnvType],
    # robot_config: RobotConfig,
    # init_scene_path: PathLike,
    env,
    model_config: ModelConfig,
    trained_model_path: PathLike,
    render_config: RenderConfig,
    animation_save_path: PathLike | dict[str, PathLike] | None,
    vision: bool = False,
):
    # Environment
    # envs.register_environment(env_config.name, environment)
    # env = envs.get_environment(
    #     env_config.name,
    #     environment_config=env_config,
    #     robot_config=robot_config,
    #     init_scene_path=init_scene_path,
    # )

    # Inference function
    ppo_inference_fn = load_inference_fn(
        trained_model_path, model_config, action_size=env.action_size, vision=vision
    )

    # Commands
    x_vel = render_config.command["x_vel"]
    y_vel = render_config.command["y_vel"]
    ang_vel = render_config.command["ang_vel"]

    movement_command = jp.array([x_vel, y_vel, ang_vel])

    demo_env = envs.training.EpisodeWrapper(
        env,
        episode_length=render_config.episode_length,
        action_repeat=1,
    )

    render_fn = partial(
        render_rollout,
        reset_fn=jax.jit(demo_env.reset),
        step_fn=jax.jit(demo_env.step),
        env=demo_env,
        render_config=render_config,
        the_command=movement_command,
        camera="track",
    )
    if isinstance(ppo_inference_fn, dict):
        for name, inference_fn in ppo_inference_fn.items():
            render_fn(
                inference_fn=jax.jit(inference_fn),
                save_path=animation_save_path[name],
            )
    else:
        render_fn(
            inference_fn=jax.jit(ppo_inference_fn),
            save_path=animation_save_path,
        )


def render_rollout(
    reset_fn: Callable,
    step_fn: Callable,
    inference_fn: Callable,
    env: PipelineEnv,
    render_config: RenderConfig,
    save_path: PathLike | None,
    camera=None,
    the_command=None,
):
    rng = jax.random.key(render_config.seed)
    render_every = render_config.render_every
    state = reset_fn(rng)
    if the_command is not None:
        state.info["command"] = the_command
    rollout = [state.pipeline_state]

    for i in range(render_config.n_steps):
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = inference_fn(state.obs, act_rng)
        state = step_fn(state, ctrl)
        if i % render_every == 0:
            rollout.append(state.pipeline_state)

    rendering = env.render(rollout, camera=camera)  # necessary
    if save_path is not None:
        media.write_video(
            save_path,
            env.render(rollout, camera=camera),
            fps=1.0 / (env.dt * render_every),
            codec="gif",
        )
    else:
        media.show_video(
            env.render(rollout, camera=camera),
            fps=1.0 / (env.dt * render_every),
            codec="gif",
        )
