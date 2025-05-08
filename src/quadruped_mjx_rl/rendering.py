# Import all necessary packages

# Supporting
from etils.epath import PathLike
from collections.abc import Callable
from functools import partial
from typing import TypeVar

# Math
import jax
import jax.numpy as jp
import mediapy as media

# Brax
from brax import envs
from brax.envs.base import PipelineEnv

from quadruped_mjx_rl.environments import EnvironmentConfig
from quadruped_mjx_rl.robots import RobotConfig
from quadruped_mjx_rl.models import ModelConfig
from quadruped_mjx_rl.models import load_inference_fn

from dataclasses import dataclass, field


@dataclass
class RenderConfig:
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
    rendering_class: str = "default"


rendering_config_classes = {
    "default": RenderConfig,
}


def render(
    env_factory: Callable[[], PipelineEnv],
    model_config: ModelConfig,
    trained_model_path: PathLike,
    render_config: RenderConfig,
    animation_save_path: PathLike | dict[str, PathLike] | None,
    vision: bool = False,
):
    # Environment
    env = env_factory()

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

    # rendering = env.render(rollout, camera=camera)  # necessary?
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
