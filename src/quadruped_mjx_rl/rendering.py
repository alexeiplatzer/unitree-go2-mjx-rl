# Import all necessary packages

# Supporting
from typing import Protocol
from etils.epath import PathLike
from collections.abc import Callable
import functools
from dataclasses import dataclass, field
from typing import TypeVar, Generic

# Math
import jax
import jax.numpy as jp
import mediapy as media

# Brax
from brax import envs
from brax.envs.base import PipelineEnv, Env
from brax.io import model
from brax.training.acme import running_statistics

from .environments import EnvironmentConfig
from .robots import RobotConfig
from .models import ModelConfig


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
    "default": lambda: RenderConfig(),
}

EnvType = TypeVar("EnvType", bound=PipelineEnv)


def render(
    environment: type[EnvType],
    env_config: EnvironmentConfig[EnvType],
    robot_config: RobotConfig,
    init_scene_path: PathLike,
    model_config: ModelConfig,
    make_networks_fn: Callable,
    make_inference_fn: Callable,
    trained_model_path: PathLike,
    render_config: RenderConfig,
    animation_save_path: PathLike | None,
):
    """
    Render a simulation rollout.

    This function sets up the simulation environment, initializes the neural network inference
    from a trained model, and executes a rollout based on a predefined movement command.
    The rendered simulation is either displayed interactively or saved as an animation file.

    Parameters:
        environment (type[PipelineEnv]): The Brax environment class to be rendered.
        env_config (EnvironmentConfig): Configuration parameters for the environment, including its name and simulation settings.
        robot_config (RobotConfig): Configuration parameters for the robot, such as its physical and control properties.
        init_scene_path (PathLike): File path to the initial scene configuration.
        model_config (ModelConfig): A mapping defining the model architecture (e.g., module names to layer sizes).
        make_networks_fn (Callable): Factory function to create the neural network architectures based on the model configuration.
        make_inference_fn (Callable): Function that generates the inference callable given the constructed networks.
        trained_model_path (PathLike): File path to the saved parameters of the trained model.
        render_config (RenderConfig): Rendering configuration that includes parameters such as seed, episode length,
                                      number of steps, rendering frequency, command velocities, and other rendering options.
        save_animation (bool, optional): If True, the rendered animation will be saved to disk instead of being displayed.
                                         Defaults to False.
        animation_save_path (PathLike or None, optional): File path to save the animation if save_animation is True.
                                                          Defaults to None.

    Returns:
        None
    """

    envs.register_environment(env_config.name, environment)
    env = envs.get_environment(
        env_config.name,
        environment_config=env_config,
        robot_config=robot_config,
        init_scene_path=init_scene_path,
    )

    # TODO: come up with more reasonable names for factories of factories and so on...
    modules_hidden_layers = {
        f"{module.name}_hidden_layer_sizes": tuple(module.hidden_layers)
        for module in model_config.modules
    }
    make_networks_factory = functools.partial(
        make_networks_fn,
        **modules_hidden_layers,
    )

    nets = make_networks_factory(
        # Observation_size argument doesn't matter since it's only used for param init.
        observation_size=1,
        action_size=env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
    )

    inference_factory = make_inference_fn(nets)

    params = model.load_params(trained_model_path)

    # Inference function
    ppo_inference_fn = inference_factory(params)

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

    render_rollout(
        jax.jit(demo_env.reset),
        jax.jit(demo_env.step),
        jax.jit(ppo_inference_fn),
        demo_env,
        render_config=render_config,
        save_path=animation_save_path,
        the_command=movement_command,
        camera="track",
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
