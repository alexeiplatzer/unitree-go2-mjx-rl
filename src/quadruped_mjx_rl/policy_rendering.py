
# Typing
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

# Supporting
import functools
from etils.epath import PathLike

# Math
import jax
from jax import numpy as jnp
import numpy as np

# Sim
from quadruped_mjx_rl.environments import PipelineEnv
from quadruped_mjx_rl.environments.wrappers import EpisodeWrapper

# IO
import mediapy as media

# ML
from quadruped_mjx_rl.models import load_inference_fn
from quadruped_mjx_rl.models import ModelConfig
from quadruped_mjx_rl.environments.wrappers import MadronaWrapper

# Configs
from quadruped_mjx_rl.config_utils import Configuration, register_config_base_class


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
    ppo_inference_fn,
    render_config: RenderConfig,
    #animation_save_path: PathLike | dict[str, PathLike] | None,
    video_maker: PolicyRenderingFn = show_video,
    vision: bool = False,
):
    # Inference function
    # ppo_inference_fn = load_inference_fn(
    #     trained_model_path, model_config, action_size=env.action_size
    # )

    demo_env = EpisodeWrapper(
        env,
        episode_length=render_config.episode_length,
        action_repeat=1,
    )
    if vision:
        demo_env = MadronaWrapper(demo_env, num_worlds=1)

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
    command = jnp.array([
        render_config.command["x_vel"],
        render_config.command["y_vel"],
        render_config.command["ang_vel"]
    ])
    render_every = render_config.render_every

    rng = jax.random.key(render_config.seed)
    state = reset_fn(rng)
    state.info["command"] = command
    rollout = [state.pipeline_state]

    for i in range(render_config.n_steps):
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = inference_fn(state.obs, act_rng)
        state = step_fn(state, ctrl)
        if i % render_every == 0:
            rollout.append(state.pipeline_state)

    frames = env.render(rollout, camera="track")
    fps = 1.0 / (env.dt.item() * render_every)
    return frames, fps
