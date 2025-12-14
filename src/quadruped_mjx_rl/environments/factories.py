from collections.abc import Callable
from typing import Any

from etils.epath import PathLike

from quadruped_mjx_rl.physics_pipeline import (
    EnvModel,
    load_to_model,
    PipelineModel,
)
from quadruped_mjx_rl.environments.quadruped import (
    EnvironmentConfig,
    QuadrupedBaseEnv,
)
from quadruped_mjx_rl.robots import RobotConfig

QuadrupedEnvFactory = Callable[[], type(QuadrupedBaseEnv)]


def is_obs_key_vision(obs_key: str) -> bool:
    """Checks if the observation key corresponds to a vision observation."""
    return obs_key.startswith("pixels/") or obs_key.endswith("_map")


def resolve_env_class(env_config: EnvironmentConfig) -> type[QuadrupedBaseEnv]:
    return type(env_config).get_environment_class()


def get_base_model(init_scene_path: PathLike, env_config: EnvironmentConfig) -> EnvModel:
    base_model = load_to_model(init_scene_path)
    return resolve_env_class(env_config).customize_model(base_model, env_config)


def get_env_factory(
    robot_config: RobotConfig,
    environment_config: EnvironmentConfig,
    env_model: EnvModel,
    customize_model: bool = True,
    use_vision: bool = False,
    renderer_maker: Callable[[PipelineModel], Any] | None = None,
) -> QuadrupedEnvFactory:
    """
    Prepares parameters to instantiate an environment. For vision environments, also wraps
    the environment with a vision wrapper.
    """
    env_class = resolve_env_class(environment_config)
    if customize_model:
        env_model = env_class.customize_model(env_model, environment_config)
    env_factory = lambda: env_class(
        robot_config=robot_config, environment_config=environment_config, env_model=env_model
    )
    if use_vision:
        if renderer_maker is None:
            raise ValueError("Vision rendering requires a renderer_maker.")
        env_factory = lambda: environment_config.vision_env_config.get_vision_wrapper_class()(
            env=env_factory(),
            vision_env_config=environment_config.vision_env_config,
            renderer_maker=renderer_maker,
        )
    return env_factory
