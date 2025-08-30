import functools
from collections.abc import Callable

from etils.epath import PathLike

from quadruped_mjx_rl.environments.base import PipelineEnv
from quadruped_mjx_rl.environments.physics_pipeline import (
    EnvModel,
    load_to_model,
)
from quadruped_mjx_rl.environments.quadruped.base import (
    EnvironmentConfig,
    QuadrupedBaseEnv,
)
from quadruped_mjx_rl.environments.quadruped.joystick_base import (
    JoystickBaseEnvConfig,
    QuadrupedJoystickBaseEnv,
)
from quadruped_mjx_rl.environments.quadruped.joystick_teacher_student import (
    QuadrupedJoystickTeacherStudentEnv,
    TeacherStudentEnvironmentConfig,
)
from quadruped_mjx_rl.environments.quadruped.simple_vision_playground import (
    QuadrupedVisionEnvConfig,
    QuadrupedVisionEnvironment,
)
from quadruped_mjx_rl.robots import RobotConfig

QuadrupedEnvFactory = Callable[[], type(QuadrupedBaseEnv)]


def get_base_model(init_scene_path: PathLike, env_config: EnvironmentConfig) -> EnvModel:
    base_model = load_to_model(init_scene_path)
    return resolve_env_class(env_config).customize_model(base_model, env_config)


def resolve_env_class(env_config: EnvironmentConfig) -> type(QuadrupedBaseEnv):
    return type(env_config).get_environment_class()


def get_env_factory(
    robot_config: RobotConfig,
    environment_config: EnvironmentConfig,
    env_model: EnvModel,
    *,
    env_class: type(QuadrupedBaseEnv) | None = None,
    **env_kwargs,
) -> QuadrupedEnvFactory:
    """
    Prepares parameters to instantiate an environment.
    """
    if env_class is None:
        env_class = resolve_env_class(environment_config)
    return functools.partial(
        env_class,
        robot_config=robot_config,
        environment_config=environment_config,
        env_spec=env_model,
        **env_kwargs,
    )
