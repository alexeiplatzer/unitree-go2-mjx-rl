from etils.epath import PathLike

from quadruped_mjx_rl.robots import RobotConfig
from quadruped_mjx_rl.environments.quadruped_base import (
    EnvironmentConfig,
    QuadrupedBaseEnv,
)
from quadruped_mjx_rl.environments.joystick_base import (
    JoystickBaseEnvConfig,
    QuadrupedJoystickBaseEnv,
)
from quadruped_mjx_rl.environments.joystick_teacher_student import (
    TeacherStudentEnvironmentConfig,
    QuadrupedJoystickTeacherStudentEnv,
)
from quadruped_mjx_rl.environments.simple_vision_playground import (
    QuadrupedVisionEnvConfig,
    QuadrupedVisionEnvironment,
)


def get_env_factory(
    robot_config: RobotConfig,
    init_scene_path: PathLike,
    env_config: EnvironmentConfig,
):
    def env_factory(**kwargs):
        return env_class(
            robot_config=robot_config,
            init_scene_path=init_scene_path,
            environment_config=env_config,
            **kwargs,
        )

    env_class = type(env_config).get_environment_class()
    uses_vision = isinstance(env_config, QuadrupedVisionEnvConfig) and env_config.use_vision
    return env_factory, uses_vision
