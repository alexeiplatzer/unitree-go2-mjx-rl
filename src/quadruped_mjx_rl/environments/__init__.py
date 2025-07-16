import functools
from etils.epath import PathLike

from quadruped_mjx_rl.robotic_vision import VisionConfig, get_renderer

from quadruped_mjx_rl.robots import RobotConfig
from quadruped_mjx_rl.environments.quadruped.base import (
    EnvironmentConfig,
    QuadrupedBaseEnv,
)
from quadruped_mjx_rl.environments.quadruped.joystick_base import (
    JoystickBaseEnvConfig,
    QuadrupedJoystickBaseEnv,
)
from quadruped_mjx_rl.environments.quadruped.joystick_teacher_student import (
    TeacherStudentEnvironmentConfig,
    QuadrupedJoystickTeacherStudentEnv,
)
from quadruped_mjx_rl.environments.quadruped.simple_vision_playground import (
    QuadrupedVisionEnvConfig,
    QuadrupedVisionEnvironment,
)


def get_env_factory(
    robot_config: RobotConfig,
    init_scene_path: PathLike,
    env_config: EnvironmentConfig,
    vision_config: VisionConfig
):
    def env_factory(**kwargs):
        return env_class(
            robot_config=robot_config,
            init_scene_path=init_scene_path,
            environment_config=env_config,
            **kwargs,
        )

    env_class = type(env_config).get_environment_class()
    sys = env_class.make_system(init_scene_path=init_scene_path, environment_config=env_config)
    uses_vision = False
    if isinstance(env_config, QuadrupedVisionEnvConfig) and env_config.use_vision:
        uses_vision = True
        batch_renderer = get_renderer(sys=sys, vision_config=vision_config)
        env_factory = functools.partial(env_factory, batch_renderer=batch_renderer)
    return env_factory, uses_vision
