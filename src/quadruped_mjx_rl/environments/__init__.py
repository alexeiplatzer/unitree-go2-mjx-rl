from etils.epath import PathLike
from functools import partial

from quadruped_mjx_rl.robots import RobotConfig
from quadruped_mjx_rl.robotic_vision import VisionConfig
from quadruped_mjx_rl.environments.base import environment_config_classes
from quadruped_mjx_rl.environments.base import configs_to_env_classes
from quadruped_mjx_rl.environments.base import EnvironmentConfig
from quadruped_mjx_rl.environments.base import QuadrupedJoystickBaseEnv
from quadruped_mjx_rl.environments.ppo_enhanced import EnhancedEnvironmentConfig
from quadruped_mjx_rl.environments.ppo_enhanced import QuadrupedJoystickEnhancedEnv
from quadruped_mjx_rl.environments.ppo_teacher_student import TeacherStudentEnvironmentConfig
from quadruped_mjx_rl.environments.ppo_teacher_student import QuadrupedJoystickTeacherStudentEnv
from quadruped_mjx_rl.environments.simple_vision_playground import QuadrupedVisionEnvConfig
from quadruped_mjx_rl.environments.simple_vision_playground import QuadrupedVisionEnvironment


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

    env_class = configs_to_env_classes[type(env_config)]
    uses_vision = isinstance(env_config, QuadrupedVisionEnvConfig) and env_config.use_vision
    return env_factory, uses_vision
