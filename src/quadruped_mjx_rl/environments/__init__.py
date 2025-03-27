from .configs import EnvironmentConfig, VisionConfig

# defined environment imports
from .ppo_simple import JoystickEnv, SimpleEnvironmentConfig
from .go2_teacher import Go2TeacherEnv, EnhancedEnvironmentConfig
from .simple_vision_playground import QuadrupedVisionEnvironment, QuadrupedVisionEnvConfig

name_to_environment_class = {
    "joystick": JoystickEnv,
    "go2_teacher": Go2TeacherEnv,
}

name_to_environment = {
    "joystick": SimpleEnvironmentConfig,
    "go2_teacher": EnhancedEnvironmentConfig,
}
