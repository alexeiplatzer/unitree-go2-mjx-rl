from .configs import EnvironmentConfig

# defined environment imports
from .ppo_simple import JoystickEnv, SimpleEnvironmentConfig
from .go2_teacher import Go2TeacherEnv

name_to_environment_class = {
    "joystick": JoystickEnv,
    "go2_teacher": Go2TeacherEnv,
}

name_to_environment = {
    "joystick": lambda: SimpleEnvironmentConfig(name="joystick"),
}
