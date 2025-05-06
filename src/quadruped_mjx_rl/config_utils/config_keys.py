from enum import Enum

from quadruped_mjx_rl.environments import environment_config_classes
from quadruped_mjx_rl.environments import EnvironmentConfig
from quadruped_mjx_rl.models import model_config_classes
from quadruped_mjx_rl.models import ModelConfig
from quadruped_mjx_rl.rendering import RenderConfig
from quadruped_mjx_rl.rendering import rendering_config_classes
from quadruped_mjx_rl.robots import robot_config_classes
from quadruped_mjx_rl.robots import RobotConfig
from quadruped_mjx_rl.training import training_config_classes
from quadruped_mjx_rl.training import TrainingConfig
from quadruped_mjx_rl.robotic_vision import vision_config_classes
from quadruped_mjx_rl.robotic_vision import VisionConfig

AnyConfig = (
    EnvironmentConfig | RobotConfig | ModelConfig | TrainingConfig | RenderConfig | VisionConfig
)


class ConfigKey(str, Enum):
    ENVIRONMENT = "environment"
    ROBOT = "robot"
    MODEL = "model"
    TRAINING = "training"
    RENDERING = "rendering"
    VISION = "vision"


def config_class_to_key(config_class: type[AnyConfig]) -> ConfigKey:
    for supertype in config_class.__mro__:
        if supertype in base_class_to_key:
            return base_class_to_key[supertype]
    raise KeyError(f"Could not find config key for {config_class}")


base_class_to_key = {
    EnvironmentConfig: ConfigKey.ENVIRONMENT,
    RobotConfig: ConfigKey.ROBOT,
    ModelConfig: ConfigKey.MODEL,
    TrainingConfig: ConfigKey.TRAINING,
    RenderConfig: ConfigKey.RENDERING,
    VisionConfig: ConfigKey.VISION,
}

key_to_resolver = {
    ConfigKey.ENVIRONMENT: environment_config_classes,
    ConfigKey.ROBOT: robot_config_classes,
    ConfigKey.MODEL: model_config_classes,
    ConfigKey.TRAINING: training_config_classes,
    ConfigKey.RENDERING: rendering_config_classes,
    ConfigKey.VISION: vision_config_classes,
}
