from enum import Enum

# One config module to rule them all
# One config module to find them
# One config module to bring them all
# And in this namespace bind them
from quadruped_mjx_rl.configs.config_classes import RobotConfig
from quadruped_mjx_rl.configs.config_classes import ModelConfig
from quadruped_mjx_rl.configs.config_classes import EnvironmentConfig
from quadruped_mjx_rl.configs.config_classes import TrainingConfig
from quadruped_mjx_rl.configs.config_classes import RenderConfig
from quadruped_mjx_rl.configs.config_classes import VisionConfig

from quadruped_mjx_rl.configs.config_classes import robot_config_classes
from quadruped_mjx_rl.configs.config_classes import model_config_classes
from quadruped_mjx_rl.configs.config_classes import environment_config_classes
from quadruped_mjx_rl.configs.config_classes import training_config_classes
from quadruped_mjx_rl.configs.config_classes import rendering_config_classes
from quadruped_mjx_rl.configs.config_classes import vision_config_classes


AnyConfig = (
    EnvironmentConfig | RobotConfig | ModelConfig | TrainingConfig | RenderConfig | VisionConfig
)


def _invert_dict(x: dict) -> dict:
    return {v: k for k, v in x.items()}


class ConfigKey(str, Enum):
    ENVIRONMENT = "environment"
    ROBOT = "robot"
    MODEL = "model"
    TRAINING = "training"
    RENDERING = "rendering"
    VISION = "vision"

    def to_config_base_class(self) -> type[AnyConfig]:
        return config_base_classes[self]

    @staticmethod
    def from_config_base_class(config_class: type[AnyConfig]) -> "ConfigKey":
        inv_dict = _invert_dict(config_base_classes)
        try:
            return inv_dict[config_class]
        except KeyError:
            for supertype in config_class.__mro__:
                if supertype in inv_dict:
                    return inv_dict[supertype]
            raise KeyError(f"Could not find config key for {config_class}")

    def to_config_class_resolver(self) -> dict[str, type[AnyConfig]]:
        return config_class_resolvers[self]


config_base_classes = {
    ConfigKey.ENVIRONMENT: EnvironmentConfig,
    ConfigKey.ROBOT: RobotConfig,
    ConfigKey.MODEL: ModelConfig,
    ConfigKey.TRAINING: TrainingConfig,
    ConfigKey.RENDERING: RenderConfig,
    ConfigKey.VISION: VisionConfig,
}

config_class_resolvers = {
    ConfigKey.ENVIRONMENT: environment_config_classes,
    ConfigKey.ROBOT: robot_config_classes,
    ConfigKey.MODEL: model_config_classes,
    ConfigKey.TRAINING: training_config_classes,
    ConfigKey.RENDERING: rendering_config_classes,
    ConfigKey.VISION: vision_config_classes,
}
