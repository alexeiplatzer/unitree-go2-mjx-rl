from etils.epath import PathLike

from dataclasses import asdict, dataclass
from enum import Enum
import yaml

from .utils import conditionally_instantiate, load_config_dicts, load_configs_from_dicts

# One config module to rule them all
# One config module to find them
# One config module to bring them all
# And in this namespace bind them
from .robots import RobotConfig, name_to_robot
from .environments import EnvironmentConfig, name_to_environment, name_to_environment_class
from .models import ModelConfig, name_to_model
from .training import TrainingConfig, name_to_training_config
from .rendering import RenderConfig, name_to_rendering_config


class ConfigKey(str, Enum):
    ENVIRONMENT = "environment"
    ROBOT = "robot"
    MODEL = "model"
    TRAINING = "training"
    RENDERING = "rendering"


AnyConfig = EnvironmentConfig | RobotConfig | ModelConfig | TrainingConfig | RenderConfig

config_to_type = {
    ConfigKey.ENVIRONMENT: EnvironmentConfig,
    ConfigKey.ROBOT: RobotConfig,
    ConfigKey.MODEL: ModelConfig,
    ConfigKey.TRAINING: TrainingConfig,
    ConfigKey.RENDERING: RenderConfig,
}

config_to_resolver = {
    ConfigKey.ENVIRONMENT: name_to_environment,
    ConfigKey.ROBOT: name_to_robot,
    ConfigKey.MODEL: name_to_model,
    ConfigKey.TRAINING: name_to_training_config,
    ConfigKey.RENDERING: name_to_rendering_config,
}


def prepare_configs(
    *config_paths: PathLike,
    **configs: AnyConfig | str | None,
) -> dict[ConfigKey, AnyConfig | None]:
    loaded_dicts = load_config_dicts(*config_paths)
    loaded_configs = load_configs_from_dicts(list(config_to_type.keys()), *loaded_dicts)
    final_configs = {}
    for config_key in config_to_type.keys():
        final_configs[config_key] = conditionally_instantiate(
            config_to_resolver[config_key],
            configs.get(config_key),
            loaded_configs[config_key],
        )
    return final_configs


def save_configs(
    save_file_path: PathLike,
    *configs: AnyConfig,
):
    type_to_key = {v: k for k, v in config_to_type.items()}
    final_dict = {type_to_key[type(config)].value(): asdict(config) for config in configs}
    with open(save_file_path, "w") as f:
        yaml.dump(final_dict, f)
