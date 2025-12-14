from etils.epath import PathLike

from quadruped_mjx_rl.terrain_gen import TerrainConfig
from quadruped_mjx_rl.environments import EnvironmentConfig
from quadruped_mjx_rl.robots import RobotConfig
from quadruped_mjx_rl.models import ModelConfig
from quadruped_mjx_rl.training import TrainingConfig
from quadruped_mjx_rl.config_utils import prepare_configs


def prepare_all_configs(
    *configs_files_paths: PathLike,
) -> tuple[RobotConfig, TerrainConfig, EnvironmentConfig, ModelConfig, TrainingConfig]:
    """Prepares the dataclasses representing all the main configs used by this package"""
    configs = prepare_configs(*configs_files_paths)
    robot_config = configs["robot"]
    assert isinstance(robot_config, RobotConfig)
    terrain_config = configs["terrain"]
    assert isinstance(terrain_config, TerrainConfig)
    environment_config = configs["environment"]
    assert isinstance(environment_config, EnvironmentConfig)
    model_config = configs["model"]
    assert isinstance(model_config, ModelConfig)
    training_config = configs["training"]
    assert isinstance(training_config, TrainingConfig)
    return robot_config, terrain_config, environment_config, model_config, training_config
