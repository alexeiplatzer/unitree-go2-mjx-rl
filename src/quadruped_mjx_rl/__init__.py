from etils.epath import PathLike
from collections.abc import Callable

from .config_utils import EnvironmentConfig
from .config_utils import RobotConfig
from .config_utils import ModelConfig
from .config_utils import TrainingConfig
from .config_utils import RenderConfig
from .config_utils import ConfigKey, prepare_configs
from .utils import load_config_dicts
from .environments import name_to_environment_class

from .rendering import render as render_from_configs
from .training import train as train_from_configs
from .training import name_to_training_fn


def render(
    *config_paths: PathLike,
    init_scene_path: PathLike,
    trained_model_path: PathLike,
    animation_save_path: PathLike | dict[str, PathLike] | None,
    environment_config: EnvironmentConfig | str | None = None,
    robot_config: RobotConfig | str | None = None,
    model_config: ModelConfig | str | None = None,
    render_config: RenderConfig | str | None = None,
):
    """
    Render a simulation rollout using configurations provided via YAML files and/or direct
    dataclass instances.

    All necessary configurations must be specified either by a dataclass config instance, or
    defined at the top level in some yaml file (e.g. A "robot" top level key in a yaml file for
    the robot config). A config dataclass from the configs module might be referred to by name.

    Parameters:
        *config_paths (PathLike): Variable number of file paths to YAML configuration files.
        init_scene_path (PathLike): File path to the initial scene configuration.
        trained_model_path (PathLike): File path to the saved parameters of the trained model.
        animation_save_path (PathLike or None):
            File path to save the animation, if None, outputs the animation directly
        environment_config (EnvironmentConfig or str, optional):
            An instance of EnvironmentConfig or the name of a predefined EnvironmentConfig
            dataclass.
        robot_config (RobotConfig or str, optional):
            An instance of RobotConfig or the name of a predefined RobotConfig dataclass.
        model_config (ModelConfig or str, optional):
            An instance of ModelConfig or the name of a predefined ModelConfig dataclass.
        render_config (RenderConfig or str, optional):
            An instance of RenderConfig or the name of a predefined RenderConfig dataclass.

    Returns:
        None
    """

    required_configs = [
        ConfigKey.ENVIRONMENT,
        ConfigKey.ROBOT,
        ConfigKey.MODEL,
        ConfigKey.RENDERING,
    ]

    final_configs = prepare_configs(
        *config_paths,
        environment_config=environment_config,
        robot_config=robot_config,
        model_config=model_config,
        render_config=render_config,
    )

    # Check that all required configs are present
    for required_config in required_configs:
        if required_config not in final_configs or final_configs[required_config] is None:
            raise ValueError(f"Config for {required_config} was not found!")

    environment = name_to_environment_class[final_configs[ConfigKey.ENVIRONMENT].name]

    # Call the original render function with the resolved configurations.
    render_from_configs(
        environment=environment,
        env_config=final_configs[ConfigKey.ENVIRONMENT],
        robot_config=final_configs[ConfigKey.ROBOT],
        init_scene_path=init_scene_path,
        model_config=final_configs[ConfigKey.MODEL],
        trained_model_path=trained_model_path,
        render_config=final_configs[ConfigKey.RENDERING],
        animation_save_path=animation_save_path,
    )


def train(
    *config_paths: PathLike,
    init_scene_path: PathLike,
    model_save_path: PathLike,
    environment_config: EnvironmentConfig | str | None = None,
    robot_config: RobotConfig | str | None = None,
    model_config: ModelConfig | str | None = None,
    training_config: TrainingConfig | str | None = None,
):
    required_configs = [
        ConfigKey.ENVIRONMENT,
        ConfigKey.ROBOT,
        ConfigKey.MODEL,
        ConfigKey.TRAINING,
    ]

    final_configs = prepare_configs(
        *config_paths,
        environment_config=environment_config,
        robot_config=robot_config,
        model_config=model_config,
        training_config=training_config,
    )

    # Check that all required configs are present
    for required_config in required_configs:
        if required_config not in final_configs or final_configs[required_config] is None:
            raise ValueError(f"Config for {required_config} was not found!")

    environment = name_to_environment_class[final_configs[ConfigKey.ENVIRONMENT].name]

    training_fn = name_to_training_fn[final_configs[ConfigKey.TRAINING].name]

    # Call the original train function with the resolved configurations.
    train_from_configs(
        environment=environment,
        env_config=final_configs[ConfigKey.ENVIRONMENT],
        robot_config=final_configs[ConfigKey.ROBOT],
        init_scene_path=init_scene_path,
        model_config=final_configs[ConfigKey.MODEL],
        make_networks_fn=None,
        model_save_path=model_save_path,
        training_config=training_config,
        train_fn=training_fn,
    )
