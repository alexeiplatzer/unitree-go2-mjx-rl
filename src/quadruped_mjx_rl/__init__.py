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
from .models import get_model_factories_by_name

from .rendering import render as render_from_configs


def render(
    *config_paths: PathLike,
    init_scene_path: PathLike,
    trained_model_path: PathLike,
    animation_save_path: PathLike | None,
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

    networks_factory, inference_factory = get_model_factories_by_name(
        final_configs[ConfigKey.MODEL].name
    )

    # Call the original render function with the resolved configurations.
    render_from_configs(
        environment=environment,
        env_config=final_configs[ConfigKey.ENVIRONMENT],
        robot_config=final_configs[ConfigKey.ROBOT],
        init_scene_path=init_scene_path,
        model_config=final_configs[ConfigKey.MODEL],
        make_inference_fn=inference_factory,
        make_networks_fn=networks_factory,
        trained_model_path=trained_model_path,
        render_config=final_configs[ConfigKey.RENDERING],
        animation_save_path=animation_save_path,
    )


def train(
    *config_paths: PathLike,
    init_scene_path: PathLike,
    trained_model_path: PathLike,
    animation_save_path: PathLike | None,
    environment_config: EnvironmentConfig | str | None = None,
    robot_config: RobotConfig | str | None = None,
    model_config: ModelConfig | str | None = None,
    training_config: TrainingConfig | str | None = None,
):
    pass
