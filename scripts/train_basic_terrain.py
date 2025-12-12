import logging
import sys

import numpy as np

import paths
from quadruped_mjx_rl.configs import prepare_all_configs
from quadruped_mjx_rl.domain_randomization.randomized_physics import domain_randomize
from quadruped_mjx_rl.environments import get_env_factory
from quadruped_mjx_rl.environments.physics_pipeline import load_to_model
from quadruped_mjx_rl.models import ActorCriticConfig
from quadruped_mjx_rl.models.io import save_params
from quadruped_mjx_rl.training.train_interface import train as train_ppo


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, force=True)
    logging.info("Logging configured.")

    # More legible printing from numpy.
    np.set_printoptions(precision=3, suppress=True, linewidth=100)

    # Prepare configs
    robot_name = "unitree_go2"
    config_file_path = (
        sys.argv[1]
        if len(sys.argv) > 1 else paths.CONFIGS_DIRECTORY / "joystick_basic_ppo_light.yaml"
    )
    # Instead of passing full path, user can pass just the name which is assumed to be
    # relative to the config directory.
    if not config_file_path.exists():
        config_file_path = paths.CONFIGS_DIRECTORY / config_file_path
        if not config_file_path.exists():
            raise FileNotFoundError(
                f"Config file {config_file_path} found neither with given path nor in the "
                f"config directory."
            )
    robot_config, env_config, model_config, training_config = prepare_all_configs(
        paths.ROBOT_CONFIGS_DIRECTORY / f"{robot_name}.yaml",
        config_file_path,
    )

    # Prepare environment model
    init_scene_path = paths.RESOURCES_DIRECTORY / robot_name / "scene_mjx.xml"
    env_model = load_to_model(init_scene_path)

    # Prepare the environment factory
    env_factory = get_env_factory(
        robot_config=robot_config,
        environment_config=env_config,
        env_model=env_model,
        customize_model=True,
    )

    logging.info("Everything configured. Starting training loop.")
    assert isinstance(model_config, ActorCriticConfig)
    trained_params = train_ppo(
        training_config=training_config,
        model_config=model_config,
        training_env=env_factory(),
        evaluation_env=env_factory(),
        randomization_fn=domain_randomize,
        run_in_cell=False,
        save_plots_path=paths.TRAINING_PLOTS_DIRECTORY,
    )
    save_params(
        params=trained_params, path=paths.TRAINED_POLICIES_DIRECTORY / "train_unitree_go2_raw"
    )
    logging.info("Training complete. Params saved.")
