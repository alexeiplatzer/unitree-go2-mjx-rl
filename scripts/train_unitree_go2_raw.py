import logging

import numpy as np

import paths
from quadruped_mjx_rl.config_utils import prepare_configs
from quadruped_mjx_rl.domain_randomization.randomized_physics import domain_randomize
from quadruped_mjx_rl.environments import get_env_factory
from quadruped_mjx_rl.environments.physics_pipeline import load_to_spec, spec_to_model
from quadruped_mjx_rl.training.train_interface import train
from quadruped_mjx_rl.models.io import save_params

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, force=True)
    logging.info("Logging configured.")

    # More legible printing from numpy.
    np.set_printoptions(precision=3, suppress=True, linewidth=100)

    # Prepare configs
    configs = prepare_configs(
        paths.CONFIGS_DIRECTORY / "unitree_go2.yaml",
        paths.CONFIGS_DIRECTORY / "basic_lightweight_example.yaml",
    )
    robot_config = configs.get("robot")
    env_config = configs.get("environment")
    model_config = configs.get("model")
    training_config = configs.get("training")

    # Prepare environment model
    init_scene_path = paths.unitree_go2_init_scene
    env_model = spec_to_model(load_to_spec(init_scene_path))

    # Prepare the environment factory
    env_factory = get_env_factory(
        robot_config=robot_config,
        environment_config=env_config,
        env_model=env_model,
        customize_model=True,
    )

    logging.info("Everything configured. Starting training loop.")
    trained_params = train(
        training_config=training_config,
        model_config=model_config,
        training_env=env_factory(),
        evaluation_env=env_factory(),
        randomization_fn=domain_randomize,
        run_in_cell=False,
        save_plots_path=paths.TRAINING_PLOTS_DIRECTORY / "simple_training_plot",
    )
    save_params(
        params=trained_params, path=paths.TRAINED_POLICIES_DIRECTORY / "train_unitree_go2_raw"
    )
    logging.info("Training complete. Params saved.")
