import functools
import logging
import sys
from pathlib import Path

import numpy as np

import paths
from quadruped_mjx_rl.configs import prepare_all_configs
from quadruped_mjx_rl.robotic_vision import get_renderer
from quadruped_mjx_rl.environments import get_env_factory
from quadruped_mjx_rl.environments.rendering import (
    large_overview_camera,
    render_model,
    save_image,
)
from quadruped_mjx_rl.models import ActorCriticConfig
from quadruped_mjx_rl.models.io import save_params
from quadruped_mjx_rl.terrain_gen import make_terrain
from quadruped_mjx_rl.training import TrainingWithVisionConfig
from quadruped_mjx_rl.training.train_interface import train as train_ppo


if __name__ == "__main__":
    debug = False
    headless = True

    # Configure logging
    logging.basicConfig(level=logging.INFO, force=True)
    logging.info("Logging configured.")

    # More legible printing from numpy.
    np.set_printoptions(precision=3, suppress=True, linewidth=100)

    # Prepare experiment results directory
    existing_numbers = []
    for experiment_path in paths.EXPERIMENTS_DIRECTORY.glob("experiment_*"):
        experiment_index = experiment_path.name.split("_")[1]
        if experiment_index.isdigit():
            existing_numbers.append(int(experiment_index))
    next_number = max(existing_numbers) + 1 if existing_numbers else 0
    experiment_dir = paths.EXPERIMENTS_DIRECTORY / f"experiment_{next_number}"
    experiment_dir.mkdir()

    # Prepare configs
    robot_name = "unitree_go2"
    config_file_paths = (
        [Path(sys.argv[i]) for i in range(1, len(sys.argv))]
        if len(sys.argv) > 1
        else [paths.CONFIGS_DIRECTORY / "joystick_basic_ppo_light.yaml"]
    )
    # Instead of passing full path, user can pass just the name which is assumed to be
    # relative to the config directory.
    config_file_paths = [
        (
            config_file_path
            if config_file_path.exists()
            else paths.MODEL_CONFIGS_DIRECTORY / config_file_path
            if (paths.MODEL_CONFIGS_DIRECTORY / config_file_path).exists()
            else paths.ENVIRONMENT_CONFIGS_DIRECTORY / config_file_path
        )
        for config_file_path in config_file_paths
    ]
    for config_file_path in config_file_paths:
        if not config_file_path.exists():
            raise FileNotFoundError(
                f"Config file {config_file_path} found neither with given path nor in the "
                f"config directory."
            )
    (
        robot_config,
        terrain_config,
        env_config,
        vision_wrapper_config,
        model_config,
        training_config,
    ) = prepare_all_configs(
        paths.ROBOT_CONFIGS_DIRECTORY / f"{robot_name}.yaml", *config_file_paths
    )

    # Prepare environment model
    env_model = make_terrain(
        resources_directory=paths.RESOURCES_DIRECTORY,
        terrain_config=terrain_config,
        robot_config=robot_config,
    )

    if not headless:
        # Render the environment model
        for camera_name in terrain_config.visualization_cameras:
            image = render_model(
                env_model,
                initial_keyframe=robot_config.initial_keyframe,
                camera=camera_name,
            )
            save_image(
                image=image, save_path=experiment_dir / f"environment_view_{camera_name}"
            )

    # Prepare the environment factory
    vision = isinstance(training_config, TrainingWithVisionConfig)
    renderer_maker = (
        training_config.get_renderer_factory(gpu_id=0, debug=debug)
        if isinstance(training_config, TrainingWithVisionConfig)
        else None
    )
    env_factory = get_env_factory(
        robot_config=robot_config,
        environment_config=env_config,
        env_model=env_model,
        customize_model=True,
        vision_wrapper_config=vision_wrapper_config if vision else None,
        renderer_maker=renderer_maker,
    )

    # Prepare everything for the training function
    assert isinstance(model_config, ActorCriticConfig)
    training_env = env_factory()
    evaluation_env = env_factory() if not vision else None
    training_plots_dir = experiment_dir / "training_plots"
    training_plots_dir.mkdir()

    logging.info("Everything configured. Starting training loop.")
    trained_params = train_ppo(
        training_config=training_config,
        model_config=model_config,
        training_env=training_env,
        evaluation_env=evaluation_env,
        randomization_config=terrain_config.randomization_config,
        show_outputs=not headless,
        run_in_cell=False,
        save_plots_path=training_plots_dir,
    )
    save_params(params=trained_params, path=experiment_dir / "trained_policy")
    logging.info("Training complete. Params saved.")
