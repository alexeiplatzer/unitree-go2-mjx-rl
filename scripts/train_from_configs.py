import logging
import sys
from pathlib import Path

import numpy as np
import jax

import paths
from quadruped_mjx_rl.configs import prepare_all_configs
from quadruped_mjx_rl.environments import get_env_factory
from quadruped_mjx_rl.environments.rendering import (
    render_model,
    save_image,
)
from quadruped_mjx_rl.models import ActorCriticConfig
from quadruped_mjx_rl.models.io import load_params, save_params
from quadruped_mjx_rl.terrain_gen import make_terrain
from quadruped_mjx_rl.training import TrainingConfig
from quadruped_mjx_rl.training.train_interface import train as train_ppo


def make_model_hyperparams_lighter(model_cfg: ActorCriticConfig) -> None:
    model_cfg.policy.layer_sizes = [
        model_cfg.policy.layer_sizes[0],
        model_cfg.policy.layer_sizes[-1],
    ]
    model_cfg.value.layer_sizes = [
        model_cfg.value.layer_sizes[0],
        model_cfg.value.layer_sizes[-1],
    ]


def make_training_hyperparams_lighter(training_cfg: TrainingConfig) -> None:
    training_cfg.num_timesteps = 100_000
    training_cfg.num_envs = 4
    training_cfg.num_eval_envs = 4
    training_cfg.batch_size = 4
    training_cfg.num_minibatches = 1
    training_cfg.num_evals = 5


if __name__ == "__main__":
    debug = not (jax.default_backend() == 'gpu')
    headless = False

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
    logging.info(f"Created directory for experiments results: {experiment_dir}.")

    # Prepare used files
    used_file_paths = (
        [Path(sys.argv[i]) for i in range(1, len(sys.argv))]
        if len(sys.argv) > 1
        else [paths.CONFIGS_DIRECTORY / "joystick_basic.yaml"]
    )
    pretrained_params_path = None
    config_file_paths = []
    for used_file_path in used_file_paths:
        if used_file_path.suffix == ".yaml":
            # Instead of passing full path, user can pass just the name which is assumed to be
            # relative to the config directory.
            if used_file_path.exists():
                config_file_paths.append(used_file_path)
            elif (paths.CONFIGS_DIRECTORY / used_file_path).exists():
                config_file_paths.append(paths.CONFIGS_DIRECTORY / used_file_path)
            elif (paths.MODEL_CONFIGS_DIRECTORY / used_file_path).exists():
                config_file_paths.append(paths.MODEL_CONFIGS_DIRECTORY / used_file_path)
            elif (paths.ENVIRONMENT_CONFIGS_DIRECTORY / used_file_path).exists():
                config_file_paths.append(paths.ENVIRONMENT_CONFIGS_DIRECTORY / used_file_path)
            else:
                raise FileNotFoundError(
                    f"Config file {used_file_path} found neither with given path nor in the "
                    f"config directory."
                )
        else:
            probable_params_path = paths.EXPERIMENTS_DIRECTORY / used_file_path / "trained_policy"
            if probable_params_path.exists():
                pretrained_params_path = probable_params_path
                logging.info(f"Pretrained params found under path: {pretrained_params_path}.")
    logging.info(f"Using configs: {config_file_paths}.")

    # Prepare configs
    robot_name = "unitree_go2"
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
    assert isinstance(model_config, ActorCriticConfig)
    if debug:
        make_model_hyperparams_lighter(model_config)
        make_training_hyperparams_lighter(training_config)

    # Prepare environment model
    env_model = make_terrain(
        resources_directory=paths.RESOURCES_DIRECTORY,
        terrain_config=terrain_config,
        robot_config=robot_config,
    )

    if not headless:
        # Render the environment model
        for camera_name in [*terrain_config.visualization_cameras, "track"] :
            image = render_model(
                env_model,
                initial_keyframe=robot_config.initial_keyframe,
                camera=camera_name,
            )
            save_image(
                image=image, save_path=experiment_dir / f"environment_view_{camera_name}"
            )

    # Prepare the environment factory
    renderer_maker = (
        training_config.get_renderer_factory(gpu_id=0, debug=debug)
        if model_config.vision
        else None
    )
    env_factory = get_env_factory(
        robot_config=robot_config,
        environment_config=env_config,
        env_model=env_model,
        customize_model=True,
        vision_wrapper_config=vision_wrapper_config if model_config.vision else None,
        renderer_maker=renderer_maker,
    )

    # Restore params
    restored_params = load_params(str(pretrained_params_path)) if pretrained_params_path else None

    # Prepare everything for the training function
    training_env = env_factory()
    evaluation_env = env_factory() if not model_config.vision else None
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
        restore_params=restored_params,
    )
    save_params(params=trained_params, path=experiment_dir / "trained_policy")
    logging.info("Training complete. Params saved.")
