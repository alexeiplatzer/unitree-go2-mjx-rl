import logging
import sys

import numpy as np

import paths
from quadruped_mjx_rl.configs import prepare_all_configs
from quadruped_mjx_rl.domain_randomization.randomized_physics import domain_randomize
from quadruped_mjx_rl.environments import get_env_factory
from quadruped_mjx_rl.environments.physics_pipeline import load_to_spec, spec_to_model
from quadruped_mjx_rl.environments.rendering import show_image, render_model, large_overview_camera, save_image
from quadruped_mjx_rl.models import ActorCriticConfig
from quadruped_mjx_rl.models.io import save_params
from quadruped_mjx_rl.terrain_gen.obstacles import FlatTile, StripesTile
from quadruped_mjx_rl.terrain_gen.tile import TerrainConfig
from quadruped_mjx_rl.training.train_interface import train as train_ppo


if __name__ == "__main__":
    # Set this to True if you want to see the outputs during the run
    show_outputs = False

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
    init_scene_path = paths.RESOURCES_DIRECTORY / robot_name / "scene_mjx_empty_arena.xml"
    env_spec = load_to_spec(init_scene_path)
    flat_tile = FlatTile()
    stripes_tile = StripesTile()
    terrain = TerrainConfig(tiles=[[flat_tile] * 6])
    logging.info(f"Terrain: {terrain.tiles}")
    terrain.make_arena(env_spec)
    env_model = spec_to_model(env_spec)

    # Render the environment model
    image = render_model(
        env_model,
        initial_keyframe=robot_config.initial_keyframe,
        camera=large_overview_camera(),
    )
    save_image(image=image, save_path=experiment_dir / "environment_view")
    if show_outputs:
        show_image(image)

    # Prepare the environment factory
    env_factory = get_env_factory(
        robot_config=robot_config,
        environment_config=env_config,
        env_model=env_model,
        customize_model=True,
    )

    logging.info("Everything configured. Starting training loop.")
    assert isinstance(model_config, ActorCriticConfig)
    training_plots_dir = experiment_dir / "training_plots"
    training_plots_dir.mkdir()
    trained_params = train_ppo(
        training_config=training_config,
        model_config=model_config,
        training_env=env_factory(),
        evaluation_env=env_factory(),
        randomization_fn=domain_randomize,
        run_in_cell=False,
        save_plots_path=training_plots_dir,
    )
    save_params(
        params=trained_params, path=experiment_dir / "trained_policy"
    )
    logging.info("Training complete. Params saved.")
