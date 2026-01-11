import logging

import jax
import numpy as np

import paths
from quadruped_mjx_rl.configs import prepare_all_configs
from quadruped_mjx_rl.environments import get_env_factory
from quadruped_mjx_rl.models import ActorCriticConfig
from quadruped_mjx_rl.models.io import load_params
from quadruped_mjx_rl.terrain_gen import make_terrain
from quadruped_mjx_rl.training import evaluate


if __name__ == "__main__":
    debug = not (jax.default_backend() == 'gpu')
    headless = True

    # Configure logging
    logging.basicConfig(level=logging.INFO, force=True)
    logging.info("Logging configured.")
    logging.info(f"Debugging: {debug}")

    # More legible printing from numpy.
    np.set_printoptions(precision=3, suppress=True, linewidth=100)

    # Prepare experiment results directory
    experiment_name = "target_reaching_2"
    experiment_dir = paths.EXPERIMENTS_DIRECTORY / "server_experiments" / experiment_name

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
        paths.ROBOT_CONFIGS_DIRECTORY / f"{robot_name}.yaml",
        paths.CONFIGS_DIRECTORY / f"target_reaching_basic.yaml",
    )
    assert isinstance(model_config, ActorCriticConfig)

    # Prepare environment model
    env_model = make_terrain(
        resources_directory=paths.RESOURCES_DIRECTORY,
        terrain_config=terrain_config,
        robot_config=robot_config,
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

    # Load params
    params = load_params(experiment_dir / "trained_policy")

    # Prepare the environment
    evaluation_env = env_factory()

    logging.info("Everything configured. Starting evaluation.")
    evaluate(
        training_config=training_config,
        model_config=model_config,
        evaluation_env=evaluation_env,
        randomization_config=terrain_config.randomization_config,
        show_outputs=not headless,
        run_in_cell=False,
        restore_params=params,
    )
    logging.info("Evaluation complete.")
