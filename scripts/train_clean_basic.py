import numpy as np
import paths

from quadruped_mjx_rl.robots import predefined_robot_configs
from quadruped_mjx_rl.environments import JoystickBaseEnvConfig
from quadruped_mjx_rl.models import ActorCriticConfig, ModuleConfigMLP
from quadruped_mjx_rl.training.configs import TrainingConfig
from quadruped_mjx_rl.environments.physics_pipeline import load_to_spec, spec_to_model
from quadruped_mjx_rl.environments import get_env_factory
from quadruped_mjx_rl.training.train_interface import train
from quadruped_mjx_rl.domain_randomization.randomized_physics import domain_randomize
from quadruped_mjx_rl.environments.rendering import (
    large_overview_camera, render_model, show_image
)
import logging

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, force=True)
    logging.info("Logging configured.")

    # More legible printing from numpy.
    np.set_printoptions(precision=3, suppress=True, linewidth=100)

    # Prepare configs
    robot_config = predefined_robot_configs["unitree_go2"]()

    env_config = JoystickBaseEnvConfig()

    model_config = ActorCriticConfig(
        policy=ModuleConfigMLP(layer_sizes=[128, 128]),
        value=ModuleConfigMLP(layer_sizes=[256, 256]),
    )

    training_config = TrainingConfig(num_timesteps=100_000, num_envs=8, num_eval_envs=8)

    init_scene_path = paths.unitree_go2_init_scene

    env_model = spec_to_model(load_to_spec(init_scene_path))

    # Render the situation
    image = render_model(
        env_model=env_model,
        initial_keyframe=robot_config.initial_keyframe,
        camera=large_overview_camera(),
    )
    show_image(image)

    env_factory = get_env_factory(
        robot_config=robot_config,
        environment_config=env_config,
        env_model=env_model,
        customize_model=True,
    )

    trained_params = train(
        training_config=training_config,
        model_config=model_config,
        training_env=env_factory(),
        evaluation_env=env_factory(),
        randomization_fn=domain_randomize,
        run_in_cell=False,
        save_plots_path=paths.ROLLOUTS_DIRECTORY / "simple_training_plot",
    )
