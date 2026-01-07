import functools
import logging

import numpy as np
import mujoco

import paths
from quadruped_mjx_rl.configs import prepare_all_configs
from quadruped_mjx_rl.environments import get_env_factory
from quadruped_mjx_rl.environments.rendering import (
    large_overview_camera,
    render_model,
    save_image,
    render_vision_observations
)
from quadruped_mjx_rl.robotic_vision import get_renderer
from quadruped_mjx_rl.models.io import load_params
from quadruped_mjx_rl.policy_rendering import render_policy_rollout, RenderConfig, save_video
from quadruped_mjx_rl.terrain_gen import make_terrain
from quadruped_mjx_rl.training import TrainingWithVisionConfig
from quadruped_mjx_rl import robots, terrain_gen, environments, training


if __name__ == "__main__":
    debug = False
    version = "joystick"
    seed = 0

    # Configure logging
    logging.basicConfig(level=logging.INFO, force=True)
    logging.info("Logging configured.")

    # More legible printing from numpy.
    np.set_printoptions(precision=3, suppress=True, linewidth=100)

    # Prepare the directory
    pictures_dir = paths.EXPERIMENTS_DIRECTORY / "terrain_pictures"
    pictures_dir.mkdir(exist_ok=True)

    # Prepare configs
    robot_config = robots.unitree_go2_config()
    if version == "joystick":
        terrain_config = terrain_gen.ColorMapTerrainConfig(add_goal=False)
        env_config = environments.JoystickBaseEnvConfig()
    else:
        terrain_config = terrain_gen.ColorMapTerrainConfig(add_goal=True)
        env_config = environments.QuadrupedVisionTargetEnvConfig()
    vision_wrapper_config = environments.ColorGuidedEnvConfig()
    training_config = training.TrainingWithVisionConfig()

    # Prepare environment model
    env_model = make_terrain(
        resources_directory=paths.RESOURCES_DIRECTORY,
        terrain_config=terrain_config,
        robot_config=robot_config,
    )

    # Prepare the environment factory
    renderer_maker = training_config.get_renderer_factory(gpu_id=0, debug=debug)
    env_factory = get_env_factory(
        robot_config=robot_config,
        environment_config=env_config,
        env_model=env_model,
        customize_model=True,
        vision_wrapper_config=vision_wrapper_config,
        renderer_maker=renderer_maker,
    )
    logging.info("Environment factory prepared.")

    # Prepare the environment
    env = env_factory()
    logging.info("Environment prepared.")

    images = render_vision_observations(
        env=env,
        seed=seed,
        domain_rand_config=terrain_config.randomization_config,
        num_worlds=training_config.num_envs,
    )
    logging.info(f"Vision observations rendered: {images.keys()}")
    for key, image in images.items():
        safe_name = key.replace("/", "_")
        if key == "privileged_terrain_map":
            # Channel 0: Friction
            save_image(image[..., 0], pictures_dir / f"vision_obs_{safe_name}_friction.png")
            # Channel 1: Stiffness
            save_image(image[..., 1], pictures_dir / f"vision_obs_{safe_name}_stiffness.png")
        else:
            save_image(image, pictures_dir / f"vision_obs_{safe_name}.png")
