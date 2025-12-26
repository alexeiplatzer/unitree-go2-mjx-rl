import functools
import logging

import numpy as np

import paths
from quadruped_mjx_rl.configs import prepare_all_configs
from quadruped_mjx_rl.environments import get_env_factory
from quadruped_mjx_rl.environments.rendering import (
    large_overview_camera,
    render_model,
    save_image,
)
from quadruped_mjx_rl.environments.vision.robotic_vision import get_renderer
from quadruped_mjx_rl.models.io import load_params
from quadruped_mjx_rl.policy_rendering import render_policy_rollout, RenderConfig, save_video
from quadruped_mjx_rl.terrain_gen import make_terrain
from quadruped_mjx_rl.training import TrainingWithVisionConfig


if __name__ == "__main__":
    debug = True

    # Configure logging
    logging.basicConfig(level=logging.INFO, force=True)
    logging.info("Logging configured.")

    # More legible printing from numpy.
    np.set_printoptions(precision=3, suppress=True, linewidth=100)

    # Prepare experiment results directory
    experiment_name = "randomized_tiles_blind"
    experiment_dir = paths.EXPERIMENTS_DIRECTORY / "server_experiments" / experiment_name

    # Prepare configs
    robot_name = "unitree_go2"
    robot_config, terrain_config, env_config, model_config, training_config = (
        prepare_all_configs(
            paths.ROBOT_CONFIGS_DIRECTORY / f"{robot_name}.yaml",
            paths.CONFIGS_DIRECTORY / f"{experiment_name}.yaml",
        )
    )
    render_config = RenderConfig(
        episode_length=2000, n_steps=1000, cameras=["track", "ego_frontal", "privileged"]
    )

    # Prepare environment model
    init_scene_path = paths.RESOURCES_DIRECTORY / robot_name / terrain_config.base_scene_file
    env_model = make_terrain(init_scene_path, terrain_config)

    # Render the environment model
    image = render_model(
        env_model,
        initial_keyframe=robot_config.initial_keyframe,
        camera=large_overview_camera(),
    )
    save_image(image=image, save_path=experiment_dir / "environment_view")

    # Prepare the environment factory
    vision = isinstance(training_config, TrainingWithVisionConfig)
    renderer_maker = (
        functools.partial(
            get_renderer, vision_config=env_config.vision_env_config.vision_config, debug=debug
        )
        if vision
        else None
    )
    env_factory = get_env_factory(
        robot_config=robot_config,
        environment_config=env_config,
        env_model=env_model,
        customize_model=True,
        use_vision=vision,
        renderer_maker=renderer_maker,
    )

    # Load params
    params = load_params(experiment_dir / "trained_policy")

    logging.info("Params loaded. Rendering policy rollout.")
    rollouts, fps = render_policy_rollout(
        env=env_factory(),
        model_config=model_config,
        model_params=params,
        render_config=render_config,
        normalize_observations=training_config.normalize_observations,
        domain_rand_config=terrain_config.randomization_config,
    )
    for rollout_name, frames in rollouts.items():
        for camera_name, camera_frames in frames.items():
            save_video(
                frames=camera_frames,
                fps=fps,
                save_path=experiment_dir / f"{rollout_name}_rollout_{camera_name}_camera.gif"
            )
