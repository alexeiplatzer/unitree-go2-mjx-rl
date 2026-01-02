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
)
from quadruped_mjx_rl.robotic_vision import get_renderer
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
    experiment_name = "randomized_tiles_blind_2"
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
        paths.MODEL_CONFIGS_DIRECTORY / f"basic_lighter.yaml",
        paths.ENVIRONMENT_CONFIGS_DIRECTORY / f"color_guided_joystick.yaml",
    )
    render_config = RenderConfig(
        episode_length=2000, n_steps=1000, cameras=["track", "ego_frontal"]
    )

    # Prepare environment model
    env_model = make_terrain(
        resources_directory=paths.RESOURCES_DIRECTORY,
        terrain_config=terrain_config,
        robot_config=robot_config,
    )

    # Render the environment model
    camera_list = [
        mujoco.mj_id2name(env_model, mujoco.mjtObj.mjOBJ_CAMERA, idx)
        for idx in range(env_model.ncam)
    ]
    for camera_name in camera_list:
        image = render_model(
            env_model,
            initial_keyframe=robot_config.initial_keyframe,
            camera=camera_name,
        )
        save_image(image=image, save_path=experiment_dir / f"environment_view_{camera_name}")

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
                save_path=experiment_dir / f"{rollout_name}_rollout_{camera_name}_camera.gif",
            )
