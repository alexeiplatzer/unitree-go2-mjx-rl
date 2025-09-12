import functools
import logging

import numpy as np

import paths
from quadruped_mjx_rl.config_utils import prepare_configs
from quadruped_mjx_rl.domain_randomization.randomized_obstacles import terrain_randomize
from quadruped_mjx_rl.environments import (
    get_env_factory,
)
from quadruped_mjx_rl.environments.rendering import (
    large_overview_camera, render_model, show_image,
)
from quadruped_mjx_rl.robotic_vision import get_renderer
from quadruped_mjx_rl.terrain_gen import make_simple_obstacle_terrain
from quadruped_mjx_rl.training.train_interface import train

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, force=True)
    logging.info("Logging configured.")

    # More legible printing from numpy.
    np.set_printoptions(precision=3, suppress=True, linewidth=100)

    # Prepare configs
    configs_dict = prepare_configs(
        paths.CONFIGS_DIRECTORY / "vision_ppo_dirty_experiment.yaml",
        paths.CONFIGS_DIRECTORY / "unitree_go2.yaml",
    )
    robot_config = configs_dict["robot"]
    env_config = configs_dict["environment"]
    model_config = configs_dict["model"]
    training_config = configs_dict["training"]
    vision_config = configs_dict["vision"]

    # Prepare the terrain
    init_scene_path = paths.unitree_go2_empty_scene

    env_model = make_simple_obstacle_terrain(init_scene_path)

    # Render the situation
    image = render_model(
        env_model, initial_keyframe=robot_config.initial_keyframe,
        camera=large_overview_camera()
    )
    show_image(image)

    # Make env factory
    renderer_maker = functools.partial(get_renderer, vision_config=vision_config, debug=True)
    env_factory = get_env_factory(
        robot_config=robot_config,
        environment_config=env_config,
        env_model=env_model,
        customize_model=True,
        vision_config=vision_config,
        renderer_maker=renderer_maker,
    )

    env = env_factory()

    trained_params = train(
        training_config=training_config,
        model_config=model_config,
        training_env=env,
        evaluation_env=None,
        randomization_fn=functools.partial(terrain_randomize, mj_model=env_model),
        run_in_cell=False,
    )
