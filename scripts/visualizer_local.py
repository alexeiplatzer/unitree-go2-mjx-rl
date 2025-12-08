import functools
import logging

import jax
import mediapy as media
import numpy as np
from jax import numpy as jnp
from mujoco import mjx

import paths
from quadruped_mjx_rl.domain_randomization.randomized_tiles import randomize_tiles
from quadruped_mjx_rl.environments import get_env_factory
from quadruped_mjx_rl.environments import QuadrupedVisionBaseEnvConfig
from quadruped_mjx_rl.environments.rendering import (
    large_overview_camera,
    render_model,
    show_image,
)
from quadruped_mjx_rl.environments.wrappers import wrap_for_training
from quadruped_mjx_rl.environments.vision.robotic_vision import get_renderer
from quadruped_mjx_rl.environments.vision.robotic_vision import VisionConfig
from quadruped_mjx_rl.robots import predefined_robot_configs
from quadruped_mjx_rl.terrain_gen.factories import make_plain_tiled_terrain

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, force=True)
    logging.info("Logging switched on.")

    # More legible printing from numpy.
    np.set_printoptions(precision=3, suppress=True, linewidth=100)

    # Prepare paths
    robot_name = "unitree_go2"
    scenes_path = paths.RESOURCES_DIRECTORY / robot_name

    # Prepare robot config
    robot_config = predefined_robot_configs[robot_name]()

    # Prepare env config
    CameraConfig = QuadrupedVisionBaseEnvConfig.ObservationConfig.CameraInputConfig
    env_config = QuadrupedVisionBaseEnvConfig(
        observation_noise=QuadrupedVisionBaseEnvConfig.ObservationConfig(
            camera_inputs=[
                CameraConfig("frontal_ego", True, True, True),
                CameraConfig("terrain", True, True, True),
            ]
        )
    )

    # Prepare the scene file
    init_scene_path = scenes_path / "scene_mjx_empty_arena.xml"

    # Prepare the terrain
    env_model = make_plain_tiled_terrain(init_scene_path)

    # Render the environments from different cameras in mujoco
    render_cam = functools.partial(
        render_model, env_model=env_model, initial_keyframe=robot_config.initial_keyframe
    )
    image_overview = render_cam(camera=large_overview_camera())
    image_tracking = render_cam(camera="track")
    image_terrain = render_cam(camera="privileged")
    image_egocentric = render_cam(camera="ego_frontal")
    show_image(image_overview)
    show_image(image_tracking)
    show_image(image_terrain)
    show_image(image_egocentric)

    # Prepare vision config
    num_envs = 64  # @param {"type":"integer"}
    enabled_cameras = [1, 2]  # @param
    enabled_geom_groups = [0, 1, 2]  # @param
    render_width = 64  # @param {"type":"integer"}
    render_height = 64  # @param {"type":"integer"}
    vision_config = VisionConfig(
        render_batch_size=num_envs,
        enabled_cameras=enabled_cameras,
        enabled_geom_groups=enabled_geom_groups,
        render_width=render_width,
        render_height=render_height,
    )

    # Make the env factory
    renderer_maker = functools.partial(get_renderer, vision_config=vision_config, debug=True)
    env_factory = get_env_factory(
        robot_config=robot_config,
        environment_config=env_config,
        env_model=env_model,
        customize_model=True,
        vision_config=vision_config,
        renderer_maker=renderer_maker,
    )

    # Execute one environment step to initialize mjx
    mjx_model = mjx.put_model(env_model)
    mjx_data = mjx.make_data(mjx_model)
    mjx_data = mjx.forward(mjx_model, mjx_data)

    # Create the environment
    print("Setup finished, initializing the environment...")
    env = env_factory()

    # Wrap the environment
    rng_key = jax.random.PRNGKey(0)
    domain_rand_key, reset_key = jax.random.split(rng_key, 2)
    env_keys = jax.random.split(domain_rand_key, num_envs)
    env = wrap_for_training(
        env=env,
        vision=True,
        num_vision_envs=num_envs,
        randomization_fn=functools.partial(randomize_tiles, env_model=env_model, rng=env_keys),
    )
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    # Execute one step
    state = jit_reset(jax.random.split(reset_key, num_envs))
    state = jit_step(state, jnp.zeros((num_envs, env.action_size)))

    frontal_view = state.obs["pixels/frontal_ego/rgb"]
    print(frontal_view.shape)
    terrain_view = state.obs["pixels/terrain/rgb"]
    print(terrain_view.shape)

    def tile(img, d):
        assert img.shape[0] == d * d
        img = img.reshape((d, d) + img.shape[1:])
        return np.concat(
            np.concat(img, axis=1), axis=1
        )  # replace with 2 for multi-camera tensors!

    # image = tile(rgb_tensor[:16], 4)
    # image.shape
    frontal_view_image = tile(frontal_view[:16], 4)
    print(frontal_view_image.shape)
    terrain_view_image = tile(terrain_view[:16], 4)
    print(terrain_view_image.shape)

    media.show_image(frontal_view_image, width=512)
    media.show_image(terrain_view_image, width=512)
