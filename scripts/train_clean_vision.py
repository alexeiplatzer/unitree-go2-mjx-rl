import numpy as np
import functools
import paths
from quadruped_mjx_rl.robots import predefined_robot_configs
from quadruped_mjx_rl.environments import (
    JoystickBaseEnvConfig,
    QuadrupedJoystickVisionEnvConfig,
)
from quadruped_mjx_rl.models import TeacherStudentVisionConfig
from quadruped_mjx_rl.training.configs import TrainingWithVisionConfig
from quadruped_mjx_rl.environments.physics_pipeline import load_to_spec, spec_to_model
from quadruped_mjx_rl.terrain_gen.obstacles import FlatTile, StripesTile
from quadruped_mjx_rl.terrain_gen.tile import TerrainConfig
from quadruped_mjx_rl.environments import resolve_env_class, get_env_factory
from quadruped_mjx_rl.training.train_interface import train
from quadruped_mjx_rl.robotic_vision import get_renderer, VisionConfig
from jax import numpy as jnp
import logging

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, force=True)
    logging.info("Logging configured.")

    # More legible printing from numpy.
    np.set_printoptions(precision=3, suppress=True, linewidth=100)

    # Prepare configs
    robot_config = predefined_robot_configs["unitree_go2"]()

    env_config = QuadrupedJoystickVisionEnvConfig()

    model_config = TeacherStudentVisionConfig(
        modules=TeacherStudentVisionConfig.ModulesConfig(
            policy=[128, 128, 128, 128],
            value=[256, 256, 256, 256],
            encoder_convolutional=[4, 8, 32],
            encoder_dense=[256, 256],
            adapter_convolutional=[4, 8, 32],
            adapter_dense=[256, 256],
        ),
        latent_size=256,
    )

    training_config = TrainingWithVisionConfig(
        num_timesteps=1_000_000, batch_size=4, num_envs=8, num_eval_envs=8
    )

    init_scene_path = paths.unitree_go2_empty_scene

    flat_tile = FlatTile()
    stripes_tile = StripesTile()
    terrain = TerrainConfig(tiles=[[flat_tile] * 4 + [stripes_tile] * 4])
    env_spec = load_to_spec(init_scene_path)
    terrain.make_arena(env_spec)
    env_model = spec_to_model(env_spec)

    # Prepare initial position
    x, y, z_offset = terrain.get_tile_center_qpos(0, 0)
    init_qpos = env_model.keyframe("home").qpos
    init_qpos[0] = x
    init_qpos[1] = y
    init_qpos[2] += z_offset

    vision_config = VisionConfig(
        render_batch_size=8, render_width=32, render_height=32, enabled_geom_groups=[0, 1, 2]
    )

    modifier_env_class = resolve_env_class(JoystickBaseEnvConfig())
    env_model = modifier_env_class.customize_model(env_model, env_config)
    renderer_maker = functools.partial(get_renderer, vision_config=vision_config, debug=True)
    env_factory = get_env_factory(
        robot_config=robot_config,
        environment_config=env_config,
        env_model=env_model,
        vision_config=vision_config,
        init_qpos=jnp.array(init_qpos),
        renderer_maker=renderer_maker,
    )

    env = env_factory()

    policy_factories, trained_params, evaluation_metrics = train(
        training_config=training_config,
        model_config=model_config,
        training_env=env,
        evaluation_env=None,
        randomization_fn=None,
        run_in_cell=False,
    )
