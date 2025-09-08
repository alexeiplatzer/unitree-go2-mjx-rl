import numpy as np
import paths
from quadruped_mjx_rl.robots import predefined_robot_configs
from quadruped_mjx_rl.environments import JoystickBaseEnvConfig
from quadruped_mjx_rl.models import ActorCriticConfig
from quadruped_mjx_rl.training.configs import TrainingConfig
from quadruped_mjx_rl.environments.physics_pipeline import load_to_spec, spec_to_model
from quadruped_mjx_rl.environments import resolve_env_class, get_env_factory
from quadruped_mjx_rl.training.train_interface import train
from quadruped_mjx_rl.domain_randomization.randomized_physics import domain_randomize
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
        modules=ActorCriticConfig.ModulesConfig(
            policy=[128, 128, 128, 128],
            value=[256, 256, 256, 256],
        ),
    )

    training_config = TrainingConfig(
        num_timesteps=1_000_000, num_envs=16, num_eval_envs=16, batch_size=16
    )

    # Set up the terrain
    init_scene_path = paths.unitree_go2_empty_scene

    from quadruped_mjx_rl.environments.physics_pipeline import load_to_spec, spec_to_model
    from quadruped_mjx_rl.terrain_gen.obstacles import FlatTile, StripesTile
    from quadruped_mjx_rl.terrain_gen.tile import TerrainConfig

    flat_tile = FlatTile()
    stripes_tile = StripesTile()
    terrain = TerrainConfig(tiles=[[flat_tile] * 6])
    print(terrain.tiles)
    env_spec = load_to_spec(init_scene_path)
    terrain.make_arena(env_spec)
    env_model = spec_to_model(env_spec)

    # Prepare initial position
    x, y, z_offset = terrain.get_tile_center_qpos(0, 0)
    init_qpos = env_model.keyframe("home").qpos
    init_qpos[0] = x
    init_qpos[1] = y
    init_qpos[2] += z_offset

    # Render the situation
    import mujoco
    from quadruped_mjx_rl.environments.rendering import show_image

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat = [0, 0, 0]
    camera.distance = 18
    camera.elevation = -30
    mj_data = mujoco.MjData(env_model)
    mj_data.qpos = init_qpos
    with mujoco.Renderer(env_model) as renderer:
        mujoco.mj_forward(env_model, mj_data)
        renderer.update_scene(mj_data, camera=camera)
        image = renderer.render()
    show_image(image)

    env_class = resolve_env_class(env_config)
    env_model = env_class.customize_model(env_model, env_config)
    env_factory = get_env_factory(
        robot_config=robot_config,
        environment_config=env_config,
        env_class=env_class,
        env_model=env_model,
    )

    policy_factories, trained_params, evaluation_metrics = train(
        training_config=training_config,
        model_config=model_config,
        training_env=env_factory(),
        evaluation_env=env_factory(),
        # randomization_fn=domain_randomize,
        run_in_cell=False,
    )
