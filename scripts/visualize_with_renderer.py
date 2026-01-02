import functools
import logging

import numpy as np
import jax

import paths
from quadruped_mjx_rl import running_statistics
from quadruped_mjx_rl.models import get_networks_factory
from quadruped_mjx_rl.models.architectures import (
    TeacherStudentAgentParams,
    TeacherStudentNetworks, TeacherStudentRecurrentNetworks,
)
from quadruped_mjx_rl.policy_rendering import render_rollout, save_video
from quadruped_mjx_rl.configs import prepare_all_configs
from quadruped_mjx_rl.environments import get_env_factory
from quadruped_mjx_rl.environments.wrappers import _vmap_wrap_with_randomization, EpisodeWrapper
from quadruped_mjx_rl.models.io import load_params
from quadruped_mjx_rl.robotic_vision import get_renderer
from quadruped_mjx_rl.policy_rendering import RenderConfig
from quadruped_mjx_rl.terrain_gen import make_terrain
from quadruped_mjx_rl.training import TrainingConfig, TrainingWithVisionConfig

if __name__ == "__main__":
    debug = False

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
    assert isinstance(training_config, TrainingWithVisionConfig)
    training_config.check_validity()
    render_config = RenderConfig(
        episode_length=2000, n_steps=1000, cameras=["track", "ego_frontal"]
    )

    # Prepare environment model
    env_model = make_terrain(
        resources_directory=paths.RESOURCES_DIRECTORY,
        terrain_config=terrain_config,
        robot_config=robot_config,
    )

    # Prepare the environment factory
    renderer_maker = (training_config.get_renderer_factory(gpu_id=0, debug=debug))
    env_factory = get_env_factory(
        robot_config=robot_config,
        environment_config=env_config,
        env_model=env_model,
        customize_model=True,
        vision_wrapper_config=vision_wrapper_config,
        renderer_maker=renderer_maker,
    )

    # Create the environment
    env = env_factory()

    # Prepare the random keys
    key = jax.random.PRNGKey(render_config.seed)
    wrapping_key, _ = jax.random.split(key, 2)

    env = _vmap_wrap_with_randomization(
        env,
        vision=True,
        worlds_random_key=wrapping_key,
        randomization_config=terrain_config.randomization_config,
    )
    demo_env = EpisodeWrapper(
        env,
        episode_length=render_config.episode_length,
        action_repeat=1,
    )

    # Prepare env
    env = env_factory()

    # Load params
    params = load_params(experiment_dir / "trained_policy")
    network_factory = get_networks_factory(model_config)
    preprocess_fn = (
        running_statistics.normalize
        if training_config.normalize_observations else lambda x, y: x
    )
    network = network_factory(
        observation_size={
            "proprioceptive": 1,
            "proprioceptive_history": 1,
            "environment_privileged": 1,
            "pixels/terrain/depth": 1,
            "pixels/frontal_ego/rgb": 1,
            "pixels/frontal_ego/rgb_adjusted": 1,
            "privileged_terrain_map": 1,
        },
        action_size=env.action_size,
        preprocess_observations_fn=preprocess_fn,
    )
    unroll_factories = {
        "training_acting_policy": network.make_unroll_fn(
            params, deterministic=True, accumulate_pipeline_states=True
        )
    }
    if isinstance(network, TeacherStudentRecurrentNetworks):
        assert isinstance(params, TeacherStudentAgentParams)
        unroll_factories["student_policy"] = network.make_student_unroll_fn(
            params, deterministic=True, accumulate_pipeline_states=True
        )
    elif isinstance(network, TeacherStudentNetworks):
        unroll_factories["student_policy"] = network.make_unroll_fn(
            params,
            policy_factory=network.get_student_policy_factory() if not network.vision else None,
            apply_encoder_fn=network.apply_student_encoder,
            deterministic=True,
            accumulate_pipeline_states=True,
        )

    logging.info("Params loaded. Rendering policy rollout.")
    render_fn = functools.partial(
        render_rollout,
        env=demo_env,
        render_config=render_config,
    )
    fps = 1.0 / (env.dt.item() * render_config.render_every)
    for unroll_name, unroll_factory in unroll_factories.items():
        frames = render_rollout(
            env=demo_env,
            render_config=render_config,
            unroll_factory=unroll_factory,
        )
        for camera_name, camera_frames in frames.items():
            save_video(
                frames=camera_frames,
                fps=fps,
                save_path=experiment_dir / f"{unroll_name}_rollout_{camera_name}_camera.gif",
            )
