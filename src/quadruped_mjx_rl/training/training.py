"""Main module to execute the training, given all necessary configs"""

from dataclasses import dataclass, field

import functools
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from etils.epath import Path, PathLike
from flax.training import orbax_utils
from orbax import checkpoint as ocp
from quadruped_mjx_rl.models.io import save_params
from IPython.display import display

from quadruped_mjx_rl.robots import RobotConfig
from quadruped_mjx_rl.environments import (
    EnvironmentConfig,
    get_base_model,
    get_env_factory,
)
from quadruped_mjx_rl.domain_randomization.randomized_physics import domain_randomize
from quadruped_mjx_rl.models import get_networks_factory
from quadruped_mjx_rl.policy_rendering import (
    RenderConfig,
    render_policy_rollout,
)

from quadruped_mjx_rl.models.configs import ActorCriticConfig, ModelConfig, TeacherStudentConfig
from quadruped_mjx_rl.robotic_vision import VisionConfig, get_renderer
from quadruped_mjx_rl.training.configs import TrainingConfig, TrainingWithVisionConfig
from quadruped_mjx_rl.training.train_main import train
from quadruped_mjx_rl.environments.physics_pipeline import Env
from quadruped_mjx_rl.training.evaluation import make_progress_fn


def train_with_vision(
    *,
    robot_config: RobotConfig,
    env_config: EnvironmentConfig,
    init_scene_path: PathLike,
    model_config: ModelConfig,
    training_config: TrainingWithVisionConfig,
    vision_config: VisionConfig,
    params_save_path: PathLike,
    policy_rendering_fn=functools.partial(
        render_policy_rollout,
        render_config=RenderConfig(),
    ),
):
    logging.info("Getting the base model...")
    env_model = get_base_model(init_scene_path, env_config)

    env_factory = get_env_factory(
        robot_config, env_config, env_model, vision_config=vision_config  # , renderer=renderer
    )
    logging.info("Creating the environment...")
    env = env_factory()
    logging.info("Instantiating the rendering backend...")
    renderer = get_renderer(env.pipeline_model, vision_config)
    env.renderer = renderer
    train_fn = get_training_fn(
        model_config=model_config, training_config=training_config, vision=True
    )
    progress_fn, eval_times = make_progress_fn(num_timesteps=training_config.num_timesteps)
    # env = maybe_wrap_env(
    #     env=env,
    #     wrap_env=True,
    #     num_envs=training_config.num_envs,
    #     episode_length=training_config.episode_length,
    #     action_repeat=training_config.action_repeat,
    #     device_count=1,
    #
    # )
    logging.info("Starting training with vision...")
    make_inference_fn, params, metrics = train_fn(
        environment=env,
        # eval_env=env,
        seed=0,
        # randomization_fn=domain_randomize,
        progress_fn=progress_fn,
        # wrap_env=False,
    )
    logging.info(f"time to jit: {eval_times[1] - eval_times[0]}")
    logging.info(f"time to train: {eval_times[-1] - eval_times[1]}")

    # Save params
    save_params(params_save_path, params)

    if policy_rendering_fn is not None:
        student_params = params[1]
        rendering_env = env_factory()
        rendering_env.renderer = renderer
        policy_rendering_fn(rendering_env, make_inference_fn(student_params))


def train_wrong(
    *,
    robot_config: RobotConfig,
    env_config: EnvironmentConfig,
    init_scene_path: PathLike,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    model_save_path: PathLike,
    checkpoints_save_path: PathLike | None = None,
    policy_rendering_fn=functools.partial(
        render_policy_rollout,
        render_config=RenderConfig(),
    ),
):
    if checkpoints_save_path is not None:
        checkpoints_save_path = Path(checkpoints_save_path)

        def policy_params_fn(current_step, make_policy, parameters):
            # save checkpoints
            orbax_checkpointer = ocp.PyTreeCheckpointer()
            save_args = orbax_utils.save_args_from_target(parameters)
            path = checkpoints_save_path / f"{current_step}"
            orbax_checkpointer.save(path, parameters, force=True, save_args=save_args)

    else:
        policy_params_fn = lambda *args: None

    env_model = get_base_model(init_scene_path, env_config)
    env_factory = get_env_factory(robot_config, env_config, env_model)
    env = env_factory()
    eval_env = env_factory()
    progress_fn, eval_times = make_progress_fn(num_timesteps=training_config.num_timesteps)
    train_fn = get_training_fn(model_config, training_config, vision=False)
    make_inference_fn, params, metrics = train_fn(
        environment=env,
        progress_fn=progress_fn,
        eval_env=eval_env,
        randomization_fn=domain_randomize,
        policy_params_fn=policy_params_fn,
        seed=0,
    )
    print(f"time to jit: {eval_times[1] - eval_times[0]}")
    print(f"time to train: {eval_times[-1] - eval_times[1]}")

    # Save params
    save_params(model_save_path, params)
    # params = model.load_params(model_save_path)

    if policy_rendering_fn is not None:
        rendering_env = env_factory()
        policy_rendering_fn(rendering_env, make_inference_fn(params))


def train_from_configs(
    *,
    robot_config: RobotConfig,
    env_config: EnvironmentConfig,
    init_scene_path: PathLike,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    model_save_path: PathLike,
    checkpoints_save_path: PathLike | None = None,
    policy_rendering_fn=functools.partial(
        render_policy_rollout,
        render_config=RenderConfig(),
    ),
):
    if checkpoints_save_path is not None:
        checkpoints_save_path = Path(checkpoints_save_path)

        def policy_params_fn(current_step, make_policy, parameters):
            # save checkpoints
            orbax_checkpointer = ocp.PyTreeCheckpointer()
            save_args = orbax_utils.save_args_from_target(parameters)
            path = checkpoints_save_path / f"{current_step}"
            orbax_checkpointer.save(path, parameters, force=True, save_args=save_args)

    else:
        policy_params_fn = lambda *args: None

    train_fn = get_training_fn(model_config, training_config, vision=False)
    train_fn = functools.partial(
        train_fn,
        randomization_fn=domain_randomize,
        policy_params_fn=policy_params_fn,
        seed=0,
    )
    env_model = get_base_model(init_scene_path, env_config)
    env_factory = get_env_factory(robot_config, env_config, env_model)
    env = env_factory()
    eval_env = env_factory()
    progress_fn, eval_times = make_progress_fn(num_timesteps=training_config.num_timesteps)
    make_inference_fn, params, metrics = train_fn(
        environment=env,
        progress_fn=progress_fn,
        eval_env=eval_env,
    )
    print(f"time to jit: {eval_times[1] - eval_times[0]}")
    print(f"time to train: {eval_times[-1] - eval_times[1]}")

    # Save params
    save_params(model_save_path, params)
    # params = model.load_params(model_save_path)

    if policy_rendering_fn is not None:
        rendering_env = env_factory()
        policy_rendering_fn(rendering_env, make_inference_fn(params))


def get_training_fn(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    vision: bool = False,
):
    networks_factory = get_networks_factory(model_config)
    if vision and training_config.num_eval_envs != training_config.num_envs:
        # batch sizes should coincide when using vision
        logging.warning(
            "All batch sizes must coincide when using vision. "
            "Number of eval envs must be equal to num_envs. "
            "Setting num_eval_envs to num_envs."
        )
        training_config.num_eval_envs = training_config.num_envs
    return functools.partial(
        train,
        training_config=training_config,
        network_factory=networks_factory,
    )


def interface_training_fn(
    *,
    training_env: Env,
    evaluation_env: Env | None,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    params_save_path: PathLike,
    vision: bool = False,
    params_restore_path: PathLike | None = None,
    randomization_fn=None,
):
    if vision and evaluation_env is not None:
        logging.warning("Vision training should reuse the training env for evaluation.")
        evaluation_env = None


    policy_factories, agent_params, metrics = train(
        training_config=training_config,
        environment=training_env,
        eval_env=evaluation_env,
        max_devices_per_host=None,
        wrap_env=True,
        wrap_env_fn=None,

    )

