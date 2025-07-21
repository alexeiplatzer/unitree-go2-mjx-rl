"""Main module to execute the training, given all necessary configs"""

# Typing
from collections.abc import Callable, Sequence
from dataclasses import dataclass

# Supporting
import functools
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from etils.epath import Path, PathLike
from flax.training import orbax_utils
from orbax import checkpoint as ocp
from quadruped_mjx_rl.models.io import save_params

# Training
from quadruped_mjx_rl.robots import RobotConfig
from quadruped_mjx_rl.environments import EnvironmentConfig, get_base_model, get_env_factory
from quadruped_mjx_rl.domain_randomization.randomized_physics import domain_randomize
from quadruped_mjx_rl.models import get_networks_factory
from quadruped_mjx_rl.models.agents.ppo.guided_ppo.training import train as guided_ppo_train
from quadruped_mjx_rl.models.agents.ppo.raw_ppo.training import train as raw_ppo_train
from quadruped_mjx_rl.policy_rendering import (
    render_rollout, RenderConfig, RolloutRenderer, PolicyRenderingFn, render_policy_rollout
)


# Configurations
from quadruped_mjx_rl.config_utils import Configuration, register_config_base_class
from quadruped_mjx_rl.models.configs import ActorCriticConfig, ModelConfig, TeacherStudentConfig
from quadruped_mjx_rl.robotic_vision import VisionConfig, get_renderer


@dataclass
class TrainingConfig(Configuration):
    num_timesteps: int = 100_000_000
    num_evals: int = 10
    reward_scaling: int = 1
    episode_length: int = 1000
    normalize_observations: bool = True
    action_repeat: int = 1
    unroll_length: int = 20
    num_minibatches: int = 32
    num_updates_per_batch: int = 4
    discounting: float = 0.97
    learning_rate: float = 0.0004
    entropy_cost: float = 0.01
    num_envs: int = 8192
    batch_size: int = 256

    @classmethod
    def config_base_class_key(cls) -> str:
        return "training"

    @classmethod
    def training_class_key(cls) -> str:
        return "PPO"

    @classmethod
    def from_dict(cls, config_dict: dict) -> Configuration:
        training_class_key = config_dict.pop("training_class")
        training_config_class = _training_config_classes[training_class_key]
        return super(TrainingConfig, training_config_class).from_dict(config_dict)

    def to_dict(self) -> dict:
        config_dict = super().to_dict()
        config_dict["training_class"] = type(self).training_class_key()
        return config_dict


register_config_base_class(TrainingConfig)


@dataclass
class TrainingWithVisionConfig(TrainingConfig):
    num_timesteps: int = 1_000_000
    num_evals: int = 5
    action_repeat: int = 1
    unroll_length: int = 20
    num_minibatches: int = 8
    num_updates_per_batch: int = 8
    num_envs: int = 256
    batch_size: int = 256

    madrona_backend: bool = True
    # wrap_env: bool = False
    num_eval_envs: int = 256
    max_grad_norm: float = 1.0
    # num_resets_per_eval: int = 1

    @classmethod
    def training_class_key(cls) -> str:
        return "PPO_Vision"


_training_config_classes = {
    "default": TrainingConfig,
    "PPO": TrainingConfig,
    "PPO_Vision": TrainingWithVisionConfig,
}


def make_progress_fn(num_timesteps, reward_max=40):
    x_data = []
    y_data = []
    ydataerr = []
    times = [datetime.now()]
    max_y, min_y = reward_max, 0

    def progress(num_steps, metrics):
        times.append(datetime.now())
        x_data.append(num_steps)
        y_data.append(metrics["eval/episode_reward"])
        ydataerr.append(metrics["eval/episode_reward_std"])

        plt.xlim([0, num_timesteps * 1.25])
        plt.ylim([min_y, max_y])

        plt.xlabel("# environment steps")
        plt.ylabel("reward per episode")
        plt.title(f"y={y_data[-1]:.3f}")

        plt.errorbar(x_data, y_data, yerr=ydataerr)
        plt.show()

    return progress, times


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
        render_policy_rollout, render_config=RenderConfig(),
    ),
):
    env_model = get_base_model(init_scene_path, env_config)

    env_factory = get_env_factory(
        robot_config, env_config, env_model, vision_config=vision_config #, renderer=renderer
    )
    env = env_factory()
    renderer = get_renderer(env.pipeline_model, vision_config)
    env.renderer = renderer
    train_fn = get_training_fn(
        model_config=model_config, training_config=training_config, vision=True
    )
    progress_fn, eval_times = make_progress_fn(num_timesteps=training_config.num_timesteps)
    make_inference_fn, params, metrics = train_fn(
        environment=env,
        #eval_env=env,
        seed=0,
        # randomization_fn=domain_randomize,
        progress_fn=progress_fn,
    )
    print(f"time to jit: {eval_times[1] - eval_times[0]}")
    print(f"time to train: {eval_times[-1] - eval_times[1]}")

    # Save params
    save_params(params_save_path, params)

    if policy_rendering_fn is not None:
        rendering_env = env_factory()
        rendering_env.renderer = renderer
        policy_rendering_fn(rendering_env, make_inference_fn(params))


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
        render_policy_rollout, render_config=RenderConfig(),
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


def train(
    *,
    robot_config: RobotConfig,
    env_config: EnvironmentConfig,
    init_scene_path: PathLike,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    model_save_path: PathLike,
    checkpoints_save_path: PathLike | None = None,
    policy_rendering_fn=functools.partial(
        render_policy_rollout, render_config=RenderConfig(),
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
    training_params = training_config.to_dict()
    training_params.pop("training_class")
    learning_rate = training_params.pop("learning_rate")
    if vision and training_params.get("num_eval_envs", 0) != training_params["num_envs"]:
        # batch sizes should coincide when using vision
        logging.warning(
            "All batch sizes must coincide when using vision. "
            "Number of eval envs must be equal to num_envs. "
            "Setting num_eval_envs to num_envs."
        )
        training_params["num_eval_envs"] = training_params["num_envs"]
    if isinstance(model_config, TeacherStudentConfig):
        return functools.partial(
            guided_ppo_train,
            teacher_network_factory=networks_factory["teacher"],
            student_network_factory=networks_factory["student"],
            teacher_learning_rate=learning_rate,
            student_learning_rate=learning_rate,
            **training_params,
        )
    elif isinstance(model_config, ActorCriticConfig):
        return functools.partial(
            raw_ppo_train,
            network_factory=networks_factory,
            learning_rate=learning_rate,
            **training_params,
        )
    else:
        raise NotImplementedError
