# Typing
from dataclasses import dataclass, asdict
from collections.abc import Callable

# Supporting
import functools
from datetime import datetime
import matplotlib.pyplot as plt
from etils.epath import PathLike, Path


# Math
from flax.training import orbax_utils
from orbax import checkpoint as ocp

# Brax
from brax import envs
from brax.envs.base import PipelineEnv
from brax.io import model

# Algorithms
from brax.training.agents.ppo import train as ppo_train

from quadruped_mjx_rl.robots import RobotConfig
from quadruped_mjx_rl.robotic_vision import VisionConfig
from quadruped_mjx_rl.environments import EnvironmentConfig, configs_to_env_classes
from quadruped_mjx_rl.models import ModelConfig, get_networks_factory
from quadruped_mjx_rl.domain_randomization import domain_randomize

from quadruped_mjx_rl.models.configs import ModelConfig, ActorCriticConfig, TeacherStudentConfig
from quadruped_mjx_rl.models.architectures import raw_actor_critic as raw_networks
from quadruped_mjx_rl.models.architectures import guided_actor_critic as guided_networks
from quadruped_mjx_rl.models.agents.ppo.raw_ppo.training import train as raw_ppo_train
from quadruped_mjx_rl.models.agents.ppo.guided_ppo.training import train as guided_ppo_train


@dataclass
class TrainingConfig:
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
    training_class: str = "PPO"


@dataclass
class TrainingWithVisionConfig(TrainingConfig):
    num_timesteps: int = 1_000_000
    num_evals: int = 5
    reward_scaling: int = 1
    episode_length: int = 1000
    normalize_observations: bool = True
    action_repeat: int = 1
    unroll_length: int = 10
    num_minibatches: int = 8
    num_updates_per_batch: int = 8
    discounting: float = 0.97
    learning_rate: float = 0.0005
    entropy_cost: float = 0.005
    num_envs: int = 512
    batch_size: int = 256
    training_class: str = "PPO_Vision"

    madrona_backend: bool = True
    # wrap_env: bool = False
    num_eval_envs: int = 512
    max_grad_norm: float = 1.0
    # num_resets_per_eval: int = 1


training_config_classes = {
    "default": TrainingConfig,
    "PPO": TrainingConfig,
    "PPO_Vision": TrainingWithVisionConfig,
}


def train(
    env_factory: Callable[[], PipelineEnv],
    model_config: ModelConfig,
    training_config: TrainingConfig,
    model_save_path: PathLike,
    checkpoints_save_path: PathLike | None = None,
    vision: bool = False,
    vision_config: VisionConfig | None = None,
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

    def progress(num_steps, metrics):
        times.append(datetime.now())
        x_data.append(num_steps)
        y_data.append(metrics["eval/episode_reward"])
        ydataerr.append(metrics["eval/episode_reward_std"])

        plt.xlim([0, train_fn.keywords["num_timesteps"] * 1.25])
        plt.ylim([min_y, max_y])

        plt.xlabel("# environment steps")
        plt.ylabel("reward per episode")
        plt.title(f"y={y_data[-1]:.3f}")

        plt.errorbar(x_data, y_data, yerr=ydataerr)
        plt.show()

    train_fn_proto = get_training_fn(model_config, training_config, vision=vision)
    train_fn = functools.partial(
        train_fn_proto,
        randomization_fn=domain_randomize,
        policy_params_fn=policy_params_fn,
        seed=0,
    )

    x_data = []
    y_data = []
    ydataerr = []
    times = [datetime.now()]
    max_y, min_y = 40, 0

    # Reset environments since internals may be overwritten by tracers from the
    # domain randomization function.
    env = env_factory()
    eval_env = env_factory()
    make_inference_fn, params, metrics = train_fn(
        environment=env,
        progress_fn=progress,
        eval_env=eval_env,
    )

    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")

    # Save params
    model.save_params(model_save_path, params)
    # params = model.load_params(model_save_path)


def get_training_fn(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    vision: bool = False,
):
    networks_factory = get_networks_factory(model_config, vision=vision)
    training_params = asdict(training_config)
    training_params.pop("training_class")
    learning_rate = training_params.pop("learning_rate")
    if isinstance(model_config, TeacherStudentConfig):
        return functools.partial(
            guided_ppo_train,
            teacher_networks_factory=networks_factory["teacher"],
            student_networks_factory=networks_factory["student"],
            teacher_learning_rate=learning_rate,
            student_learning_rate=learning_rate,
            **training_params,
        )
    elif isinstance(model_config, ActorCriticConfig):
        return functools.partial(
            raw_ppo_train,
            networks_factory=networks_factory,
            learning_rate=learning_rate,
            **training_params,
        )
    else:
        raise NotImplementedError
