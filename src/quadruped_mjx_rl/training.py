import functools
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime

import matplotlib.pyplot as plt
from brax.envs.base import PipelineEnv
from brax.io import model
from etils.epath import Path, PathLike
from flax.training import orbax_utils
from orbax import checkpoint as ocp

from quadruped_mjx_rl.domain_randomization import domain_randomize
from quadruped_mjx_rl.models import get_networks_factory
from quadruped_mjx_rl.models.agents.ppo.guided_ppo.training import train as guided_ppo_train
from quadruped_mjx_rl.models.agents.ppo.raw_ppo.training import train as raw_ppo_train
from quadruped_mjx_rl.models.configs import ActorCriticConfig, ModelConfig, TeacherStudentConfig
from quadruped_mjx_rl.robotic_vision import VisionConfig


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
    action_repeat: int = 1
    unroll_length: int = 20
    num_minibatches: int = 8
    num_updates_per_batch: int = 8
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
    env_factory: Callable[..., PipelineEnv],
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

    train_fn = get_training_fn(model_config, training_config, vision=vision)
    train_fn = functools.partial(
        train_fn,
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
    if vision:
        if vision_config is None:
            raise ValueError("vision_config must be provided when vision is True")
        if vision_config.render_batch_size != training_config.num_envs:
            logging.warning(
                "All batch sizes must coincide when using vision. "
                "Render batch size must be equal to num_envs. "
                "Setting render_batch_size to num_envs."
            )
            vision_config.render_batch_size = training_config.num_envs
        env = env_factory(vision_config=vision_config)
        eval_env = None
    else:
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
    networks_factory = get_networks_factory(model_config)
    training_params = asdict(training_config)
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
