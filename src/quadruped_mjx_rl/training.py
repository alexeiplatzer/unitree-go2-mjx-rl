# Supporting
import functools
from datetime import datetime
from pathlib import Path
from typing import Generic, TypeVar

import matplotlib.pyplot as plt
from etils.epath import PathLike, Path
from dataclasses import dataclass, asdict
from collections.abc import Callable

# Math
from flax.training import orbax_utils
from orbax import checkpoint as ocp

# Sim

# Brax
from brax import envs
from brax.envs.base import PipelineEnv
from brax.io import model

# Algorithms
from brax.training.agents.ppo import train as ppo_train

from quadruped_mjx_rl.environments import EnvironmentConfig
from quadruped_mjx_rl.robots import RobotConfig
from quadruped_mjx_rl.models import ModelConfig
from quadruped_mjx_rl.domain_randomization import domain_randomize

name_to_training_config = {
    "simple_ppo": lambda: TrainingConfig(),
}

name_to_training_fn = {
    "simple_ppo": ppo_train,
}


@dataclass
class TrainingConfig:
    name: str = "simple_ppo"
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


EnvType = TypeVar("EnvType", bound=PipelineEnv)


def train(
    environment: type[EnvType],
    env_config: EnvironmentConfig[EnvType],
    robot_config: RobotConfig,
    init_scene_path: PathLike,
    model_config: ModelConfig,
    make_networks_fn: Callable,
    training_config: TrainingConfig,
    train_fn: Callable,
    model_save_path: PathLike,
    checkpoints_save_path: PathLike | None = None,
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

    envs.register_environment(env_config.name, environment)
    env = envs.get_environment(
        env_config.name,
        environment_config=env_config,
        robot_config=robot_config,
        init_scene_path=init_scene_path,
    )

    modules_hidden_layers = {
        f"{module.name}_hidden_layer_sizes": tuple(module.hidden_layers)
        for module in model_config.modules
    }
    make_networks_factory = functools.partial(
        make_networks_fn,
        **modules_hidden_layers,
    )
    train_fn = functools.partial(
        train_fn,
        network_factory=make_networks_factory,
        randomization_fn=domain_randomize,
        policy_params_fn=policy_params_fn,
        seed=0,
        **asdict(training_config),
    )

    x_data = []
    y_data = []
    ydataerr = []
    times = [datetime.now()]
    max_y, min_y = 40, 0

    # Reset environments since internals may be overwritten by tracers from the
    # domain randomization function.
    env = envs.get_environment(
        env_config.name,
        environment_config=env_config,
        robot_config=robot_config,
        init_scene_path=init_scene_path,
    )
    eval_env = envs.get_environment(
        env_config.name,
        environment_config=env_config,
        robot_config=robot_config,
        init_scene_path=init_scene_path,
    )
    make_inference_fn, params, metrics = train_fn(
        environment=env, progress_fn=progress, eval_env=eval_env
    )

    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")

    # Save and reload params.
    model.save_params(model_save_path, params)
    params = model.load_params(model_save_path)
