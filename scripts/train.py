# Supporting
import os
import time
import itertools
import functools
from datetime import datetime
from etils import epath
from typing import Any, Dict, Sequence, Tuple, Callable, NamedTuple, Optional, Union, List
from ml_collections import config_dict
import matplotlib.pyplot as plt

# Math
import jax
import jax.numpy as jp
import numpy as np
from jax import config  # Analytical gradients work much better with double precision.
from flax.training import orbax_utils
from flax import struct
from orbax import checkpoint as ocp

# Sim
import mujoco
import mujoco.mjx as mjx

# Brax
from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
from brax.mjx.pipeline import _reformat_contact
from brax.training.acme import running_statistics
from brax.io import html, mjcf, model

# Algorithm
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks

from scripts import ckpt_path, model_path
from domain_randomization import domain_randomize
import training_environments.go2_ppo


def progress(num_steps, metrics):
    times.append(datetime.now())
    x_data.append(num_steps)
    y_data.append(metrics['eval/episode_reward'])
    ydataerr.append(metrics['eval/episode_reward_std'])

    plt.xlim([0, train_fn.keywords['num_timesteps'] * 1.25])
    plt.ylim([min_y, max_y])

    plt.xlabel('# environment steps')
    plt.ylabel('reward per episode')
    plt.title(f'y={y_data[-1]:.3f}')

    plt.errorbar(
        x_data, y_data, yerr=ydataerr)
    plt.show()


def policy_params_fn(current_step, make_policy, params):
    # save checkpoints
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(params)
    path = ckpt_path / f'{current_step}'
    orbax_checkpointer.save(path, params, force=True, save_args=save_args)


if __name__ == '__main__':
    env_name = 'joystick_go2'
    env = envs.get_environment(env_name)

    make_networks_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(128, 128, 128, 128))
    train_fn = functools.partial(
        ppo.train, num_timesteps=100_000_000, num_evals=10,
        reward_scaling=1, episode_length=1000, normalize_observations=True,
        action_repeat=1, unroll_length=20, num_minibatches=32,
        num_updates_per_batch=4, discounting=0.97, learning_rate=3.0e-4,
        entropy_cost=1e-2, num_envs=8192, batch_size=256,
        network_factory=make_networks_factory,
        randomization_fn=domain_randomize,
        policy_params_fn=policy_params_fn,
        seed=0)

    x_data = []
    y_data = []
    ydataerr = []
    times = [datetime.now()]
    max_y, min_y = 40, 0

    # Reset environments since internals may be overwritten by tracers from the
    # domain randomization function.
    env = envs.get_environment(env_name)
    eval_env = envs.get_environment(env_name)
    make_inference_fn, params, _ = train_fn(environment=env,
                                            progress_fn=progress,
                                            eval_env=eval_env)

    print(f'time to jit: {times[1] - times[0]}')
    print(f'time to train: {times[-1] - times[1]}')

    # Save and reload params.
    model.save_params(model_path, params)
    params = model.load_params(model_path)