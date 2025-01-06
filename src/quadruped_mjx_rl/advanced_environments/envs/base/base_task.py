import sys

import gym
import torch

from gym import spaces
import numpy as np

from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple, Union, List

import jax
import jax.numpy as jnp
from functools import partial

from brax.base import System
from brax.envs.base import PipelineEnv, State

# from dial_mpc.config.base_env_config import BaseEnvConfig


# Base class for RL tasks
class BaseEnv(PipelineEnv):

    def __init__(self, config, sim_params, eval_cfg=None):

        self.sim_params = sim_params
        self._config = config
        n_frames = int(config.dt / config.timestep)

        sys = self.make_system(config)

        super().__init__(sys, backend="mjx", n_frames=n_frames)

        self.num_obs = config.env.num_observations
        self.num_privileged_obs = config.env.num_privileged_obs
        self.num_actions = config.env.num_actions

        if eval_cfg is not None:
            self.num_eval_envs = eval_cfg.env.num_envs
        else:
            self.num_eval_envs = 0
        self.num_train_envs = config.env.num_envs
        self.num_envs = self.num_eval_envs + self.num_train_envs

        # allocate buffers
        self.obs_buf = jnp.zeros(shape=(self.num_envs, self.num_obs), dtype=jnp.float32)
        self.rew_buf = jnp.zeros(self.num_envs, dtype=jnp.float32)
        self.reset_buf = jnp.ones(self.num_envs, dtype=jnp.int32)
        self.episode_length_buf = jnp.zeros(self.num_envs, dtype=jnp.int32)
        self.time_out_buf = jnp.zeros(self.num_envs, dtype=jnp.bool_)
        self.privileged_obs_buf = jnp.zeros(
            shape=(self.num_envs, self.num_privileged_obs), dtype=jnp.float32
        )

        self.extras = {}

    def make_system(self, config) -> System:
        """
        Make the system for the environment. Called in BaseEnv.__init__.
        """
        raise NotImplementedError

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self, horizon=0):
        if horizon == 0:
            return self.privileged_obs_buf
        else:
            env_timesteps_remaining_until_rand = int(
                self._config.domain_rand.rand_interval
            ) - self.episode_length_buf % int(self.cfg.domain_rand.rand_interval)
            switched_env_ids = torch.arange(self.num_envs, device=self.device)[
                env_timesteps_remaining_until_rand >= horizon
            ]
            privileged_obs_buf = self.privileged_obs_buf
            privileged_obs_buf[switched_env_ids] = self.next_privileged_obs_buf[
                switched_env_ids
            ]
            return privileged_obs_buf

    def reset_idx(self, env_ids):
        """Reset selected robots"""
        raise NotImplementedError

    def reset(self, rng: jax.Array):
        """Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs))
        obs, privileged_obs, _, _, _ = self.step(
            torch.zeros(self.num_envs, self.num_actions, requires_grad=False)
        )
        return obs, privileged_obs

    def step(self, state: State, actions: jax.Array):
        raise NotImplementedError
