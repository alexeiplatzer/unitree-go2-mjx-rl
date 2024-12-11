from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from brax import envs
from brax import math
from brax.base import System
from brax.base import State as PipelineState
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf

from .paths import SCENE_PATH


@dataclass
class Go2TeacherEnvConfig:
    model_path: str = SCENE_PATH.as_posix()
    dt: float = 0.02
    timestep: float = 0.002


class Go2TeacherEnv(PipelineEnv):

    def __init__(self, config: Go2TeacherEnvConfig = Go2TeacherEnvConfig()):

        self._config = config
        n_frames = int(config.dt / config.timestep)
        sys = self.make_system(config)
        super().__init__(sys, backend="mjx", n_frames=n_frames)

        self._init_q = jnp.array(sys.mj_model.keyframe("home").qpos)

        self._nv = sys.nv

    def make_system(self, config: Go2TeacherEnvConfig) -> System:
        model_path = config.model_path
        sys = mjcf.load(model_path)
        sys = sys.tree_replace({"opt.timestep": config.timestep})
        return sys

    def reset(self, rng: jax.Array) -> State:
        rng, key = jax.random.split(rng)

        pipeline_state = self.pipeline_init(self._init_q, jnp.zeros(self._nv))

        state_info = {
            "rng": rng,
            "last_contact": jnp.zeros(4, dtype=jnp.bool),
            "step": 0
        }

        obs = self._get_obs(pipeline_state, state_info)
        reward, done = jnp.zeros(2)
        metrics = {}
        state = State(pipeline_state, obs, reward, done, metrics, state_info)
        return state

    def step(self, state: State, action: jax.Array) -> State:

        # physics step
        motor_targets = self._default_pose + action * self._action_scale
        motor_targets = jnp.clip(motor_targets, self.lowers, self.uppers)
        pipeline_state = self.pipeline_step(state, motor_targets)
        x, xd = pipeline_state.x, pipeline_state.xd

        # observation data
        obs = self._get_obs(pipeline_state, state.info, state.obs)

        # done if joint limits are reached or robot is falling
        up = jnp.array([0.0, 0.0, 1.0])
        done = jnp.dot(math.rotate(up, x.rot[self._torso_idx - 1]), up) < 0

        # reward
        rewards = {}
        reward = jnp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

        state = state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )
        return state

    def _get_obs(self, pipeline_state: PipelineState, state_info: dict[str, Any]) -> jax.Array:

        obs = jnp.concatenate([
            state_info["last_act"]
        ])

        return obs


envs.register_environment(env_name='go2_teacher', env_class=Go2TeacherEnv)
