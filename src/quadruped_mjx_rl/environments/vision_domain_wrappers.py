import functools
from typing import Any

import jax
from flax import struct as flax_struct
from jax import numpy as jnp

from quadruped_mjx_rl.domain_randomization import (
    DomainRandomizationConfig,
    TerrainMapRandomizationConfig,
)
from quadruped_mjx_rl.physics_pipeline import PipelineModel, EnvModel, PipelineState
from quadruped_mjx_rl.physics_pipeline import Env, State, Wrapper
from quadruped_mjx_rl.types import PRNGKey, Observation
from quadruped_mjx_rl.environments.vision import VisionWrapper, ColorGuidedVisionWrapper


class TerrainMapWrapper(Wrapper):
    """Wrapper for domain randomization with colored terrain, preserving the terrain info."""

    def __init__(
        self,
        env: ColorGuidedVisionWrapper,
        randomization_fn,
        key_envs: jax.Array,
        num_envs: int,
    ):
        super().__init__(env)
        (
            self._sys_v,
            self._in_axes,
            (self._rgba_table_v, self._friction_table_v, self._stiffness_table_v),
        ) = randomization_fn(
            pipeline_model=self.pipeline_model,
            env_model=self.env_model,
            key=key_envs,
            num_worlds=num_envs,
        )

    def _env_fn(
        self,
        pipeline_model: PipelineModel,
        rgba_table: jax.Array,
        friction_table: jax.Array,
        stiffness_table: jax.Array,
    ) -> Env:
        env = self.env
        env.unwrapped._pipeline_model = pipeline_model
        env.unwrapped._rgba_table = rgba_table
        env.unwrapped._friction_table = friction_table
        env.unwrapped._stiffness_table = stiffness_table
        return env

    def reset(self, rng: jax.Array) -> State:
        def reset(pipeline_model, rgba_table, friction_table, stiffness_table, rng):
            env = self._env_fn(
                pipeline_model=pipeline_model,
                rgba_table=rgba_table,
                friction_table=friction_table,
                stiffness_table=stiffness_table,
            )
            return env.reset(rng)

        state = jax.vmap(reset, in_axes=[self._in_axes, 0, 0, 0, 0])(
            self._sys_v,
            self._rgba_table_v,
            self._friction_table_v,
            self._stiffness_table_v,
            rng,
        )
        return state

    def step(self, state: State, action: jax.Array) -> State:
        def step(
            pipeline_model,
            rgba_table,
            friction_table,
            stiffness_table,
            state_local,
            action_local,
        ):
            env = self._env_fn(
                pipeline_model=pipeline_model,
                rgba_table=rgba_table,
                friction_table=friction_table,
                stiffness_table=stiffness_table,
            )
            return env.step(state_local, action_local)

        res = jax.vmap(step, in_axes=[self._in_axes, 0, 0, 0, 0, 0])(
            self._sys_v,
            self._rgba_table_v,
            self._friction_table_v,
            self._stiffness_table_v,
            state,
            action,
        )
        return res

    # def init_vision_obs(
    #     self, pipeline_state: PipelineState, state_info: dict[str, Any]
    # ) -> None:
    #     def init_vision_obs(
    #         pipeline_model,
    #         rgba_table,
    #         friction_table,
    #         stiffness_table,
    #         local_pipeline_state,
    #         local_state_info,
    #     ):
    #         env = self._env_fn(
    #             pipeline_model=pipeline_model,
    #             rgba_table=rgba_table,
    #             friction_table=friction_table,
    #             stiffness_table=stiffness_table,
    #         )
    #         return env.init_vision_obs(local_pipeline_state, local_state_info)
    #     jax.vmap(init_vision_obs, in_axes=[self._in_axes, 0, 0, 0, 0, 0])(
    #         self._sys_v,
    #         self._rgba_table_v,
    #         self._friction_table_v,
    #         self._stiffness_table_v,
    #         pipeline_state,
    #         state_info,
    #     )

    def get_vision_obs(
        self,
        pipeline_state: PipelineState,
        state_info: dict[str, Any],
    ) -> Observation:
        def get_vision_obs(
            pipeline_model,
            rgba_table,
            friction_table,
            stiffness_table,
            local_pipeline_state,
            local_state_info,
        ):
            env = self._env_fn(
                pipeline_model=pipeline_model,
                rgba_table=rgba_table,
                friction_table=friction_table,
                stiffness_table=stiffness_table,
            )
            return env.get_vision_obs(local_pipeline_state, local_state_info)

        return jax.vmap(get_vision_obs, in_axes=[self._in_axes, 0, 0, 0, 0, 0])(
            self._sys_v,
            self._rgba_table_v,
            self._friction_table_v,
            self._stiffness_table_v,
            pipeline_state,
            state_info,
        )
