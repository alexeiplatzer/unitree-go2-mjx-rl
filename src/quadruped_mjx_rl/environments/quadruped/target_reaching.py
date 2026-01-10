from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp

from quadruped_mjx_rl import math
from quadruped_mjx_rl.physics_pipeline import (
    EnvModel,
    EnvSpec,
    Motion,
    Transform,
    PipelineState,
    State,
)
from quadruped_mjx_rl.environments.quadruped.base import (
    register_environment_config_class,
    QuadrupedBaseEnv,
    EnvironmentConfig,
)
from quadruped_mjx_rl.robots import RobotConfig
from quadruped_mjx_rl.types import Action, Observation, PRNGKey
from quadruped_mjx_rl import math


@dataclass
class QuadrupedVisionTargetEnvConfig(EnvironmentConfig):

    domain_rand: "QuadrupedVisionTargetEnvConfig.DomainRandConfig" = field(
        default_factory=lambda: QuadrupedVisionTargetEnvConfig.DomainRandConfig(
            apply_kicks=False
        )
    )

    @dataclass
    class RewardConfig(EnvironmentConfig.RewardConfig):
        yaw_alignment_threshold: float = 1.35
        success_radius: float = 2.5  # TODO: configure with goal size

        @dataclass
        class ScalesConfig(EnvironmentConfig.RewardConfig.ScalesConfig):
            goal_progress: float = 2.5
            speed_towards_goal: float = 2.5
            goal_yaw_alignment: float = 1.0
            goal_reached: float = 5.0

        scales: ScalesConfig = field(default_factory=ScalesConfig)

    rewards: RewardConfig = field(default_factory=RewardConfig)

    @classmethod
    def config_class_key(cls) -> str:
        return "QuadrupedVisionTarget"

    @classmethod
    def get_environment_class(cls) -> type[QuadrupedBaseEnv]:
        return QuadrupedVisionTargetEnv


register_environment_config_class(QuadrupedVisionTargetEnvConfig)


class QuadrupedVisionTargetEnv(QuadrupedBaseEnv):
    """In this environment, the robot is tasked with reaching a target body using both visual
    and proprioceptive input."""

    def __init__(
        self,
        environment_config: QuadrupedVisionTargetEnvConfig,
        robot_config: RobotConfig,
        env_model: EnvSpec | EnvModel,
    ):
        super().__init__(
            environment_config,
            robot_config,
            env_model,
        )
        self._rewards_config = environment_config.rewards
        self._goal_id = self._env_model.body("goal_sphere").id

    @staticmethod
    def _update_vectors(pipeline_state: PipelineState, state_info: dict[str, Any]) -> None:
        state_info["goal_dir_world"] = state_info["goal_pos"] - pipeline_state.x.pos[0]
        inv_torso_rot = math.quat_inv(pipeline_state.x.rot[0])
        state_info["goal_dir_local"] = math.rotate(state_info["goal_dir_world"], inv_torso_rot)

    def _init_obs(
        self,
        pipeline_state: PipelineState,
        state_info: dict[str, Any],
    ) -> Observation:
        obs = super()._init_obs(pipeline_state, state_info)
        state_info["goal_pos"] = pipeline_state.data.xpos[self._goal_id]
        self._update_vectors(pipeline_state, state_info)
        state_info["last_pos"] = pipeline_state.x.pos[0]
        obs["goal_direction"] = state_info["goal_dir_local"]
        return obs

    def _get_obs(
        self,
        pipeline_state: PipelineState,
        state_info: dict[str, Any],
        previous_obs: Observation,
    ) -> Observation:
        obs = super()._get_obs(
            pipeline_state=pipeline_state, state_info=state_info, previous_obs=previous_obs
        )
        self._update_vectors(pipeline_state, state_info)
        obs["goal_direction"] = state_info["goal_dir_local"]
        return obs

    def _check_success(self, pipeline_state: PipelineState) -> jax.Array:
        goal_pos = pipeline_state.data.xpos[self._goal_id]
        curr_pos = pipeline_state.x.pos[0]
        dist = jnp.linalg.norm(goal_pos - curr_pos)
        return dist < self._rewards_config.success_radius

    def _check_termination(self, pipeline_state: PipelineState) -> jax.Array:
        # 1. Check for standard failures (falling, flipping)
        failure = super()._check_termination(pipeline_state)

        # 2. Check for success (reaching the goal)
        success = self._check_success(pipeline_state)

        # Terminate on EITHER failure OR success
        return failure | success

    def _get_rewards(
        self,
        pipeline_state: PipelineState,
        state_info: dict[str, Any],
        action: Action,
        done: jax.Array,
    ) -> dict[str, jax.Array]:
        rewards = QuadrupedBaseEnv._get_rewards(self, pipeline_state, state_info, action, done)

        x, xd = pipeline_state.x, pipeline_state.xd

        goal_dir = state_info["goal_dir_world"]
        last_goal_dir = state_info["goal_pos"] - state_info["last_pos"]

        rewards["goal_progress"] = self._reward_goal_progress(last_goal_dir, goal_dir)
        rewards["speed_towards_goal"] = self._reward_speed_towards_goal(xd, goal_dir)
        rewards["goal_yaw_alignment"] = self._reward_goal_yaw_alignment(x, goal_dir)

        success = self._check_success(pipeline_state)
        rewards["termination"] = jnp.where(success, 0.0, rewards["termination"])
        rewards["goal_reached"] = jnp.where(success, 1.0, 0.0)

        state_info["last_pos"] = x.pos[0]

        return rewards

    # ------------ reward functions ------------
    def _reward_goal_progress(
        self, last_goal_dir: jax.Array, goal_dir: jax.Array
    ) -> jax.Array:
        return jnp.linalg.norm(last_goal_dir) - jnp.linalg.norm(goal_dir)

    def _reward_speed_towards_goal(self, xd: Motion, goal_dir: jax.Array) -> jax.Array:
        goalwards_direction_xy = goal_dir / (jnp.linalg.norm(goal_dir) + 1e-6)
        return jnp.dot(goalwards_direction_xy, xd.vel[0])

    def _reward_goal_yaw_alignment(self, x: Transform, goal_dir: jax.Array) -> jax.Array:
        forward = math.rotate(jnp.array([1.0, 0.0, 0.0]), x.rot[0])
        forward_direction = forward / (jnp.linalg.norm(forward) + 1e-6)
        goal_direction = goal_dir / (jnp.linalg.norm(goal_dir) + 1e-6)

        cos_angle = jnp.clip(jnp.dot(forward_direction, goal_direction), -1.0, 1.0)
        angle = jnp.arccos(cos_angle)

        return jnp.where(
            angle <= self._rewards_config.yaw_alignment_threshold,
            1.0 - angle / self._rewards_config.yaw_alignment_threshold,
            0.0,
        )
