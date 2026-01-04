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

        @dataclass
        class ScalesConfig(EnvironmentConfig.RewardConfig.ScalesConfig):
            goal_progress: float = 2.5
            speed_towards_goal: float = 2.5
            goal_yaw_alignment: float = 1.0

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

    def _init_obs(
        self,
        pipeline_state: PipelineState,
        state_info: dict[str, Any],
    ) -> Observation:
        obs = super()._init_obs(pipeline_state, state_info)
        state_info["goal_xy"] = pipeline_state.data.xpos[self._goal_id, :2]
        state_info["last_pos_xy"] = pipeline_state.x.pos[0, :2]
        state_info["goalwards_xy"] = state_info["goal_xy"] - pipeline_state.x.pos[0, :2]
        obs["goalwards_xy"] = state_info["goalwards_xy"]
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
        state_info["goalwards_xy"] = state_info["goal_xy"] - pipeline_state.x.pos[0, :2]
        obs["goalwards_xy"] = state_info["goalwards_xy"]
        return obs

    def _get_rewards(
        self,
        pipeline_state: PipelineState,
        state_info: dict[str, Any],
        action: Action,
        done: jax.Array,
    ) -> dict[str, jax.Array]:
        rewards = QuadrupedBaseEnv._get_rewards(self, pipeline_state, state_info, action, done)

        x, xd = pipeline_state.x, pipeline_state.xd

        goalwards_xy = state_info["goalwards_xy"]
        last_goalwards_xy = state_info["goal_xy"] - state_info["last_pos_xy"]

        state_info["last_pos_xy"] = x.pos[0, :2]

        rewards["goal_progress"] = self._reward_goal_progress(last_goalwards_xy, goalwards_xy)
        rewards["speed_towards_goal"] = self._reward_speed_towards_goal(xd, goalwards_xy)
        rewards["goal_yaw_alignment"] = self._reward_goal_yaw_alignment(x, goalwards_xy)

        return rewards

    # ------------ reward functions ------------
    def _reward_goal_progress(
        self, last_goalwards_xy: jax.Array, goalwards_xy: jax.Array
    ) -> jax.Array:
        return jnp.linalg.norm(last_goalwards_xy) - jnp.linalg.norm(goalwards_xy)

    def _reward_speed_towards_goal(self, xd: Motion, goalwards_xy: jax.Array) -> jax.Array:
        goalwards_direction_xy = goalwards_xy / (jnp.linalg.norm(goalwards_xy) + 1e-6)
        return jnp.dot(goalwards_direction_xy, xd.vel[0, :2])

    def _reward_goal_yaw_alignment(self, x: Transform, goalwards_xy: jax.Array) -> jax.Array:
        forward = math.rotate(jnp.array([1.0, 0.0, 0.0]), x.rot[0])[:2]
        forward_direction = forward / (jnp.linalg.norm(forward) + 1e-6)
        goal_direction = goalwards_xy / (jnp.linalg.norm(goalwards_xy) + 1e-6)

        cos_angle = jnp.clip(jnp.dot(forward_direction, goal_direction), -1.0, 1.0)
        angle = jnp.arccos(cos_angle)

        return jnp.where(
            angle <= self._rewards_config.yaw_alignment_threshold,
            1.0 - angle / self._rewards_config.yaw_alignment_threshold,
            0.0,
        )
