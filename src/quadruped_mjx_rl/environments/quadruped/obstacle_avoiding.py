from collections.abc import Callable
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from quadruped_mjx_rl.environments.physics_pipeline import (
    EnvModel,
    EnvSpec,
    PipelineModel,
    PipelineState,
    State,
)
from quadruped_mjx_rl.environments.quadruped.base import (
    register_environment_config_class,
    QuadrupedBaseEnv,
)
from quadruped_mjx_rl.environments.quadruped.target_reaching import (
    QuadrupedVisionTargetEnvConfig,
    QuadrupedVisionTargetEnv,
)
from quadruped_mjx_rl.robotic_vision import VisionConfig
from quadruped_mjx_rl.robots import RobotConfig


@dataclass
class QuadrupedObstacleAvoidingEnvConfig(QuadrupedVisionTargetEnvConfig):
    @dataclass
    class DomainRandConfig(QuadrupedVisionTargetEnvConfig.DomainRandConfig):
        obstacle_location_noise: float = 3.0

    domain_rand: DomainRandConfig = field(default_factory=DomainRandConfig)

    @dataclass
    class RewardConfig(QuadrupedVisionTargetEnvConfig.RewardConfig):
        goal_radius: float = 0.2
        obstacle_margin: float = 0.9
        collision_radius: float = 0.2
        yaw_alignment_threshold: float = 1.35

        @dataclass
        class ScalesConfig(QuadrupedVisionTargetEnvConfig.RewardConfig.ScalesConfig):
            goal_progress: float = 2.5
            speed_towards_goal: float = 2.5
            obstacle_proximity: float = -1.0
            collision: float = -5.0
            goal_yaw_alignment: float = 1.0

        scales: ScalesConfig = field(default_factory=ScalesConfig)

    rewards: RewardConfig = field(default_factory=RewardConfig)

    @classmethod
    def config_class_key(cls) -> str:
        return "QuadrupedObstacleAvoiding"

    @classmethod
    def get_environment_class(cls) -> type[QuadrupedBaseEnv]:
        return QuadrupedObstacleAvoidingEnv


register_environment_config_class(QuadrupedObstacleAvoidingEnvConfig)


class QuadrupedObstacleAvoidingEnv(QuadrupedVisionTargetEnv):
    """An expansion of the target-reaching environment to include obstacles, collision with
    which is punished."""

    def __init__(
        self,
        environment_config: QuadrupedObstacleAvoidingEnvConfig,
        robot_config: RobotConfig,
        env_model: EnvSpec | EnvModel,
        vision_config: VisionConfig | None = None,
        init_qpos: jax.Array | None = None,
        renderer_maker: Callable[[PipelineModel], ...] | None = None,
    ):
        super().__init__(
            environment_config=environment_config,
            robot_config=robot_config,
            env_model=env_model,
            vision_config=vision_config,
            init_qpos=init_qpos,
            renderer_maker=renderer_maker,
        )

        self._obstacle_margin = environment_config.rewards.obstacle_margin
        self._collision_radius = environment_config.rewards.collision_radius

        self._obstacle_location_noise = environment_config.domain_rand.obstacle_location_noise

        obstacle_names = ["cylinder_0", "cylinder_1"]
        self._obstacle_ids = [
            self._env_model.body(obstacle_name).id for obstacle_name in obstacle_names
        ]
        self._obstacles_qposadr = [
            self._env_model.jnt_qposadr[self._env_model.body(obstacle_name).jntadr[0]]
            for obstacle_name in obstacle_names
        ]

    def reset(self, rng: jax.Array) -> State:
        state = super().reset(rng)

        state.info["obstacles_xy"] = jnp.stack(
            [
                state.pipeline_state.data.xpos[obstacle_id, :2]
                for obstacle_id in self._obstacle_ids
            ]
        )

        return state

    def _set_init_qpos(self, rng: jax.Array) -> jax.Array:
        # perturb obstacle locations
        init_q = self._init_q
        for qadr in self._obstacles_qposadr:
            rng, obstacle_key = jax.random.split(rng, 2)
            obstacle_offset = jax.random.uniform(
                obstacle_key,
                (1,),
                minval=-self._obstacle_location_noise,
                maxval=self._obstacle_location_noise,
            )
            obstacle_pos = init_q[qadr + 1 : qadr + 2]
            init_q = init_q.at[qadr + 1 : qadr + 2].set(obstacle_pos + obstacle_offset)
        return init_q

    def _get_rewards(
        self,
        pipeline_state: PipelineState,
        state_info: dict[str, ...],
        action: jax.Array,
        done: jax.Array,
    ) -> dict[str, jax.Array]:
        rewards = QuadrupedVisionTargetEnv._get_rewards(
            self, pipeline_state, state_info, action, done
        )

        x, xd = pipeline_state.x, pipeline_state.xd

        distances = jnp.linalg.norm(state_info["obstacles_xy"] - x.pos[0, :2], axis=-1)

        rewards["obstacle_proximity"] = self._reward_obstacle_proximity(distances)
        rewards["collision"] = self._reward_collision(distances)

        return rewards

    # ------------ reward functions ------------
    def _reward_obstacle_proximity(self, obstacle_distances: jax.Array) -> jax.Array:
        return jnp.sum(jnp.maximum(self._obstacle_margin - obstacle_distances, 0.0))

    def _reward_collision(self, obstacle_distances: jax.Array) -> jax.Array:
        return jnp.sum(obstacle_distances < self._collision_radius)
