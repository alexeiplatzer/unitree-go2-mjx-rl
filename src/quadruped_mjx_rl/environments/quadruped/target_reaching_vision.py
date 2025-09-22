from collections.abc import Callable
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import mujoco

from quadruped_mjx_rl import math
from quadruped_mjx_rl.environments import QuadrupedBaseEnv
from quadruped_mjx_rl.environments.physics_pipeline import (
    EnvModel,
    EnvSpec,
    Motion,
    Transform,
    PipelineModel,
    PipelineState,
    State,
)
from quadruped_mjx_rl.environments.quadruped.base import (
    register_environment_config_class,
    EnvironmentConfig,
    QuadrupedBaseEnv,
)
from quadruped_mjx_rl.robotic_vision import VisionConfig
from quadruped_mjx_rl.robots import RobotConfig


def adjust_brightness(img, scale):
    """Adjusts the brightness of an image by scaling the pixel values."""
    return jnp.clip(img * scale, 0, 1)


@dataclass
class QuadrupedVisionTargetEnvConfig(EnvironmentConfig):
    use_vision: bool = True

    @dataclass
    class ObservationConfig(EnvironmentConfig.ObservationConfig):
        brightness: list[float] = field(default_factory=lambda: [0.75, 2.0])

    observation_noise: ObservationConfig = field(default_factory=ObservationConfig)

    @dataclass
    class DomainRandConfig(EnvironmentConfig.DomainRandConfig):
        apply_kicks: bool = False
        obstacle_location_noise: float = 3.0

    domain_rand: DomainRandConfig = field(default_factory=DomainRandConfig)

    @dataclass
    class RewardConfig(EnvironmentConfig.RewardConfig):
        goal_radius: float = 0.2
        obstacle_margin: float = 0.9
        collision_radius: float = 0.2
        yaw_alignment_threshold: float = 1.35

        @dataclass
        class ScalesConfig(EnvironmentConfig.RewardConfig.ScalesConfig):
            goal_progress: float = 2.5
            speed_towards_goal: float = 2.5
            obstacle_proximity: float = -1.0
            collision: float = -5.0
            goal_yaw_alignment: float = 1.0

        scales: ScalesConfig = field(default_factory=ScalesConfig)

    rewards: RewardConfig = field(default_factory=RewardConfig)

    @classmethod
    def config_class_key(cls) -> str:
        return "QuadrupedVisionTarget"

    @classmethod
    def get_environment_class(cls) -> type[QuadrupedBaseEnv]:
        return QuadrupedVisionTargetEnvironment


register_environment_config_class(QuadrupedVisionTargetEnvConfig)


class QuadrupedVisionTargetEnvironment(QuadrupedBaseEnv):

    def __init__(
        self,
        environment_config: QuadrupedVisionTargetEnvConfig,
        robot_config: RobotConfig,
        env_model: EnvSpec | EnvModel,
        vision_config: VisionConfig | None = None,
        init_qpos: jax.Array | None = None,
        renderer_maker: Callable[[PipelineModel], ...] | None = None,
    ):
        super().__init__(environment_config, robot_config, env_model)
        self._init_q = self._init_q if init_qpos is None else init_qpos

        self._goal_radius = environment_config.rewards.goal_radius
        self._obstacle_margin = environment_config.rewards.obstacle_margin
        self._collision_radius = environment_config.rewards.collision_radius

        self._obstacle_location_noise = environment_config.domain_rand.obstacle_location_noise

        self._goal_id = self._env_model.body("goal_sphere").id

        obstacle_names = ["cylinder_0", "cylinder_1"]
        self._obstacle_ids = [
            self._env_model.body(obstacle_name).id for obstacle_name in obstacle_names
        ]
        self._obstacles_qposadr = [
            self._env_model.jnt_qposadr[self._env_model.body(obstacle_name).jntadr[0]]
            for obstacle_name in obstacle_names
        ]

        self._use_vision = environment_config.use_vision
        if self._use_vision:
            if vision_config is None:
                raise ValueError("use_vision set to true, VisionConfig not provided.")
            self.renderer = renderer_maker(self.pipeline_model)

    @staticmethod
    def customize_model(
        env_model: EnvModel, environment_config: QuadrupedVisionTargetEnvConfig
    ):
        env_model = QuadrupedBaseEnv.customize_model(env_model, environment_config)
        floor_id = mujoco.mj_name2id(env_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        env_model.geom_size[floor_id, :2] = [25.0, 25.0]
        return env_model

    def reset(self, rng: jax.Array) -> State:
        state = super().reset(rng)

        state.info["goal_xy"] = state.pipeline_state.data.xpos[self._goal_id, :2]
        state.info["last_pos_xy"] = state.pipeline_state.x.pos[0, :2]

        state.info["obstacles_xy"] = jnp.stack([
            state.pipeline_state.data.xpos[obstacle_id, :2]
            for obstacle_id in self._obstacle_ids
        ])

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
            obstacle_pos = init_q[qadr + 1: qadr + 2]
            init_q = init_q.at[qadr + 1: qadr + 2].set(
                obstacle_pos + obstacle_offset
            )
        return init_q

    def _init_obs(
        self, pipeline_state: PipelineState, state_info: dict[str, ...]
    ) -> dict[str, jax.Array]:
        obs = {
            "state": QuadrupedBaseEnv._init_obs(self, pipeline_state, state_info),
        }
        if self._use_vision:
            rng = state_info["rng"]
            rng_brightness, rng = jax.random.split(rng)
            state_info["rng"] = rng
            brightness = jax.random.uniform(
                rng_brightness,
                (1,),
                minval=self._obs_config.brightness[0],
                maxval=self._obs_config.brightness[1],
            )
            state_info["brightness"] = brightness

            render_token, rgb, depth = self.renderer.init(
                pipeline_state.data, self._pipeline_model.model
            )
            state_info["render_token"] = render_token

            rgb_norm = jnp.asarray(rgb[0][..., :3], dtype=jnp.float32) / 255.0
            rgb_adjusted = adjust_brightness(rgb_norm, brightness)

            obs |= {"pixels/view_frontal_ego": rgb_adjusted, "pixels/view_terrain": depth[1]}

        return obs

    def _get_obs(
        self,
        pipeline_state: PipelineState,
        state_info: dict[str, ...],
        last_obs: jax.Array | dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        obs = {
            "state": QuadrupedBaseEnv._get_obs(
                self, pipeline_state, state_info, last_obs["state"]
            ),
        }
        if self._use_vision:
            _, rgb, depth = self.renderer.render(
                state_info["render_token"], pipeline_state.data
            )
            rgb_norm = jnp.asarray(rgb[0][..., :3], dtype=jnp.float32) / 255.0
            rgb_adjusted = adjust_brightness(rgb_norm, state_info["brightness"])
            obs |= {"pixels/view_frontal_ego": rgb_adjusted, "pixels/view_terrain": depth[1]}
        return obs

    def _get_rewards(
        self,
        pipeline_state: PipelineState,
        state_info: dict[str, ...],
        action: jax.Array,
        done: jax.Array,
    ) -> dict[str, jax.Array]:
        rewards = QuadrupedBaseEnv._get_rewards(self, pipeline_state, state_info, action, done)

        x, xd = pipeline_state.x, pipeline_state.xd

        goalwards_xy = state_info["goal_xy"] - x.pos[0, :2]
        last_goalwards_xy = state_info["goal_xy"] - state_info["last_pos_xy"]

        state_info["last_pos_xy"] = x.pos[0, :2]

        rewards["goal_progress"] = self._reward_goal_progress(
            last_goalwards_xy, goalwards_xy
        )
        rewards["speed_towards_goal"] = self._reward_speed_towards_goal(xd, goalwards_xy)
        rewards["goal_yaw_alignment"] = self._reward_goal_yaw_alignment(x, goalwards_xy)

        distances = jnp.linalg.norm(state_info["obstacles_xy"] - x.pos[0, :2], axis=-1)

        rewards["obstacle_proximity"] = self._reward_obstacle_proximity(distances)
        rewards["collision"] = self._reward_collision(distances)

        return rewards

    # ------------ reward functions ------------
    def _reward_goal_progress(
        self, last_goalwards_xy: jax.Array, goalwards_xy: jax.Array
    ) -> jax.Array:
        return jnp.linalg.norm(last_goalwards_xy) - jnp.linalg.norm(goalwards_xy)

    def _reward_speed_towards_goal(self, xd: Motion, goalwards_xy: jax.Array) -> jax.Array:
        goalwards_direction_xy = goalwards_xy / (jnp.linalg.norm(goalwards_xy) + 1e-6)
        return jnp.dot(goalwards_direction_xy, xd.vel[0, :2])

    def _reward_goal_yaw_alignment(
        self, x: Transform, goalwards_xy: jax.Array
    ) -> jax.Array:
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

    def _reward_obstacle_proximity(
        self, obstacle_distances: jax.Array
    ) -> jax.Array:
        return jnp.sum(jnp.maximum(self._obstacle_margin - obstacle_distances, 0.0))

    def _reward_collision(
        self, obstacle_distances: jax.Array
    ) -> jax.Array:
        return jnp.sum(obstacle_distances < self._collision_radius)

