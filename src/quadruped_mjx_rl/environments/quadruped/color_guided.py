from collections.abc import Callable
from dataclasses import dataclass, field

import jax
from jax import numpy as jnp

from quadruped_mjx_rl.domain_randomization.randomized_tiles import color_meaning_fn
from quadruped_mjx_rl.environments.physics_pipeline import (
    EnvModel,
    EnvSpec,
    PipelineModel,
    PipelineState, State,
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
class QuadrupedColorGuidedEnvConfig(QuadrupedVisionTargetEnvConfig):
    @dataclass
    class ObservationConfig(QuadrupedVisionTargetEnvConfig.ObservationConfig):
        camera_inputs: list[
            QuadrupedVisionTargetEnvConfig.ObservationConfig.CameraInputConfig
        ] = field(
            default_factory=lambda: [
                QuadrupedVisionTargetEnvConfig.ObservationConfig.CameraInputConfig(
                    name="frontal_ego", use_brightness_randomized_rgb=True
                ),
                QuadrupedVisionTargetEnvConfig.ObservationConfig.CameraInputConfig(
                    name="terrain", use_actual_rgb=True
                ),
            ]
        )

    observation_noise: ObservationConfig = field(default_factory=ObservationConfig)

    @classmethod
    def config_class_key(cls) -> str:
        return "QuadrupedColorGuided"

    @classmethod
    def get_environment_class(cls) -> type[QuadrupedBaseEnv]:
        return QuadrupedColorGuidedEnv


register_environment_config_class(QuadrupedColorGuidedEnvConfig)


class QuadrupedColorGuidedEnv(QuadrupedVisionTargetEnv):
    """An expansion of the target-reaching environment to include obstacles, collision with
    which is punished."""

    def __init__(
        self,
        environment_config: QuadrupedColorGuidedEnvConfig,
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
        # the initial values and sizes should be set up by the terrain map wrapper
        self._rgba_table = jnp.array(())
        self._friction_table = jnp.array(())
        self._stiffness_table = jnp.array(())

    def _init_obs(
        self, pipeline_state: PipelineState, state_info: dict[str, ...]
    ) -> dict[str, jax.Array]:
        obs = super()._init_obs(pipeline_state, state_info)

        return obs

    def _get_obs(
        self,
        pipeline_state: PipelineState,
        state_info: dict[str, ...],
        previous_obs: dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        obs = super()._get_obs(pipeline_state, state_info, previous_obs)

        return obs

    def _privileged_terrain_map(self, terrain_rgb: jax.Array) -> jax.Array:
        # TODO apply the color meaning fn to every pixel and get an image of the friction
        # and stiffenss values and return then as an image with two channels of the same size
        # so we get from (WxHx4) to (WxHx2), 4 is for rgba, 2 is for friction and stiffness
        # the rgba table and friction and stiffenss tables are stored in the env and set
        # to the correct values with the Terrain Map wrapper
        pass
