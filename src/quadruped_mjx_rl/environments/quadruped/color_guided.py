from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax
from jax import numpy as jnp

from quadruped_mjx_rl.domain_randomization.randomized_tiles import color_meaning_fn
from quadruped_mjx_rl.environments.physics_pipeline import (
    EnvModel,
    EnvSpec,
    PipelineModel,
    PipelineState,
)
from quadruped_mjx_rl.environments.quadruped.base import (
    register_environment_config_class,
    QuadrupedBaseEnv,
)
from quadruped_mjx_rl.environments.quadruped.target_reaching import (
    QuadrupedVisionTargetEnvConfig,
    QuadrupedVisionTargetEnv,
)
from quadruped_mjx_rl.environments.vision.robotic_vision import VisionConfig
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
        renderer_maker: Callable[[PipelineModel], Any] | None = None,
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
        self, pipeline_state: PipelineState, state_info: dict[str, Any]
    ) -> dict[str, jax.Array]:
        obs = super()._init_obs(pipeline_state, state_info)
        obs["privileged_terrain_map"] = self._privileged_terrain_map(obs["pixels/terrain/rgb"])
        return obs

    def _get_obs(
        self,
        pipeline_state: PipelineState,
        state_info: dict[str, Any],
        previous_obs: dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        obs = super()._get_obs(pipeline_state, state_info, previous_obs)
        obs["privileged_terrain_map"] = self._privileged_terrain_map(obs["pixels/terrain/rgb"])
        return obs

    def _privileged_terrain_map(self, terrain_rgba: jax.Array) -> jax.Array:
        """apply the color meaning fn to every pixel and get an image of the friction
        and stiffness values and return then as an image with two channels of the same size.
        So we get from (WxHx4) to (WxHx2), 4 is for rgba, 2 is for friction and stiffness.
        The rgba table and friction and stiffness tables are stored in the env and set
        to the correct values with the Terrain Map wrapper."""
        # flat_rgba = terrain_rgba.reshape(-1, terrain_rgba.shape[-1])
        return jax.vmap(
            lambda rgba: jnp.stack(
                color_meaning_fn(
                    rgba=rgba,
                    rgba_table=self._rgba_table,
                    friction_table=self._friction_table,
                    stiffness_table=self._stiffness_table,
                )
            )
        )(terrain_rgba)
        # return jnp.stack([terrain_friction, terrain_stiffness], axis=-1).reshape(
        #     terrain_rgba.shape[:-1] + (2,)
        # )
