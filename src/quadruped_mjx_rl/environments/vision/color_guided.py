from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from jax import numpy as jnp
import jax

from quadruped_mjx_rl.physics_pipeline import PipelineModel, PipelineState
from quadruped_mjx_rl.environments.base import Env
from quadruped_mjx_rl.domain_randomization.randomized_tiles import color_meaning_fn
from quadruped_mjx_rl.environments.vision.vision_wrappers import (
    VisionWrapper,
    VisionEnvConfig,
)
from quadruped_mjx_rl.types import Observation


@dataclass
class ColorGuidedEnvConfig(VisionEnvConfig):
    """Configuration for a Color Guided Vision Environment."""

    camera_inputs: list[VisionEnvConfig.CameraInputConfig] = field(
        default_factory=lambda: [
            VisionEnvConfig.CameraInputConfig(
                name="frontal_ego", use_brightness_randomized_rgb=True
            ),
            VisionEnvConfig.CameraInputConfig(name="terrain", use_actual_rgb=True),
        ]
    )

    @classmethod
    def get_vision_wrapper_class(cls) -> type["VisionWrapper"]:
        return ColorGuidedVisionWrapper


class ColorGuidedVisionWrapper(VisionWrapper):
    def __init__(
        self,
        env: Env,
        vision_env_config: VisionEnvConfig,
        renderer_maker: Callable[[PipelineModel], Any],
    ):
        super().__init__(env, vision_env_config, renderer_maker)
        # the initial values and sizes should be set up by the terrain map wrapper
        self._rgba_table = jnp.array(())
        self._friction_table = jnp.array(())
        self._stiffness_table = jnp.array(())

    # def init_vision_obs(
    #     self, pipeline_state: PipelineState, state_info: dict[str, Any]
    # ) -> tuple[Observation, dict[str, Any]]:
    #     obs, state_info = super().init_vision_obs(pipeline_state, state_info)
    #     obs["privileged_terrain_map"] = self._privileged_terrain_map(obs["pixels/terrain/rgb"])
    #     return obs, state_info

    def get_vision_obs(
        self,
        pipeline_state: PipelineState,
        state_info: dict[str, Any],
    ) -> Observation:
        obs = super().get_vision_obs(pipeline_state, state_info)
        obs["privileged_terrain_map"] = self._privileged_terrain_map(obs["pixels/terrain/rgb"])
        return obs

    def _privileged_terrain_map(self, terrain_rgba: jax.Array) -> jax.Array:
        """apply the color meaning fn to every pixel and get an image of the friction
        and stiffness values and return then as an image with two channels of the same size.
        So we get from (WxHx4) to (WxHx2), 4 is for rgba, 2 is for friction and stiffness.
        The rgba table and friction and stiffness tables are stored in the env and set
        to the correct values with the Terrain Map wrapper."""
        # flat_rgba = terrain_rgba.reshape(-1, terrain_rgba.shape[-1])
        return jnp.moveaxis(
            jnp.stack(
                jax.vmap(
                    jax.vmap(color_meaning_fn, in_axes=(0, None, None, None)),
                    in_axes=(0, None, None, None),
                )(
                    terrain_rgba,
                    self.env.unwrapped._rgba_table,
                    self.env.unwrapped._friction_table,
                    self.env.unwrapped._stiffness_table,
                )
            ),
            -3,
            -1,
        )
        # return jax.vmap(jax.vmap(
        #     lambda rgba: jnp.stack(
        #         color_meaning_fn(
        #             rgba=rgba,
        #             rgba_table=self._rgba_table,
        #             friction_table=self._friction_table,
        #             stiffness_table=self._stiffness_table,
        #         )
        #     )
        # ))(terrain_rgba)
        # return jnp.stack([terrain_friction, terrain_stiffness], axis=-1).reshape(
        #     terrain_rgba.shape[:-1] + (2,)
        # )
