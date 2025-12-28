from collections.abc import Callable
from typing import Any
from dataclasses import dataclass, field

import jax
from jax import numpy as jnp
from mujoco import mjx

from quadruped_mjx_rl.physics_pipeline import (
    PipelineModel,
    PipelineState,
)
from quadruped_mjx_rl.config_utils import Configuration, register_config_base_class
from quadruped_mjx_rl.physics_pipeline import Env, State, Wrapper
from quadruped_mjx_rl.environments.vision.robotic_vision import RendererConfig
from quadruped_mjx_rl.types import Observation, Action, PRNGKey


def adjust_brightness(img, scale):
    """Adjusts the brightness of an image by scaling the pixel values."""
    return jnp.clip(img * scale, 0, 1)


@dataclass
class VisionWrapperConfig(Configuration):
    renderer_config: RendererConfig = field(default_factory=RendererConfig)
    brightness: list[float] = field(default_factory=lambda: [0.75, 2.0])

    @dataclass
    class CameraInputConfig:
        name: str = "default"
        use_depth: bool = False
        use_brightness_randomized_rgb: bool = False
        use_actual_rgb: bool = False

    camera_inputs: list[CameraInputConfig] = field(
        default_factory=lambda: [
            VisionWrapperConfig.CameraInputConfig(
                name="frontal_ego", use_brightness_randomized_rgb=True
            ),
            VisionWrapperConfig.CameraInputConfig(name="terrain", use_depth=True),
        ]
    )

    @classmethod
    def config_base_class_key(cls) -> str:
        return "vision_wrapper"

    @classmethod
    def config_class_key(cls) -> str:
        return "VisionBaseWrapper"

    @classmethod
    def get_vision_wrapper_class(cls) -> type["VisionWrapper"]:
        return VisionWrapper

    @classmethod
    def _get_config_class_dict(cls) -> dict[str, type["Configuration"]]:
        return _vision_wrapper_config_classes


register_config_base_class(VisionWrapperConfig)

_vision_wrapper_config_classes = {}

register_vision_wrapper_config_class = VisionWrapperConfig.make_register_config_class()

register_vision_wrapper_config_class(VisionWrapperConfig)


class VisionWrapper(Wrapper):
    """A wrapper for an arbitrary MJX Environment that renders visual observations for it.
    Only renders the initial observations by default. Later observations must be updated
    explicitly."""

    def __init__(
        self,
        env: Env,
        vision_env_config: VisionWrapperConfig,
        renderer_maker: Callable[[PipelineModel], Any],
    ):
        super().__init__(env)
        if vision_env_config is None:
            raise ValueError("use_vision set to true, VisionEnvConfig not provided.")
        self._brightness_scaling = vision_env_config.brightness
        self._camera_inputs_config = vision_env_config.camera_inputs

        # Execute one environment step to initialize mjx before madrona
        mjx_model = mjx.put_model(self.env_model)
        mjx_data = mjx.make_data(mjx_model)
        _ = mjx.forward(mjx_model, mjx_data)

        self.renderer = renderer_maker(self.pipeline_model)

    def reset(self, rng: PRNGKey) -> State:
        state = self.env.reset(rng)
        # to update the state info
        self.init_vision_obs(state.pipeline_state, state.info)
        return state

    def init_vision_obs(
        self, pipeline_state: PipelineState, state_info: dict[str, Any]
    ) -> None:
        rng = state_info["rng"]
        rng_brightness, rng = jax.random.split(rng)
        state_info["rng"] = rng
        brightness = jax.random.uniform(
            rng_brightness,
            (1,),
            minval=self._brightness_scaling[0],
            maxval=self._brightness_scaling[1],
        )
        state_info["brightness"] = brightness

        render_token, _, _ = self.renderer.init(pipeline_state.data, self.pipeline_model.model)
        state_info["render_token"] = render_token

    def get_vision_obs(
        self,
        pipeline_state: PipelineState,
        state_info: dict[str, Any],
    ) -> Observation:
        _, rgb, depth = self.renderer.render(state_info["render_token"], pipeline_state.data)
        return self._format_camera_observations(rgb, depth, state_info)

    def _format_camera_observations(
        self, rgb_inputs: jax.Array, depth_inputs: jax.Array, state_info: dict[str, Any]
    ) -> Observation:
        """Prepares rgb and depth observations from cameras' inputs."""
        camera_obs = {}
        for idx, camera_input_config in enumerate(self._camera_inputs_config):
            if camera_input_config.use_depth:
                camera_obs[f"pixels/{camera_input_config.name}/depth"] = depth_inputs[idx]
            if camera_input_config.use_actual_rgb:
                rgb = jnp.asarray(rgb_inputs[idx][..., :3], dtype=jnp.float32) / 255.0
                camera_obs[f"pixels/{camera_input_config.name}/rgb"] = rgb
            if camera_input_config.use_brightness_randomized_rgb:
                rgb = jnp.asarray(rgb_inputs[idx][..., :3], dtype=jnp.float32) / 255.0
                rgb_adjusted = adjust_brightness(rgb, state_info["brightness"])
                camera_obs[f"pixels/{camera_input_config.name}/rgb_adjusted"] = rgb_adjusted
        return camera_obs
