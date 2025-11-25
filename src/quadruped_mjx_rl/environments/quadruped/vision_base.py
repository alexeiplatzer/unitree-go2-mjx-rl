from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from quadruped_mjx_rl.environments.physics_pipeline import (
    EnvModel,
    EnvSpec,
    PipelineModel,
    PipelineState,
)
from quadruped_mjx_rl.environments.quadruped.base import (
    EnvironmentConfig as EnvCfg,
    QuadrupedBaseEnv,
    register_environment_config_class,
)
from quadruped_mjx_rl.environments.vision.robotic_vision import VisionConfig
from quadruped_mjx_rl.robots import RobotConfig


def adjust_brightness(img, scale):
    """Adjusts the brightness of an image by scaling the pixel values."""
    return jnp.clip(img * scale, 0, 1)


class QuadrupedVisionBaseEnvConfig(EnvCfg):
    use_vision: bool = True

    @dataclass
    class ObservationConfig(EnvCfg.ObservationConfig):
        @dataclass
        class CameraInputConfig:
            name: str = "default"
            use_depth: bool = False
            use_brightness_randomized_rgb: bool = False
            use_actual_rgb: bool = False

        brightness: list[float] = field(default_factory=lambda: [0.75, 2.0])

        camera_inputs: list[CameraInputConfig] = field(
            default_factory=lambda: [
                QuadrupedVisionBaseEnvConfig.ObservationConfig.CameraInputConfig(
                    name="frontal_ego", use_brightness_randomized_rgb=True
                ),
                QuadrupedVisionBaseEnvConfig.ObservationConfig.CameraInputConfig(
                    name="terrain", use_depth=True
                ),
            ]
        )

    observation_noise: ObservationConfig = field(default_factory=ObservationConfig)

    @classmethod
    def config_class_key(cls) -> str:
        return "VisionBase"

    @classmethod
    def get_environment_class(cls) -> type[QuadrupedBaseEnv]:
        return QuadrupedVisionBaseEnv


register_environment_config_class(QuadrupedVisionBaseEnvConfig)


class QuadrupedVisionBaseEnv(QuadrupedBaseEnv):
    """Extends :class:`QuadrupedBaseEnv` with rendered camera observations."""

    def __init__(
        self,
        environment_config: QuadrupedVisionBaseEnvConfig,
        robot_config: RobotConfig,
        env_model: EnvSpec | EnvModel,
        *,
        vision_config: VisionConfig | None = None,
        renderer_maker: Callable[[PipelineModel], Any] | None = None,
    ) -> None:
        super().__init__(environment_config, robot_config, env_model)
        self._use_vision = environment_config.use_vision
        if self._use_vision:
            if vision_config is None:
                raise ValueError("use_vision set to true, VisionConfig not provided.")
            self._brightness_scaling = environment_config.observation_noise.brightness
            self._camera_inputs_config = environment_config.observation_noise.camera_inputs

            # Execute one environment step to initialize mjx before madrona
            mjx_model = mjx.put_model(env_model)
            mjx_data = mjx.make_data(mjx_model)
            _ = mjx.forward(mjx_model, mjx_data)

            self.renderer = renderer_maker(self.pipeline_model)

    @staticmethod
    def customize_model(env_model: EnvModel, environment_config: QuadrupedVisionBaseEnvConfig):
        env_model = QuadrupedBaseEnv.customize_model(env_model, environment_config)
        floor_id = mujoco.mj_name2id(env_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        env_model.geom_size[floor_id, :2] = [25.0, 25.0]
        return env_model

    def _set_init_qpos(self, rng: jax.Array) -> jax.Array:
        # vectorize the init qpos across batch dimensions
        return self._init_q + jax.random.uniform(
            key=rng, shape=self._init_q.shape, minval=0.0, maxval=0.0
        )

    def _init_obs(
        self, pipeline_state: PipelineState, state_info: dict[str, Any]
    ) -> dict[str, jax.Array]:
        obs = {"proprioceptive": self._init_proprioceptive_obs(pipeline_state, state_info)}
        if self._use_vision:
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

            render_token, rgb, depth = self.renderer.init(
                pipeline_state.data, self._pipeline_model.model
            )
            state_info["render_token"] = render_token

            camera_obs = self._format_camera_observations(rgb, depth, state_info)
            obs |= camera_obs
        return obs

    def _get_obs(
        self,
        pipeline_state: PipelineState,
        state_info: dict[str, Any],
        previous_obs: dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        obs = {"proprioceptive": self._init_proprioceptive_obs(pipeline_state, state_info)}
        if self._use_vision:
            _, rgb, depth = self.renderer.init(pipeline_state.data, self._pipeline_model.model)
            camera_obs = self._format_camera_observations(rgb, depth, state_info)
            obs |= camera_obs
        return obs

    def _format_camera_observations(
        self, rgb_inputs: jax.Array, depth_inputs: jax.Array, state_info: dict[str, Any]
    ) -> dict[str, jax.Array]:
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
