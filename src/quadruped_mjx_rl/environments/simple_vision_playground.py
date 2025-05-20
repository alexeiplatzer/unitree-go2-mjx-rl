from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from brax.base import State as PipelineState
from brax.base import System
from etils.epath import PathLike

from quadruped_mjx_rl.environments.base import configs_to_env_classes
from quadruped_mjx_rl.environments.base import environment_config_classes
from quadruped_mjx_rl.environments.ppo_teacher_student import QuadrupedJoystickTeacherStudentEnv
from quadruped_mjx_rl.environments.ppo_teacher_student import TeacherStudentEnvironmentConfig
from quadruped_mjx_rl.robotic_vision import VisionConfig
from quadruped_mjx_rl.robots import RobotConfig

_ENVIRONMENT_CLASS = "QuadrupedVision"


def adjust_brightness(img, scale):
    """Adjusts the brightness of an image by scaling the pixel values."""
    return jnp.clip(img * scale, 0, 1)


@dataclass
class QuadrupedVisionEnvConfig(TeacherStudentEnvironmentConfig):
    environment_class = _ENVIRONMENT_CLASS
    use_vision: bool = True

    @dataclass
    class ObservationNoiseConfig(TeacherStudentEnvironmentConfig.ObservationNoiseConfig):
        brightness: list[float] = field(default_factory=lambda: [1.0, 1.0])

    obs_noise: ObservationNoiseConfig = field(default_factory=ObservationNoiseConfig)


environment_config_classes[_ENVIRONMENT_CLASS] = QuadrupedVisionEnvConfig


class QuadrupedVisionEnvironment(QuadrupedJoystickTeacherStudentEnv):

    def __init__(
        self,
        environment_config: QuadrupedVisionEnvConfig,
        robot_config: RobotConfig,
        init_scene_path: PathLike,
        vision_config: VisionConfig | None = None,
    ):
        super().__init__(environment_config, robot_config, init_scene_path)

        self._use_vision = environment_config.use_vision
        if self._use_vision:
            if vision_config is None:
                raise ValueError("Use vision set to true, VisionConfig not provided.")

            from madrona_mjx.renderer import BatchRenderer

            self.renderer = BatchRenderer(
                m=self.sys,
                gpu_id=vision_config.gpu_id,
                num_worlds=vision_config.render_batch_size,
                batch_render_view_width=vision_config.render_width,
                batch_render_view_height=vision_config.render_height,
                enabled_geom_groups=np.asarray(vision_config.enabled_geom_groups),
                enabled_cameras=np.asarray(vision_config.enabled_cameras),
                add_cam_debug_geo=False,
                use_rasterizer=vision_config.use_rasterizer,
                viz_gpu_hdls=None,
            )

    def _override_menagerie_params(
        self, sys: System, environment_config: QuadrupedVisionEnvConfig
    ) -> System:
        sys = super()._override_menagerie_params(sys, environment_config)

        # TODO: verify that this works
        floor_id = mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        sys = sys.replace(geom_size=sys.geom_size.at[floor_id, :2].set([5.0, 5.0]))

        return sys

    def _init_obs(
        self, pipeline_state: PipelineState, state_info: dict[str, ...]
    ) -> dict[str, jax.Array]:
        if not self._use_vision:
            obs = super()._init_obs(pipeline_state, state_info)
        else:
            rng = state_info["rng"]
            rng_brightness, rng = jax.random.split(rng)
            state_info["rng"] = rng
            brightness = jax.random.uniform(
                rng_brightness,
                (1,),
                minval=self._obs_noise_config.brightness[0],
                maxval=self._obs_noise_config.brightness[1],
            )
            state_info["brightness"] = brightness

            render_token, rgb, depth = self.renderer.init(pipeline_state, self.sys)
            state_info["render_token"] = render_token

            rgb_norm = jnp.asarray(rgb[1][..., :3], dtype=jnp.float32) / 255.0
            rgb_adjusted = adjust_brightness(rgb_norm, brightness)

            obs = {"pixels/view_frontal_ego": rgb_adjusted, "pixels/view_terrain": depth[2]}

        return obs

    def _get_obs(
        self,
        pipeline_state: PipelineState,
        state_info: dict[str, ...],
        last_obs: jax.Array | dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        if not self._use_vision:
            obs = super()._get_obs(pipeline_state, state_info, last_obs)
        else:
            _, rgb, depth = self.renderer.render(state_info["render_token"], pipeline_state)
            rgb_norm = jnp.asarray(rgb[0][..., :3], dtype=jnp.float32) / 255.0
            rgb_adjusted = adjust_brightness(rgb_norm, state_info["brightness"])
            obs = {"pixels/view_frontal_ego": rgb_adjusted, "pixels/view_terrain": depth[2]}
        return obs


configs_to_env_classes[QuadrupedVisionEnvConfig] = QuadrupedVisionEnvironment
