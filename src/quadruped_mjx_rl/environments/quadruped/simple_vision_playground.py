
# Typing
from dataclasses import dataclass, field
from typing import Optional, Any

from brax import base
# Supporting
from etils.epath import PathLike

# Math
import jax
import jax.numpy as jnp
import numpy as np

# Sim
import mujoco

from quadruped_mjx_rl.environments import QuadrupedBaseEnv
from quadruped_mjx_rl.environments.physics_pipeline import PipelineState, EnvModel, State
from quadruped_mjx_rl.environments.quadruped.base import register_environment_config_class

# Definitions
from quadruped_mjx_rl.robotic_vision import VisionConfig
from quadruped_mjx_rl.robots import RobotConfig
from quadruped_mjx_rl.environments.quadruped.joystick_base import (
    JoystickBaseEnvConfig,
    QuadrupedJoystickBaseEnv,
)


def adjust_brightness(img, scale):
    """Adjusts the brightness of an image by scaling the pixel values."""
    return jnp.clip(img * scale, 0, 1)


@dataclass
class QuadrupedVisionEnvConfig(JoystickBaseEnvConfig):
    use_vision: bool = True

    @dataclass
    class ObservationNoiseConfig(JoystickBaseEnvConfig.ObservationNoiseConfig):
        brightness: list[float] = field(default_factory=lambda: [1.0, 1.0])

    observation_noise: ObservationNoiseConfig = field(default_factory=ObservationNoiseConfig)

    @classmethod
    def environment_class_key(cls) -> str:
        return "QuadrupedVision"

    @classmethod
    def get_environment_class(cls) -> type[QuadrupedBaseEnv]:
        return QuadrupedVisionEnvironment


register_environment_config_class(QuadrupedVisionEnvConfig)


class QuadrupedVisionEnvironment(QuadrupedJoystickBaseEnv):

    def __init__(
        self,
        environment_config: QuadrupedVisionEnvConfig,
        robot_config: RobotConfig,
        env_model: EnvModel,
        vision_config: VisionConfig | None = None,
        # renderer=None,
    ):
        super().__init__(environment_config, robot_config, env_model)

        self._use_vision = environment_config.use_vision
        if self._use_vision:
            if vision_config is None:
                raise ValueError("use_vision set to true, VisionConfig not provided.")
            # if renderer is None:
            #     raise ValueError("use_vision set to false, renderer not provided.")

            self._num_vision_envs = vision_config.render_batch_size
            self.renderer = None  # must be set!

    @staticmethod
    def customize_model(
        init_scene_path: PathLike, environment_config: QuadrupedVisionEnvConfig
    ):
        env_model = QuadrupedJoystickBaseEnv.customize_model(
            init_scene_path, environment_config
        )
        floor_id = mujoco.mj_name2id(env_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        env_model.geom_size[floor_id, :2] = [5.0, 5.0]
        return env_model

    def reset(self, rng: jax.Array) -> State:
        pipeline_state = self.pipeline_init(
            self._init_q + jax.random.uniform(
                rng, shape=self._init_q.shape, minval=0.0, maxval=0.0
            ),
            jnp.zeros(self._nv),
        )

        state_info = {
            "rng": rng,
            "step": 0,
            "rewards": {k: jnp.zeros(()) for k in self.reward_scales.keys()},
            "last_act": jnp.zeros(self.action_size),
        }

        obs = self._init_obs(pipeline_state, state_info)

        reward, done = jnp.zeros(2)

        metrics = {
            f"reward/{k}": jnp.zeros(()) for k in self.reward_scales.keys()
        }
        metrics["total_dist"] = jnp.zeros(())

        state = State(pipeline_state, obs, reward, done, metrics, state_info)
        return state

    def _init_obs(
        self, pipeline_state: PipelineState, state_info: dict[str, ...]
    ) -> dict[str, jax.Array]:
        obs = {
            "state": QuadrupedJoystickBaseEnv._init_obs(self, pipeline_state, state_info),
        }
        if self._use_vision:
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

            obs |= {"pixels/view_frontal_ego": rgb_adjusted, "pixels/view_terrain": depth[1]}

        return obs

    def _get_obs(
        self,
        pipeline_state: PipelineState,
        state_info: dict[str, ...],
        last_obs: jax.Array | dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        obs = {
            "state": self._get_state_obs(pipeline_state, state_info),
        }
        if self._use_vision:
            _, rgb, depth = self.renderer.render(state_info["render_token"], pipeline_state)
            rgb_norm = jnp.asarray(rgb[0][..., :3], dtype=jnp.float32) / 255.0
            rgb_adjusted = adjust_brightness(rgb_norm, state_info["brightness"])
            obs |= {"pixels/view_frontal_ego": rgb_adjusted, "pixels/view_terrain": depth[1]}
        return obs
