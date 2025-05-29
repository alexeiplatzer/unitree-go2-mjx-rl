# Typing
from dataclasses import dataclass, asdict, field
from collections.abc import Sequence

# Math
import jax
import jax.numpy as jnp

# Sim
import mujoco
import numpy as np

# Brax
from brax import base
from brax import math
from brax.base import Motion, Transform
from brax.base import State as PipelineState
from brax.base import System
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils.epath import PathLike

from quadruped_mjx_rl.robotic_vision import VisionConfig
from quadruped_mjx_rl.robots import RobotConfig
from quadruped_mjx_rl.environments.base import QuadrupedBaseEnv
from quadruped_mjx_rl.environments.joystick_base import JoystickBaseEnvConfig


class VisionDebugEnv(QuadrupedBaseEnv):
    def __init__(
        self,
        environment_config: JoystickBaseEnvConfig,
        robot_config: RobotConfig,
        init_scene_path: PathLike,
        vision_config: VisionConfig | None = None,
    ):
        super().__init__(environment_config, robot_config, init_scene_path)

        if vision_config is not None:
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
            self._use_vision = True
        else:
            self._use_vision = False

    @staticmethod
    def make_system(
        init_scene_path: PathLike, environment_config: JoystickBaseEnvConfig
    ) -> System:
        sys = QuadrupedBaseEnv.make_system(init_scene_path, environment_config)
        # sys = sys.replace(
        #     dof_damping=sys.dof_damping.at[6:].set(environment_config.sim.override.Kd),
        # )
        return sys

    def reset(self, rng: jax.Array) -> State:
        state = super().reset(rng)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        state = super().step(state, action)
        return state

    def _init_obs(
        self,
        pipeline_state: PipelineState,
        state_info: dict[str, ...],
    ) -> jax.Array | dict[str, jax.Array]:
        obs = {
            "state": QuadrupedBaseEnv._get_state_obs(self, pipeline_state, state_info)
        }

        if self._use_vision:
            render_token, rgb, depth = self.renderer.init(pipeline_state, self.sys)
            state_info["render_token"] = render_token

            rgb_norm = jnp.asarray(rgb[1][..., :3], dtype=jnp.float32) / 255.0

            obs |= {"pixels/view_frontal_ego": rgb_norm, "pixels/view_terrain": depth[2]}

        return obs

    def _get_obs(
        self,
        pipeline_state: PipelineState,
        state_info: dict[str, ...],
        previous_obs: jax.Array | dict[str, jax.Array],
    ) -> jax.Array | dict[str, jax.Array]:
        obs = {
            "state": QuadrupedBaseEnv._get_state_obs(self, pipeline_state, state_info)
        }
        if self._use_vision:
            _, rgb, depth = self.renderer.render(state_info["render_token"], pipeline_state)
            rgb_norm = jnp.asarray(rgb[1][..., :3], dtype=jnp.float32) / 255.0
            obs |= {"pixels/view_frontal_ego": rgb_norm, "pixels/view_terrain": depth[2]}
        return obs
