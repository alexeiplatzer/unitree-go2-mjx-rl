# Math
import jax
import jax.numpy as jnp

# Sim
import numpy as np

# Brax
from brax.base import State as PipelineState
from brax.base import System
from brax.envs.base import State
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
        vision_config: VisionConfig,
    ):
        super().__init__(environment_config, robot_config, init_scene_path)

        self.reward_scales = {}  # remove all the rewards form the joystick base config

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
        pipeline_state = self.pipeline_init(
            self._init_q + jax.random.uniform(
                rng, shape=self._init_q.shape, minval=0.0, maxval=0.0
            ),
            jnp.zeros(self._nv),
        )

        state_info = {"rng": rng}

        obs = self._init_obs(pipeline_state, state_info)

        reward, done = jnp.zeros(2)
        metrics = {}
        state = State(pipeline_state, obs, reward, done, metrics, state_info)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        pipeline_state = self._physics_step(state, action)

        # observation data
        obs = self._get_obs(pipeline_state, state.info, state.obs)

        state = state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=state.reward, done=state.done
        )
        return state

    def _init_obs(
        self,
        pipeline_state: PipelineState,
        state_info: dict[str, ...],
    ) -> jax.Array | dict[str, jax.Array]:
        render_token, rgb, depth = self.renderer.init(pipeline_state, self.sys)
        state_info["render_token"] = render_token

        # rgb_norm = jnp.asarray(rgb[1][..., :3], dtype=jnp.float32) / 255.0
        #
        # obs |= {"pixels/view_frontal_ego": rgb_norm, "pixels/view_terrain": depth[2]}
        obs = {"pixels/rgb_tensor": rgb, "pixels/depth_tensor": depth}
        return obs

    def _get_obs(
        self,
        pipeline_state: PipelineState,
        state_info: dict[str, ...],
        previous_obs: jax.Array | dict[str, jax.Array],
    ) -> jax.Array | dict[str, jax.Array]:
        _, rgb, depth = self.renderer.render(state_info["render_token"], pipeline_state)
        # rgb_norm = jnp.asarray(rgb[1][..., :3], dtype=jnp.float32) / 255.0
        # obs |= {"pixels/view_frontal_ego": rgb_norm, "pixels/view_terrain": depth[2]}
        obs = {"pixels/rgb_tensor": rgb, "pixels/depth_tensor": depth}
        return obs
