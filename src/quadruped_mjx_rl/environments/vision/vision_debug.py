import jax
import jax.numpy as jnp
import numpy as np
from etils.epath import PathLike
from typing import Any

from quadruped_mjx_rl.environments.base import PipelineEnv
from quadruped_mjx_rl.physics_pipeline import load_to_model, PipelineState, State
from quadruped_mjx_rl.environments.vision.robotic_vision import RendererConfig
from quadruped_mjx_rl.types import Observation, Action, PRNGKey


class VisionDebugEnv(PipelineEnv):
    def __init__(
        self,
        init_scene_path: PathLike,
        vision_config: RendererConfig,
    ):
        env_model = load_to_model(init_scene_path)
        super().__init__(env_model)

        self._init_q = self.pipeline_model.model.qpos0
        self._nv = self.pipeline_model.model.nv

        # Setup vision with the madrona mjx engine
        from madrona_mjx.renderer import BatchRenderer

        self.renderer = BatchRenderer(
            m=self.pipeline_model.model,
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

    def reset(self, rng: PRNGKey) -> State:
        pipeline_state = self.pipeline_init(
            self._init_q
            + jax.random.uniform(rng, shape=self._init_q.shape, minval=0.0, maxval=0.0),
            jnp.zeros(self._nv),
        )

        state_info = {"rng": rng}

        obs = self._init_obs(pipeline_state, state_info)

        reward, done = jnp.zeros(2)
        metrics = {}
        state = State(pipeline_state, obs, reward, done, metrics, state_info)
        return state

    def step(self, state: State, action: Action) -> State:
        pipeline_state = self.pipeline_step(state.pipeline_state, jnp.zeros(self.action_size))

        # observation data
        obs = self._get_obs(pipeline_state, state.info, state.obs)

        state = state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=state.reward, done=state.done
        )
        return state

    def _init_obs(
        self,
        pipeline_state: PipelineState,
        state_info: dict[str, Any],
    ) -> Observation:
        render_token, rgb, depth = self.renderer.init(
            pipeline_state.data, self.pipeline_model.model
        )
        state_info["render_token"] = render_token

        # rgb_norm = jnp.asarray(rgb[1][..., :3], dtype=jnp.float32) / 255.0
        #
        # obs |= {"pixels/view_frontal_ego": rgb_norm, "pixels/view_terrain": depth[2]}
        obs = {"pixels/rgb_tensor": rgb, "pixels/depth_tensor": depth}
        return obs

    def _get_obs(
        self,
        pipeline_state: PipelineState,
        state_info: dict[str, Any],
        previous_obs: Observation,
    ) -> Observation:
        _, rgb, depth = self.renderer.render(state_info["render_token"], pipeline_state.data)
        # rgb_norm = jnp.asarray(rgb[1][..., :3], dtype=jnp.float32) / 255.0
        # obs |= {"pixels/view_frontal_ego": rgb_norm, "pixels/view_terrain": depth[2]}
        obs = {"pixels/rgb_tensor": rgb, "pixels/depth_tensor": depth}
        return obs
