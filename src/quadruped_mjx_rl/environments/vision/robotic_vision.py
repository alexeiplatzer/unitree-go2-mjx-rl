import functools
from dataclasses import dataclass, field

import numpy as np
import jax.numpy as jnp

from quadruped_mjx_rl.physics_pipeline import PipelineModel


@dataclass
class RendererConfig:
    gpu_id: int = 0
    render_batch_size: int = 256
    render_width: int = 128
    render_height: int = 128
    use_rasterizer: bool = False
    enabled_geom_groups: list[int] = field(default_factory=lambda: [0, 1, 2])
    enabled_cameras: list[int] = field(default_factory=lambda: [1, 2])


def get_renderer_maker(vision_config: RendererConfig):
    return functools.partial(get_renderer, vision_config=vision_config)


def get_renderer(
    pipeline_model: PipelineModel, vision_config: RendererConfig, debug: bool = False
):
    if debug:
        return MockRenderer(vision_config)
    else:
        from madrona_mjx.renderer import BatchRenderer

        return BatchRenderer(
            m=pipeline_model.model,
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


class MockRenderer:
    def __init__(self, vision_config: RendererConfig):
        self.batch_size = vision_config.render_batch_size
        self.width = vision_config.render_width
        self.height = vision_config.render_height
        self.num_cams = len(vision_config.enabled_cameras)

    def init(self, data, model):

        return (
            jnp.zeros(()),
            jnp.zeros(
                (self.num_cams, self.height, self.width, 4),
                dtype=jnp.float32,
            ),
            jnp.zeros(
                (self.num_cams, self.height, self.width, 1),
                dtype=jnp.float32,
            ),
        )

    def render(self, token, data):

        return (
            jnp.zeros(()),
            jnp.zeros(
                (self.num_cams, self.height, self.width, 4),
                dtype=jnp.float32,
            ),
            jnp.zeros(
                (self.num_cams, self.height, self.width, 1),
                dtype=jnp.float32,
            ),
        )
