from dataclasses import dataclass, field
import numpy as np

from brax.base import System

from quadruped_mjx_rl.config_utils import Configuration, register_config_base_class


@dataclass
class VisionConfig(Configuration):
    gpu_id: int = 0
    render_batch_size: int = 512
    render_width: int = 128
    render_height: int = 64
    use_rasterizer: bool = False
    enabled_geom_groups: list[int] = field(default_factory=lambda: [0])
    enabled_cameras: list[int] = field(default_factory=lambda: [1, 2])

    @classmethod
    def config_base_class_key(cls) -> str:
        return "vision"


register_config_base_class(VisionConfig)


def get_renderer(sys: System, vision_config: VisionConfig):
    from madrona_mjx.renderer import BatchRenderer

    return BatchRenderer(
        m=sys,
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
