from dataclasses import dataclass, field
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
