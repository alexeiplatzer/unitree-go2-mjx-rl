from dataclasses import dataclass, field


@dataclass
class VisionConfig:
    gpu_id: int = 0
    render_batch_size: int = 1024
    render_width: int = 128
    render_height: int = 64
    use_rasterizer: bool = False
    enabled_geom_groups: list[int] = field(default_factory=lambda: [0])
    enabled_cameras: list[int] = field(default_factory=lambda: [1, 2])
    vision_class: str = "default"


vision_config_classes = {
    "default": VisionConfig,
}
