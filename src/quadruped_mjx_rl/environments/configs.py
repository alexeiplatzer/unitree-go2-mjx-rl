from typing import Generic, TypeVar

from dataclasses import dataclass, field

from brax.envs import PipelineEnv

EnvType = TypeVar("EnvType", bound=PipelineEnv)


@dataclass
class EnvironmentConfig(Generic[EnvType]):
    name: str


@dataclass
class VisionConfig:
    gpu_id: int = 0
    render_batch_size: int = 1024
    render_width: int = 64
    render_height: int = 64
    use_rasterizer: bool = False
    enabled_geom_groups: list[int] = field(default_factory=lambda: [0, 1, 2])
