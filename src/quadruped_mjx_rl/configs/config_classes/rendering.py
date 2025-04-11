from dataclasses import dataclass, field


@dataclass
class RenderConfig:
    episode_length: int = 1000
    render_every: int = 2
    seed: int = 0
    n_steps: int = 500
    command: dict[str, float] = field(default_factory=lambda: {
        "x_vel": 0.5,
        "y_vel": 0.0,
        "ang_vel": 0.0,
    })
    rendering_class: str = "default"


rendering_config_classes = {
    "default": RenderConfig,
}
