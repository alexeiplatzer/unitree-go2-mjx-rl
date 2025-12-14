from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from quadruped_mjx_rl.physics_pipeline import (
    Env,
    EnvModel,
    EnvSpec,
    PipelineModel,
)
from quadruped_mjx_rl.environments.quadruped.base import register_environment_config_class
from quadruped_mjx_rl.environments.quadruped.joystick_base import (
    JoystickBaseEnvConfig,
    QuadrupedJoystickBaseEnv,
)
from quadruped_mjx_rl.environments.vision import VisionWrapper
from quadruped_mjx_rl.robots import RobotConfig


@dataclass
class QuadrupedJoystickVisionEnvConfig(JoystickBaseEnvConfig):
    domain_rand: "QuadrupedJoystickVisionEnvConfig.DomainRandConfig" = field(
        default_factory=lambda: QuadrupedJoystickVisionEnvConfig.DomainRandConfig(
            apply_kicks=False
        )
    )

    @classmethod
    def config_class_key(cls) -> str:
        return "JoystickVision"

    @classmethod
    def get_environment_class(cls) -> type[Env]:
        return QuadrupedJoystickVisionEnv


register_environment_config_class(QuadrupedJoystickVisionEnvConfig)


class QuadrupedJoystickVisionEnv(VisionWrapper):

    def __init__(
        self,
        environment_config: QuadrupedJoystickVisionEnvConfig,
        robot_config: RobotConfig,
        env_model: EnvSpec | EnvModel,
        renderer_maker: Callable[[PipelineModel], Any] | None = None,
    ):
        env = QuadrupedJoystickBaseEnv(environment_config, robot_config, env_model)
        super().__init__(
            env=env,
            vision_env_config=environment_config.vision_env_config,
            renderer_maker=renderer_maker,
        )
