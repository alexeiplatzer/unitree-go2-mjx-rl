from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax

from quadruped_mjx_rl.environments.physics_pipeline import (
    EnvModel,
    EnvSpec,
    PipelineModel,
)
from quadruped_mjx_rl.environments.quadruped.base import QuadrupedBaseEnv
from quadruped_mjx_rl.environments.quadruped.base import register_environment_config_class
from quadruped_mjx_rl.environments.quadruped.joystick_base import (
    JoystickBaseEnvConfig,
    QuadrupedJoystickBaseEnv,
)
from quadruped_mjx_rl.environments.quadruped.vision_base import (
    QuadrupedVisionBaseEnv,
    QuadrupedVisionBaseEnvConfig,
)
from quadruped_mjx_rl.environments.vision.robotic_vision import VisionConfig
from quadruped_mjx_rl.robots import RobotConfig


@dataclass
class QuadrupedJoystickVisionEnvConfig(QuadrupedVisionBaseEnvConfig, JoystickBaseEnvConfig):
    domain_rand: "QuadrupedJoystickVisionEnvConfig.DomainRandConfig" = field(
        default_factory=lambda: QuadrupedJoystickVisionEnvConfig.DomainRandConfig(
            apply_kicks=False
        )
    )

    @classmethod
    def config_class_key(cls) -> str:
        return "JoystickVision"

    @classmethod
    def get_environment_class(cls) -> type[QuadrupedBaseEnv]:
        return QuadrupedJoystickVisionEnv


register_environment_config_class(QuadrupedJoystickVisionEnvConfig)


class QuadrupedJoystickVisionEnv(QuadrupedVisionBaseEnv, QuadrupedJoystickBaseEnv):

    def __init__(
        self,
        environment_config: QuadrupedJoystickVisionEnvConfig,
        robot_config: RobotConfig,
        env_model: EnvSpec | EnvModel,
        *,
        vision_config: VisionConfig | None = None,
        init_qpos: jax.Array | None = None,
        renderer_maker: Callable[[PipelineModel], Any] | None = None,
    ):
        super().__init__(
            environment_config,
            robot_config,
            env_model,
            vision_config=vision_config,
            renderer_maker=renderer_maker,
        )
        self._init_q = self._init_q if init_qpos is None else init_qpos
