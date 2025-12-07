from dataclasses import dataclass, field

from quadruped_mjx_rl.environments.quadruped.base import (
    QuadrupedBaseEnv, register_environment_config_class,
)
from quadruped_mjx_rl.environments.quadruped.target_reaching import (
    QuadrupedVisionTargetEnv, QuadrupedVisionTargetEnvConfig,
)
from quadruped_mjx_rl.environments.vision import ColorGuidedEnvConfig


@dataclass
class QuadrupedColorGuidedEnvConfig(QuadrupedVisionTargetEnvConfig):
    vision_env_config: ColorGuidedEnvConfig = field(default_factory=ColorGuidedEnvConfig)

    @classmethod
    def config_class_key(cls) -> str:
        return "QuadrupedColorGuided"

    @classmethod
    def get_environment_class(cls) -> type[QuadrupedBaseEnv]:
        return QuadrupedVisionTargetEnv


register_environment_config_class(QuadrupedColorGuidedEnvConfig)
