from collections.abc import Callable
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import mujoco

from quadruped_mjx_rl import math
from quadruped_mjx_rl.environments import QuadrupedBaseEnv
from quadruped_mjx_rl.environments.physics_pipeline import (
    EnvModel,
    EnvSpec,
    Motion,
    Transform,
    PipelineModel,
    PipelineState,
    State,
)
from quadruped_mjx_rl.environments.quadruped.base import (
    register_environment_config_class,
    EnvironmentConfig,
    QuadrupedBaseEnv,
)
from quadruped_mjx_rl.environments.quadruped.target_reaching import (
    QuadrupedVisionTargetEnvConfig,
    QuadrupedVisionTargetEnvironment,
)
from quadruped_mjx_rl.robotic_vision import VisionConfig
from quadruped_mjx_rl.robots import RobotConfig


@dataclass
class QuadrupedColorGuidedEnvConfig(QuadrupedVisionTargetEnvConfig):

    @classmethod
    def config_class_key(cls) -> str:
        return "QuadrupedColorGuided"

    @classmethod
    def get_environment_class(cls) -> type[QuadrupedBaseEnv]:
        return QuadrupedColorGuidedEnvironment


register_environment_config_class(QuadrupedColorGuidedEnvConfig)


class QuadrupedColorGuidedEnvironment(QuadrupedVisionTargetEnvironment):
    """An expansion of the target-reaching environment to include obstacles, collision with
    which is punished."""

    def __init__(
        self,
        environment_config: QuadrupedColorGuidedEnvConfig,
        robot_config: RobotConfig,
        env_model: EnvSpec | EnvModel,
        vision_config: VisionConfig | None = None,
        init_qpos: jax.Array | None = None,
        renderer_maker: Callable[[PipelineModel], ...] | None = None,
    ):
        super().__init__(
            environment_config=environment_config,
            robot_config=robot_config,
            env_model=env_model,
            vision_config=vision_config,
            init_qpos=init_qpos,
            renderer_maker=renderer_maker,
        )

    def reset(self, rng: jax.Array) -> State:
        state = super().reset(rng)

        return state

    def _set_init_qpos(self, rng: jax.Array) -> jax.Array:
        init_q = self._init_q
        return init_q
