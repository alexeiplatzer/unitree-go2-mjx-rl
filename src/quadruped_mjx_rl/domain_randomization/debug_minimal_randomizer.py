from dataclasses import dataclass

import jax

from quadruped_mjx_rl.types import PRNGKey
from quadruped_mjx_rl.physics_pipeline import (
    EnvModel,
    PipelineModel,
)
from quadruped_mjx_rl.domain_randomization.types import DomainRandomizationConfig


@dataclass
class DebugMinimalRandomizationConfig(DomainRandomizationConfig):
    def domain_randomize(
        self,
        pipeline_model: PipelineModel,
        env_model: EnvModel,
        key: PRNGKey,
        num_worlds: int,
    ) -> tuple[PipelineModel, PipelineModel]:
        in_axes = jax.tree.map(lambda x: None, pipeline_model)
        return pipeline_model, in_axes
