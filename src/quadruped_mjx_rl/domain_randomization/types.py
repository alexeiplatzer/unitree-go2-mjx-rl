from typing import Protocol

import jax

from quadruped_mjx_rl.environments import EnvModel
from quadruped_mjx_rl.environments.physics_pipeline import PipelineModel
from quadruped_mjx_rl.types import PRNGKey


class DomainRandomizationFn(Protocol):
    def __call__(
        self,
        pipeline_model: PipelineModel,
        env_model: EnvModel,
        rng_key: PRNGKey,
        num_worlds: int,
    ) -> tuple[PipelineModel, PipelineModel]:
        pass


class TerrainMapRandomizationFn(Protocol):
    def __call__(
        self,
        pipeline_model: PipelineModel,
        env_model: EnvModel,
        rng_key: PRNGKey,
        num_worlds: int,
        num_colors: int = 2,
    ) -> tuple[PipelineModel, PipelineModel, tuple[jax.Array, jax.Array, jax.Array]]:
        pass
