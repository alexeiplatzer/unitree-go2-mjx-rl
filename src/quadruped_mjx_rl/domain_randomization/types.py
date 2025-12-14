from dataclasses import dataclass
from abc import ABC, abstractmethod

import jax

from quadruped_mjx_rl.physics_pipeline import PipelineModel, EnvModel
from quadruped_mjx_rl.types import PRNGKey


@dataclass
class DomainRandomizationConfig(ABC):

    @abstractmethod
    def domain_randomize(
        self,
        pipeline_model: PipelineModel,
        env_model: EnvModel,
        key: PRNGKey,
        num_worlds: int,
    ) -> tuple[PipelineModel, PipelineModel]:
        pass


@dataclass
class TerrainMapRandomizationConfig(ABC):

    @abstractmethod
    def domain_randomize(
        self,
        pipeline_model: PipelineModel,
        env_model: EnvModel,
        key: PRNGKey,
        num_worlds: int,
    ) -> tuple[PipelineModel, PipelineModel, tuple[jax.Array, jax.Array, jax.Array]]:
        pass
