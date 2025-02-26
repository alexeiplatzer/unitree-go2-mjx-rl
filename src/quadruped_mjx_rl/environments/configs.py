from typing import Protocol

from dataclasses import dataclass

from brax.envs import PipelineEnv


@dataclass
class EnvironmentConfig[Env: PipelineEnv](Protocol):
    name: str
