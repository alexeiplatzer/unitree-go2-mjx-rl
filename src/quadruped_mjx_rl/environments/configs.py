from typing import Protocol, Generic, TypeVar

from dataclasses import dataclass

from brax.envs import PipelineEnv

EnvType = TypeVar("EnvType", bound=PipelineEnv)


@dataclass
class EnvironmentConfig(Generic[EnvType]):
    name: str
