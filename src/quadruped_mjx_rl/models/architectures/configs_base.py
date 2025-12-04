from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic

import jax

from quadruped_mjx_rl.config_utils import Configuration, register_config_base_class
from quadruped_mjx_rl.environments import Env, State
from quadruped_mjx_rl.environments.vision.vision_wrappers import VisionWrapper
from quadruped_mjx_rl.models.base_modules import ModuleConfigMLP
from quadruped_mjx_rl.models.types import AgentNetworkParams, PolicyFactory, AgentParams
from quadruped_mjx_rl.types import Observation, PRNGKey, Transition


@dataclass
class ModelConfig(Configuration):
    policy: ModuleConfigMLP

    @classmethod
    def config_base_class_key(cls) -> str:
        return "model"

    @classmethod
    def config_class_key(cls) -> str:
        return "custom"

    @classmethod
    def _get_config_class_dict(cls) -> dict[str, type["Configuration"]]:
        return _model_config_classes

    @classmethod
    def get_model_class(cls) -> type["ComponentNetworksArchitecture"]:
        return ComponentNetworksArchitecture


register_config_base_class(ModelConfig)

_model_config_classes = {}

register_model_config_class = ModelConfig.make_register_config_class()

register_model_config_class(ModelConfig)


class ComponentNetworksArchitecture(ABC, Generic[AgentNetworkParams]):
    @abstractmethod
    def initialize(self, rng: PRNGKey) -> AgentNetworkParams:
        pass

    @staticmethod
    @abstractmethod
    def agent_params_class() -> type[AgentParams[AgentNetworkParams]]:
        pass

    @abstractmethod
    def get_acting_policy_factory(self) -> PolicyFactory[AgentNetworkParams]:
        pass

    @abstractmethod
    def generate_training_unroll(
        self,
        params: AgentParams,
        env: Env | VisionWrapper,
        env_state: State,
        key: PRNGKey,
        unroll_length: int,
        extra_fields: Sequence[str] = (),
    ) -> tuple[State, Transition]:
        pass
