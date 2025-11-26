from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic

import jax

from quadruped_mjx_rl.config_utils import Configuration, register_config_base_class
from quadruped_mjx_rl.models.networks_utils import AgentNetworkParams, PolicyFactory
from quadruped_mjx_rl.types import Observation, PRNGKey


@dataclass
class ModelConfig(Configuration):
    modules: dict[str, list[int]]

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
    def get_model_class(cls) -> type["ComponentNetworkArchitecture"]:
        return ComponentNetworkArchitecture


register_config_base_class(ModelConfig)

_model_config_classes = {}

register_model_config_class = ModelConfig.make_register_config_class()

register_model_config_class(ModelConfig)


class ComponentNetworkArchitecture(ABC, Generic[AgentNetworkParams]):
    @abstractmethod
    def agent_params_class(self):
        pass

    @abstractmethod
    def initialize(self, rng: PRNGKey) -> AgentNetworkParams:
        pass

    @abstractmethod
    def apply_rollout_policy(
        self,
        params: AgentNetworkParams,
        observation: Observation,
    ) -> jax.Array:
        """Gets the logits for applying the network's policy with the provided params to the
        provided observations"""
        pass

    @abstractmethod
    def policy_metafactory(self) -> tuple[PolicyFactory[AgentNetworkParams], ...]:
        pass
