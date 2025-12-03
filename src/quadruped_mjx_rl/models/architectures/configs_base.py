from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic, Any

import jax
from flax import linen

from quadruped_mjx_rl.config_utils import Configuration, register_config_base_class
from quadruped_mjx_rl.models.base_modules import ActivationFn, MLP
from quadruped_mjx_rl.models.networks_utils import AgentNetworkParams, PolicyFactory
from quadruped_mjx_rl.types import Observation, PRNGKey


@dataclass
class ModuleConfigMLP:
    layer_sizes: list[int] = field(default_factory=lambda: [256, 256, 256, 256])

    def create(
        self,
        activation_fn: ActivationFn = linen.swish,
        activate_final: bool = False,
        extra_final_layer_size: int | None = None,
    ):
        layer_sizes = (
            self.layer_sizes + [extra_final_layer_size]
            if extra_final_layer_size
            else self.layer_sizes
        )
        return MLP(
            layer_sizes=layer_sizes,
            activation=activation_fn,
            activate_final=activate_final,
        )


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
    def get_model_class(cls) -> type["AgentModel"]:
        return AgentModel


register_config_base_class(ModelConfig)

_model_config_classes = {}

register_model_config_class = ModelConfig.make_register_config_class()

register_model_config_class(ModelConfig)


class ComponentNetworksArchitecture(ABC, Generic[AgentNetworkParams]):
    @abstractmethod
    def initialize(self, rng: PRNGKey) -> AgentNetworkParams:
        pass


class AgentModel(ABC, Generic[AgentNetworkParams]):
    @staticmethod
    @abstractmethod
    def agent_params_class() -> type[AgentNetworkParams]:
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
