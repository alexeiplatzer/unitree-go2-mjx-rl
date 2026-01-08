from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic

from quadruped_mjx_rl.config_utils import Configuration, register_config_base_class
from quadruped_mjx_rl.models.acting import GenerateUnrollFn
from quadruped_mjx_rl.models.base_modules import ModuleConfigMLP
from quadruped_mjx_rl.models.types import AgentNetworkParams, AgentParams
from quadruped_mjx_rl.types import PRNGKey


@dataclass
class ModelConfig(Configuration):
    policy: ModuleConfigMLP

    @classmethod
    def default(cls) -> "ModelConfig":
        return ModelConfig(
            policy=ModuleConfigMLP(
                layer_sizes=[128, 128, 128, 128, 128],
                obs_key="proprioceptive"
            )
        )

    @property
    def vision(self) -> bool:
        return self.policy.vision

    @property
    def recurrent(self) -> bool:
        return False

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

    def make_acting_unroll_fn(
        self,
        agent_params: AgentParams[AgentNetworkParams],
        *,
        deterministic: bool = False,
        accumulate_pipeline_states: bool = False,
    ) -> GenerateUnrollFn:
        pass
