import functools
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic

import jax
from jax import numpy as jnp

from quadruped_mjx_rl.config_utils import Configuration, register_config_base_class
from quadruped_mjx_rl.physics_pipeline import Env, State
from quadruped_mjx_rl.models.acting import actor_step, GenerateUnrollFn, vision_actor_step
from quadruped_mjx_rl.models.base_modules import ModuleConfigMLP
from quadruped_mjx_rl.models.types import AgentNetworkParams, AgentParams, PolicyFactory
from quadruped_mjx_rl.types import PRNGKey, Transition


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

    def make_unroll_fn(
        self,
        agent_params: AgentParams[AgentNetworkParams],
        deterministic: bool = False,
        vision: bool = False,
        policy_factory: PolicyFactory | None = None,
        proprio_steps_per_vision_step: int = 1,
    ) -> GenerateUnrollFn:
        if policy_factory is None:
            policy_factory = self.get_acting_policy_factory()
        acting_policy = policy_factory(agent_params, deterministic)
        step_fn = (
            functools.partial(vision_actor_step, proprio_substeps=proprio_steps_per_vision_step)
            if vision
            else actor_step
        )

        def reshape(data):
            if vision:
                return jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data)
            else:
                return data

        def generate_unroll(
            env_state: State,
            key: PRNGKey,
            env: Env,
            unroll_length: int,
            extra_fields: Sequence[str] = (),
        ) -> tuple[State, Transition]:
            (env_state, _), transitions = jax.lax.scan(
                functools.partial(
                    step_fn, env=env, policy=acting_policy, extra_fields=extra_fields
                ),
                (env_state, key),
                (),
                length=unroll_length,
            )
            transitions = reshape(transitions)
            return env_state, transitions

        return generate_unroll
