import functools
from collections.abc import Callable
from dataclasses import dataclass, field

import jax
from flax import linen
from flax.struct import dataclass as flax_dataclass
from jax import numpy as jnp

from quadruped_mjx_rl.models import distributions, networks_utils
from quadruped_mjx_rl.models.architectures.configs_base import (
    ComponentNetworksArchitecture,
    ModelConfig,
    register_model_config_class,
)
from quadruped_mjx_rl.models.base_modules import ActivationFn
from quadruped_mjx_rl.models.base_modules import ModuleConfigMLP
from quadruped_mjx_rl.models.types import (
    Policy,
    AgentParams,
    identity_observation_preprocessor,
    Params,
    PolicyFactory,
    PreprocessObservationFn,
    PreprocessorParams,
    AgentNetworkParams,
)
from quadruped_mjx_rl.types import Observation, ObservationSize, PRNGKey, Action, Extra


@dataclass
class ActorCriticConfig(ModelConfig):
    policy_obs_key: str = "proprioceptive"
    policy: ModuleConfigMLP = field(
        default_factory=lambda: ModuleConfigMLP(layer_sizes=[128, 128, 128, 128, 128])
    )
    value_obs_key: str = "proprioceptive"
    value: ModuleConfigMLP = field(
        default_factory=lambda: ModuleConfigMLP(layer_sizes=[256, 256, 256, 256, 256])
    )

    @classmethod
    def config_class_key(cls) -> str:
        return "ActorCritic"

    @classmethod
    def get_model_class(cls) -> type["ActorCriticNetworks"]:
        return ActorCriticNetworks


register_model_config_class(ActorCriticConfig)


@flax_dataclass
class ActorCriticNetworkParams:
    policy: Params
    value: Params


@flax_dataclass
class ActorCriticAgentParams(AgentParams[ActorCriticNetworkParams]):
    """Full usable parameters for an actor critic architecture."""

    def restore_params(
        self,
        restore_params: "ActorCriticAgentParams",
        restore_value: bool = False,
    ):
        value_params = (
            restore_params.network_params.value if restore_value else self.network_params.value
        )
        return self.replace(
            network_params=ActorCriticNetworkParams(
                policy=restore_params.network_params.policy,
                value=value_params,
            ),
            preprocessor_params=restore_params.preprocessor_params,
        )


class ActorCriticNetworks(ComponentNetworksArchitecture[ActorCriticNetworkParams]):
    """A very basic Actor-Critic architecture consiting of two separate networks:
    the actor (policy) and the critic (value)."""

    def __init__(
        self,
        *,
        model_config: ActorCriticConfig,
        observation_size: ObservationSize,
        action_size: int,
        preprocess_observations_fn: PreprocessObservationFn = identity_observation_preprocessor,
        activation: ActivationFn = linen.swish,
    ):
        """Make Actor Critic networks with preprocessor."""
        self.dummy_obs = networks_utils.make_dummy_obs(obs_size=observation_size)
        self.preprocess_obs = functools.partial(
            networks_utils.preprocess_obs_by_key, preprocess_obs_fn=preprocess_observations_fn
        )
        self.parametric_action_distribution = distributions.NormalTanhDistribution(
            event_size=action_size
        )
        self.policy_obs_key = model_config.policy_obs_key
        self.policy_module = model_config.policy.create(
            activation_fn=activation,
            activate_final=False,
            extra_final_layer_size=self.parametric_action_distribution.param_size,
        )
        self.value_obs_key = model_config.value_obs_key
        self.value_module = model_config.value.create(
            activation_fn=activation, activate_final=False, extra_final_layer_size=1
        )

    @staticmethod
    def agent_params_class() -> type[ActorCriticAgentParams]:
        return ActorCriticAgentParams

    def initialize(self, rng: PRNGKey) -> ActorCriticNetworkParams:
        policy_key, value_key = jax.random.split(rng)
        return ActorCriticNetworkParams(
            policy=self.policy_module.init(policy_key, self.dummy_obs[self.policy_obs_key]),
            value=self.value_module.init(value_key, self.dummy_obs[self.value_obs_key]),
        )

    def apply_policy(
        self,
        preprocessor_params: PreprocessorParams,
        network_params: ActorCriticNetworkParams,
        observation: Observation,
    ) -> jax.Array:
        observation = self.preprocess_obs(preprocessor_params, observation)
        return self.policy_module.apply(network_params.policy, observation[self.policy_obs_key])

    def apply_value(
        self,
        preprocessor_params: PreprocessorParams,
        network_params: ActorCriticNetworkParams,
        observation: Observation,
        terminal: bool = False,
    ) -> jax.Array:
        observation = self.preprocess_obs(preprocessor_params, observation)
        return jnp.squeeze(
            self.value_module.apply(network_params.value, observation[self.value_obs_key]),
            axis=-1,
        )

    def policy_metafactory(
        self,
        policy_apply_fn: (
            Callable[[PreprocessorParams, AgentNetworkParams, Observation, ...], jax.Array]
            | Callable[[PreprocessorParams, AgentNetworkParams, Observation], jax.Array]
        ),
    ) -> PolicyFactory[AgentNetworkParams]:
        def make_policy(
            params: AgentParams[ActorCriticNetworkParams], deterministic: bool = False
        ) -> Policy:
            def policy(
                sample_key: PRNGKey, observation: Observation, *args, **kwargs
            ) -> tuple[Action, Extra]:
                policy_logits = policy_apply_fn(
                    params.preprocessor_params,
                    params.network_params,
                    observation,
                    *args,
                    **kwargs,
                )
                return networks_utils.process_policy_logits(
                    parametric_action_distribution=self.parametric_action_distribution,
                    logits=policy_logits,
                    sample_key=sample_key,
                    deterministic=deterministic,
                )

            return policy

        return make_policy

    def get_acting_policy_factory(self) -> PolicyFactory[AgentNetworkParams]:
        return self.policy_metafactory(self.apply_policy)
