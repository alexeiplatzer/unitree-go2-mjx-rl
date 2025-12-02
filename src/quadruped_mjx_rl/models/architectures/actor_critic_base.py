from dataclasses import dataclass, field

import jax
from flax import linen
from flax.struct import dataclass as flax_dataclass

from quadruped_mjx_rl import types
from quadruped_mjx_rl.models import base_modules, distributions, networks_utils
from quadruped_mjx_rl.models.architectures.configs_base import (
    AgentModel, ComponentNetworksArchitecture, ModelConfig,
    register_model_config_class, ModuleConfigMLP
)
from quadruped_mjx_rl.types import Observation, PRNGKey


@dataclass
class ActorCriticConfig(ModelConfig):
    policy_obs_key: str = "proprioceptive"
    policy: ModuleConfigMLP = field(
        default_factory=lambda: ModuleConfigMLP(layer_sizes=[128, 128, 128, 128])
    )
    value_obs_key: str = "proprioceptive"
    value: ModuleConfigMLP = field(
        default_factory=lambda: ModuleConfigMLP(layer_sizes=[256, 256, 256, 256])
    )

    @classmethod
    def config_class_key(cls) -> str:
        return "ActorCritic"

    @classmethod
    def get_model_class(cls) -> type["ActorCriticAgent"]:
        return ActorCriticAgent


register_model_config_class(ActorCriticConfig)


@flax_dataclass
class ActorCriticNetworkParams:
    policy: types.Params
    value: types.Params


@flax_dataclass
class ActorCriticAgentParams(networks_utils.AgentParams[ActorCriticNetworkParams]):
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
        observation_size: types.ObservationSize,
        output_size: int,
        preprocess_observations_fn: types.PreprocessObservationFn = (
            types.identity_observation_preprocessor
        ),
        activation: base_modules.ActivationFn = linen.swish,
    ):
        policy_module = base_modules.MLP(
            layer_sizes=(
                model_config.policy.layer_sizes + [output_size]),
            activation=activation,
            activate_final=False,
        )
        self._policy_network = networks_utils.make_network(
            module=policy_module,
            obs_size=observation_size,
            preprocess_observations_fn=preprocess_observations_fn,
            preprocess_obs_keys=(model_config.policy_obs_key,),
            apply_to_obs_keys=(model_config.policy_obs_key,),
            squeeze_output=False,
        )
        value_module = base_modules.MLP(
            layer_sizes=model_config.value.layer_sizes + [1],
            activation=activation,
            activate_final=False,
        )
        self._value_network = networks_utils.make_network(
            module=value_module,
            obs_size=observation_size,
            preprocess_observations_fn=preprocess_observations_fn,
            preprocess_obs_keys=(model_config.value_obs_key,),
            apply_to_obs_keys=(model_config.value_obs_key,),
            squeeze_output=True,
        )

    def initialize(self, rng: PRNGKey) -> ActorCriticNetworkParams:
        policy_key, value_key = jax.random.split(rng)
        return ActorCriticNetworkParams(
            policy=self._policy_network.init(policy_key),
            value=self._value_network.init(value_key),
        )

    def apply_policy_module(
        self, params: ActorCriticAgentParams, observation: types.Observation
    ) -> jax.Array:
        return self._policy_network.apply(
            params.preprocessor_params, params.network_params.policy, observation
        )

    def apply_value_module(
        self, params: ActorCriticAgentParams, observation: types.Observation
    ) -> jax.Array:
        return self._value_network.apply(
            params.preprocessor_params, params.network_params.value, observation
        )


class ActorCriticAgent(AgentModel[ActorCriticNetworkParams]):

    def __init__(
        self,
        *,
        model_config: ActorCriticConfig,
        observation_size: types.ObservationSize,
        action_size: int,
        preprocess_observations_fn: types.PreprocessObservationFn = (
            types.identity_observation_preprocessor
        ),
        activation: base_modules.ActivationFn = linen.swish,
    ):
        """Make Actor Critic networks with preprocessor."""
        parametric_action_distribution = distributions.NormalTanhDistribution(
            event_size=action_size
        )
        self.networks = self.networks_class()(
            model_config=model_config,
            observation_size=observation_size,
            output_size=parametric_action_distribution.param_size,
            preprocess_observations_fn=preprocess_observations_fn,
            activation=activation,
        )
        self.parametric_action_distribution = parametric_action_distribution

    @staticmethod
    def agent_params_class() -> type[ActorCriticAgentParams]:
        return ActorCriticAgentParams

    @staticmethod
    def networks_class() -> type[ActorCriticNetworks]:
        return ActorCriticNetworks

    def apply_rollout_policy(
        self,
        params: ActorCriticAgentParams,
        observation: Observation,
    ) -> jax.Array:
        return self.networks.apply_policy_module(params, observation)

    def apply_rollout_value(
        self, params: ActorCriticAgentParams, observation: Observation
    ) -> jax.Array:
        return self.networks.apply_value_module(params, observation)

    def policy_metafactory(
        self
    ) -> tuple[networks_utils.PolicyFactory[ActorCriticNetworkParams], ...]:
        """Creates params and inference function for the PPO agent."""

        def make_policy(
            params: ActorCriticAgentParams, deterministic: bool = False
        ) -> types.Policy:
            return networks_utils.policy_factory(
                policy_apply=self.apply_rollout_policy,
                parametric_action_distribution=self.parametric_action_distribution,
                params=params,
                deterministic=deterministic,
            )

        return (make_policy,)
