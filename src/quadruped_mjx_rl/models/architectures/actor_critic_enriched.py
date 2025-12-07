"""Actor-Critic architecture with an additional common encoder."""

from dataclasses import dataclass, field

import jax
from flax import linen
from flax.struct import dataclass as flax_dataclass
from jax import numpy as jnp

from quadruped_mjx_rl.models import ActorCriticConfig
from quadruped_mjx_rl.models.architectures import ActorCriticNetworkParams
from quadruped_mjx_rl.models.architectures.actor_critic_base import ActorCriticNetworks
from quadruped_mjx_rl.models.architectures.configs_base import (
    ComponentNetworksArchitecture,
    register_model_config_class,
)
from quadruped_mjx_rl.models.base_modules import ActivationFn
from quadruped_mjx_rl.models.base_modules import ModuleConfigCNN, ModuleConfigMLP
from quadruped_mjx_rl.models.types import (
    AgentParams,
    identity_observation_preprocessor,
    Params,
    PreprocessObservationFn,
    PreprocessorParams,
)
from quadruped_mjx_rl.types import Observation, ObservationSize, PRNGKey


@dataclass
class ActorCriticEnrichedConfig(ActorCriticConfig):
    encoder_obs_key: str = "pixels/frontal_ego/rgb_adjusted"
    encoder: ModuleConfigCNN = field(
        default_factory=lambda: ModuleConfigCNN(
            filter_sizes=[32, 64, 64], dense=ModuleConfigMLP(layer_sizes=[256, 256])
        )
    )
    latent_encoding_size: int = 16

    @classmethod
    def config_class_key(cls) -> str:
        return "ActorCriticEnriched"

    @classmethod
    def get_model_class(cls) -> type["ActorCriticNetworks"]:
        return ActorCriticNetworks


register_model_config_class(ActorCriticEnrichedConfig)


@flax_dataclass
class ActorCriticEnrichedNetworkParams(ActorCriticNetworkParams):
    acting_encoder: Params


@flax_dataclass
class ActorCriticEnrichedAgentParams(AgentParams[ActorCriticEnrichedNetworkParams]):
    """Full usable parameters for an actor critic architecture."""

    def restore_params(
        self,
        restore_params: "ActorCriticEnrichedAgentParams",
        restore_value: bool = False,
    ):
        value_params = (
            restore_params.network_params.value if restore_value else self.network_params.value
        )
        return self.replace(
            network_params=ActorCriticEnrichedNetworkParams(
                policy=restore_params.network_params.policy,
                value=value_params,
                acting_encoder=restore_params.network_params.acting_encoder,
            ),
            preprocessor_params=restore_params.preprocessor_params,
        )


class ActorCriticEnrichedNetworks(
    ActorCriticNetworks,
    ComponentNetworksArchitecture[ActorCriticEnrichedNetworkParams],
):
    """An actor-critic architecture with an additional common encoder."""

    def __init__(
        self,
        *,
        model_config: ActorCriticEnrichedConfig,
        observation_size: ObservationSize,
        action_size: int,
        preprocess_observations_fn: PreprocessObservationFn = identity_observation_preprocessor,
        activation: ActivationFn = linen.swish,
    ):
        """Make Actor Critic networks with preprocessor."""
        self.acting_encoder_obs_key = model_config.encoder_obs_key
        self.acting_encoder_module = model_config.encoder.create(
            activation_fn=activation,
            activate_final=True,
            extra_final_layer_size=model_config.latent_encoding_size,
        )
        self.dummy_latent = jax.zeros((1, model_config.latent_encoding_size))
        super().__init__(
            model_config=model_config,
            observation_size=observation_size,
            action_size=action_size,
            preprocess_observations_fn=preprocess_observations_fn,
            activation=activation,
        )

    @staticmethod
    def agent_params_class() -> type[ActorCriticEnrichedAgentParams]:
        return ActorCriticEnrichedAgentParams

    def initialize(self, rng: PRNGKey) -> ActorCriticEnrichedNetworkParams:
        policy_key, value_key, encoder_key = jax.random.split(rng, 3)
        return ActorCriticEnrichedNetworkParams(
            policy=self.policy_module.init(
                policy_key,
                jnp.concatenate(
                    (self.dummy_obs[self.policy_obs_key], self.dummy_latent), axis=-1
                ),
            ),
            value=self.value_module.init(
                value_key,
                jnp.concatenate(
                    (self.dummy_obs[self.value_obs_key], self.dummy_latent), axis=-1
                ),
            ),
            acting_encoder=self.acting_encoder_module.init(
                encoder_key, self.dummy_obs[self.acting_encoder_obs_key]
            ),
        )

    def apply_acting_encoder(
        self,
        preprocessor_params: PreprocessorParams,
        network_params: ActorCriticEnrichedNetworkParams,
        observation: Observation,
    ) -> jax.Array:
        observation = self.preprocess_obs(preprocessor_params, observation)
        return self.acting_encoder_module.apply(
            network_params.acting_encoder, observation[self.acting_encoder_obs_key]
        )

    def apply_policy_with_latents(
        self,
        preprocessor_params: PreprocessorParams,
        network_params: ActorCriticEnrichedNetworkParams,
        observation: Observation,
        latent_encoding: jax.Array,
    ) -> jax.Array:
        observation = self.preprocess_obs(preprocessor_params, observation)
        input_vector = jnp.concatenate(
            (observation[self.policy_obs_key], latent_encoding), axis=-1
        )
        return self.policy_module.apply(
            network_params.policy, input_vector
        )

    def apply_value_with_latents(
        self,
        preprocessor_params: PreprocessorParams,
        network_params: ActorCriticEnrichedNetworkParams,
        observation: Observation,
        latent_encoding: jax.Array,
    ) -> jax.Array:
        observation = self.preprocess_obs(preprocessor_params, observation)
        input_vector = jnp.concatenate(
            (observation[self.policy_obs_key], latent_encoding), axis=-1
        )
        return jnp.squeeze(
            self.value_module.apply(network_params.value, input_vector),
            axis=-1,
        )

    def apply_policy(
        self,
        preprocessor_params: PreprocessorParams,
        network_params: ActorCriticEnrichedNetworkParams,
        observation: Observation,
    ) -> jax.Array:
        latent_encoding = self.apply_acting_encoder(
            preprocessor_params, network_params, observation
        )
        return self.apply_policy_with_latents(
            preprocessor_params, network_params, observation, latent_encoding
        )

    def apply_value(
        self,
        preprocessor_params: PreprocessorParams,
        network_params: ActorCriticEnrichedNetworkParams,
        observation: Observation,
    ) -> jax.Array:
        latent_encoding = self.apply_acting_encoder(
            preprocessor_params, network_params, observation
        )
        return self.apply_value_with_latents(
            preprocessor_params, network_params, observation, latent_encoding
        )
