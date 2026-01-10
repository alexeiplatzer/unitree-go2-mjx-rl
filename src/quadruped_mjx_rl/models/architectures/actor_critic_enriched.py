"""Actor-Critic architecture with an additional common encoder."""

from dataclasses import dataclass

import jax
from flax import linen
from flax.struct import dataclass as flax_dataclass
from jax import numpy as jnp

from quadruped_mjx_rl.models.acting import GenerateUnrollFn
from quadruped_mjx_rl.models.architectures.actor_critic_base import (
    ActorCriticConfig,
    ActorCriticNetworkParams,
    ActorCriticNetworks,
)
from quadruped_mjx_rl.models.architectures.configs_base import (
    ComponentNetworksArchitecture,
    register_model_config_class,
)
from quadruped_mjx_rl.models.base_modules import ActivationFn, ModuleConfigMixedModeCNN
from quadruped_mjx_rl.models.base_modules import ModuleConfigCNN
from quadruped_mjx_rl.models.base_modules import ModuleConfigMLP
from quadruped_mjx_rl.models.types import AgentNetworkParams, AgentParams
from quadruped_mjx_rl.models.types import (
    identity_observation_preprocessor,
    Params,
    PreprocessObservationFn,
    PreprocessorParams,
)
from quadruped_mjx_rl.types import Observation, PRNGKey
from quadruped_mjx_rl.types import ObservationSize


@dataclass
class ActorCriticEnrichedConfig(ActorCriticConfig):
    encoder: ModuleConfigMLP | ModuleConfigCNN | ModuleConfigMixedModeCNN
    latent_encoding_size: int

    @classmethod
    def default(cls) -> "ActorCriticEnrichedConfig":
        default_super = ActorCriticConfig.default()
        return ActorCriticEnrichedConfig(
            policy=ModuleConfigMLP(
                layer_sizes=[128, 128, 128, 128], obs_key=default_super.policy.obs_key
            ),
            value=ModuleConfigMLP(
                layer_sizes=[256, 256, 256, 256], obs_key=default_super.value.obs_key
            ),
            encoder=ModuleConfigMLP(layer_sizes=[256], obs_key="goal_direction"),
            latent_encoding_size=256,
        )

    @classmethod
    def default_vision(cls) -> "ActorCriticEnrichedConfig":
        default_super = ActorCriticEnrichedConfig.default()
        return ActorCriticEnrichedConfig(
            policy=default_super.policy,
            value=default_super.value,
            encoder=ModuleConfigCNN(
                filter_sizes=[16, 24, 32],
                dense=ModuleConfigMLP(layer_sizes=[256]),
                obs_key="pixels/frontal_ego/rgb_adjusted"
            ),
            latent_encoding_size=256,
        )

    @classmethod
    def default_privileged_map(cls) -> "ActorCriticEnrichedConfig":
        default_super = ActorCriticEnrichedConfig.default_vision()
        return ActorCriticEnrichedConfig(
            policy=default_super.policy,
            value=default_super.value,
            encoder=ModuleConfigCNN(
                filter_sizes=[16, 24, 32],
                dense=ModuleConfigMLP(layer_sizes=[256]),
                obs_key="privileged_terrain_map"
            ),
            latent_encoding_size=256,
        )

    @classmethod
    def default_mixed(cls) -> "ActorCriticEnrichedConfig":
        default_super = ActorCriticEnrichedConfig.default_vision()
        return ActorCriticEnrichedConfig(
            policy=default_super.policy,
            value=default_super.value,
            encoder=ModuleConfigMixedModeCNN(
                vision_preprocessing=ModuleConfigCNN(
                    filter_sizes=[8, 16, 32],
                    dense=ModuleConfigMLP(layer_sizes=[256]),
                    obs_key="privileged_terrain_map",
                ),
                joint_processing=ModuleConfigMLP(layer_sizes=[256], obs_key="goal_direction"),
            ),
            latent_encoding_size=default_super.latent_encoding_size,
        )

    @property
    def vision(self) -> bool:
        return super().vision or self.encoder.vision

    @classmethod
    def config_class_key(cls) -> str:
        return "ActorCriticEnriched"

    @classmethod
    def get_model_class(cls) -> type["ActorCriticEnrichedNetworks"]:
        return ActorCriticEnrichedNetworks


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
        vision_obs_period: int | None = None,
        preprocess_observations_fn: PreprocessObservationFn = identity_observation_preprocessor,
        activation: ActivationFn = linen.swish,
    ):
        """Make Actor Critic networks with preprocessor."""
        self.acting_encoder_module = model_config.encoder.create(
            activation_fn=activation,
            activate_final=True,
            extra_final_layer_size=model_config.latent_encoding_size,
        )
        self.dummy_latent = jnp.zeros((1, model_config.latent_encoding_size))
        super().__init__(
            model_config=model_config,
            observation_size=observation_size,
            action_size=action_size,
            vision_obs_period=vision_obs_period,
            preprocess_observations_fn=preprocess_observations_fn,
            activation=activation,
        )
        self.acting_encoder_vision = model_config.encoder.vision

    @staticmethod
    def agent_params_class() -> type[ActorCriticEnrichedAgentParams]:
        return ActorCriticEnrichedAgentParams

    def initialize(self, rng: PRNGKey) -> ActorCriticEnrichedNetworkParams:
        policy_key, value_key, encoder_key = jax.random.split(rng, 3)
        return ActorCriticEnrichedNetworkParams(
            policy=self.policy_module.init(policy_key, self.dummy_obs, self.dummy_latent),
            value=self.value_module.init(value_key, self.dummy_obs, self.dummy_latent),
            acting_encoder=self.acting_encoder_module.init(encoder_key, self.dummy_obs),
        )

    def apply_acting_encoder(
        self,
        preprocessor_params: PreprocessorParams,
        network_params: ActorCriticEnrichedNetworkParams,
        observation: Observation,
        repeat_output: bool = True,
    ) -> jax.Array:
        observation = self.preprocess_obs(preprocessor_params, observation)
        latent_encoding = self.acting_encoder_module.apply(
            network_params.acting_encoder, observation
        )
        if repeat_output and self.acting_encoder_vision:
            latent_encoding = jnp.repeat(latent_encoding, self.vision_obs_period or 1, axis=0)
        return latent_encoding

    def apply_policy_with_latents(
        self,
        preprocessor_params: PreprocessorParams,
        network_params: ActorCriticEnrichedNetworkParams,
        observation: Observation,
        latent_encoding: jax.Array,
    ) -> jax.Array:
        observation = self.preprocess_obs(preprocessor_params, observation)
        return self.policy_module.apply(network_params.policy, observation, latent_encoding)

    def apply_value_with_latents(
        self,
        preprocessor_params: PreprocessorParams,
        network_params: ActorCriticEnrichedNetworkParams,
        observation: Observation,
        latent_encoding: jax.Array,
    ) -> jax.Array:
        observation = self.preprocess_obs(preprocessor_params, observation)
        return jnp.squeeze(
            self.value_module.apply(network_params.value, observation, latent_encoding),
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
        terminal: bool = False,
    ) -> jax.Array:
        latent_encoding = self.apply_acting_encoder(
            preprocessor_params, network_params, observation, repeat_output=not terminal
        )
        return self.apply_value_with_latents(
            preprocessor_params, network_params, observation, latent_encoding
        )

    def make_acting_unroll_fn(
        self,
        agent_params: AgentParams[AgentNetworkParams],
        *,
        deterministic: bool = False,
        accumulate_pipeline_states: bool = False,
    ) -> GenerateUnrollFn:
        if self.vision:
            policy_factory = self.policy_metafactory(self.apply_policy_with_latents)
            encoder_factory = self.apply_acting_encoder
        else:
            policy_factory = self.policy_metafactory(self.apply_policy)
            encoder_factory = None
        return self.make_unroll_fn(
            agent_params=agent_params,
            policy_factory=policy_factory,
            apply_encoder_fn=encoder_factory,
            deterministic=deterministic,
            accumulate_pipeline_states=accumulate_pipeline_states,
        )