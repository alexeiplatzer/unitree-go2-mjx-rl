"""Actor-Critic architecture with an additional common encoder."""

from abc import abstractmethod
from dataclasses import dataclass, field
from collections.abc import Sequence

import jax
from flax import linen
from flax.struct import dataclass as flax_dataclass

from quadruped_mjx_rl.environments import Env, State, is_obs_key_vision
from quadruped_mjx_rl.environments.vision.vision_wrappers import VisionWrapper
from quadruped_mjx_rl.models.architectures import ActorCriticNetworkParams
from quadruped_mjx_rl.models.architectures.actor_critic_base import ActorCriticNetworks
from quadruped_mjx_rl.models.types import (
    AgentNetworkParams,
    Params,
    AgentParams,
    PolicyFactory,
    PreprocessObservationFn,
    identity_observation_preprocessor,
)
from quadruped_mjx_rl.models.base_modules import ActivationFn
from quadruped_mjx_rl.models import ActorCriticConfig, distributions, networks_utils
from quadruped_mjx_rl.models.architectures.configs_base import (
    ComponentNetworksArchitecture,
    ModelConfig,
    register_model_config_class,
)
from quadruped_mjx_rl.models.base_modules import ModuleConfigMLP
from quadruped_mjx_rl.types import Observation, PRNGKey, ObservationSize, Transition
from quadruped_mjx_rl.models.acting import generate_unroll


@dataclass
class ActorCriticEnrichedConfig(ActorCriticConfig):
    encoder_obs_key: str = "pixels/frontal_ego/rgb_adjusted"
    latent_obs_key: str = "latent"
    encoder: ModuleConfigMLP = field(
        default_factory=lambda: ModuleConfigMLP(layer_sizes=[256, 256])
    )
    latent_encoding_size: int = 16

    @classmethod
    def config_class_key(cls) -> str:
        return "ActorCritic"

    @classmethod
    def get_model_class(cls) -> type["ActorCriticNetworks"]:
        return ActorCriticNetworks


register_model_config_class(ActorCriticEnrichedConfig)


@flax_dataclass
class ActorCriticEnrichedNetworkParams(ActorCriticNetworkParams):
    encoder: Params


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
                encoder=restore_params.network_params.encoder,
            ),
            preprocessor_params=restore_params.preprocessor_params,
        )


class ActorCriticEnrichedNetworks(ActorCriticNetworks[ActorCriticEnrichedNetworkParams]):
    """An actor-critic architecture with an additional common encoder."""

    def __init__(
        self,
        *,
        model_config: ActorCriticEnrichedConfig,
        observation_size: ObservationSize,
        action_size: int,
        preprocess_observations_fn: PreprocessObservationFn = identity_observation_preprocessor,
        activation: ActivationFn = linen.swish,
        encoder_supersteps_size: int = 1,
    ):
        """Make Actor Critic networks with preprocessor."""
        # Do not preprocess visual observations
        preprocess_encoder_obs_keys, self.use_vision = (
            ((), True)
            if is_obs_key_vision(model_config.encoder_obs_key)
            else ((model_config.encoder_obs_key,), False)
        )
        encoder_module = model_config.encoder.create(
            activation_fn=activation,
            activate_final=True,
            extra_final_layer_size=model_config.latent_encoding_size,
        )
        self.acting_encoder_network = networks_utils.make_network(
            module=encoder_module,
            obs_size=observation_size,
            preprocess_observations_fn=preprocess_observations_fn,
            preprocess_obs_keys=preprocess_encoder_obs_keys,
            apply_to_obs_keys=(model_config.encoder_obs_key,),
        )
        self.encoder_supersteps = encoder_supersteps_size
        self.latent_obs_key = model_config.latent_obs_key
        super().__init__(
            model_config=model_config,
            observation_size=observation_size,
            action_size=action_size,
            preprocess_observations_fn=preprocess_observations_fn,
            activation=activation,
            extra_input_keys=(self.latent_obs_key,),
        )

    @staticmethod
    def agent_params_class() -> type[ActorCriticEnrichedAgentParams]:
        return ActorCriticEnrichedAgentParams

    def initialize(self, rng: PRNGKey) -> ActorCriticEnrichedNetworkParams:
        policy_key, value_key, encoder_key = jax.random.split(rng, 3)
        return ActorCriticEnrichedNetworkParams(
            policy=self.policy_network.init(policy_key),
            value=self.value_network.init(value_key),
            encoder=self.acting_encoder_network.init(encoder_key),
        )

    def apply_acting_encoder(
        self, params: ActorCriticEnrichedAgentParams, observation: Observation
    ) -> jax.Array:
        return self.acting_encoder_network.apply(
            params.preprocessor_params, params.network_params.encoder, observation
        )

    def apply_policy(
        self, params: ActorCriticEnrichedAgentParams, observation: Observation
    ) -> jax.Array:
        latent_encoding = self.apply_acting_encoder(params, observation)
        observation = observation | {self.latent_obs_key: latent_encoding}
        return self.policy_network.apply(
            params.preprocessor_params, params.network_params.policy, observation
        )

    def apply_value(
        self, params: ActorCriticEnrichedAgentParams, observation: Observation
    ) -> jax.Array:
        latent_encoding = self.apply_acting_encoder(params, observation)
        observation = observation | {self.latent_obs_key: latent_encoding}
        return self.value_network.apply(
            params.preprocessor_params, params.network_params.policy, observation
        )

    def generate_training_unroll(
        self,
        params: ActorCriticEnrichedAgentParams,
        env: Env | VisionWrapper,
        env_state: State,
        key: PRNGKey,
        unroll_length: int,
        extra_fields: Sequence[str] = (),
    ) -> tuple[State, Transition]:
        env_state, transitions, _ = generate_unroll(
            env=env,
            env_state=env_state,
            policy=self.get_acting_policy_factory()(params, deterministic=False),
            key=key,
            unroll_length=unroll_length,
            extra_fields=extra_fields,
            add_vision_obs=self.use_vision,
            proprio_steps_per_vision_step=self.encoder_supersteps,
        )
        return env_state, transitions
