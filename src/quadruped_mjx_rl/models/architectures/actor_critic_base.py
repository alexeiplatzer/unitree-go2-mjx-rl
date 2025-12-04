from abc import abstractmethod
from dataclasses import dataclass, field
from collections.abc import Sequence

import jax
from flax import linen
from flax.struct import dataclass as flax_dataclass

from quadruped_mjx_rl.environments import Env, State
from quadruped_mjx_rl.environments.vision.vision_wrappers import VisionWrapper
from quadruped_mjx_rl.models.types import (
    AgentNetworkParams,
    Params,
    AgentParams,
    PolicyFactory,
    PreprocessObservationFn,
    identity_observation_preprocessor,
)
from quadruped_mjx_rl.models.base_modules import ActivationFn
from quadruped_mjx_rl.models import distributions, networks_utils
from quadruped_mjx_rl.models.architectures.configs_base import (
    ComponentNetworksArchitecture,
    ModelConfig,
    register_model_config_class,
)
from quadruped_mjx_rl.models.base_modules import ModuleConfigMLP
from quadruped_mjx_rl.types import Observation, PRNGKey, ObservationSize, Transition
from quadruped_mjx_rl.models.acting import generate_unroll


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
        extra_input_keys: tuple[str, ...] = (),
    ):
        """Make Actor Critic networks with preprocessor."""
        self.parametric_action_distribution = distributions.NormalTanhDistribution(
            event_size=action_size
        )
        policy_module = model_config.policy.create(
            activation_fn=activation,
            activate_final=False,
            extra_final_layer_size=self.parametric_action_distribution.param_size,
        )
        self.policy_network = networks_utils.make_network(
            module=policy_module,
            obs_size=observation_size,
            preprocess_observations_fn=preprocess_observations_fn,
            preprocess_obs_keys=(model_config.policy_obs_key,),
            apply_to_obs_keys=(model_config.policy_obs_key,) + extra_input_keys,
        )
        value_module = model_config.value.create(
            activation_fn=activation, activate_final=False, extra_final_layer_size=1
        )
        self.value_network = networks_utils.make_network(
            module=value_module,
            obs_size=observation_size,
            preprocess_observations_fn=preprocess_observations_fn,
            preprocess_obs_keys=(model_config.value_obs_key,),
            apply_to_obs_keys=(model_config.value_obs_key,) + extra_input_keys,
            squeeze_output=True,
        )

    @staticmethod
    def agent_params_class() -> type[ActorCriticAgentParams]:
        return ActorCriticAgentParams

    def initialize(self, rng: PRNGKey) -> ActorCriticNetworkParams:
        policy_key, value_key = jax.random.split(rng)
        return ActorCriticNetworkParams(
            policy=self.policy_network.init(policy_key),
            value=self.value_network.init(value_key),
        )

    def apply_policy(
        self, params: ActorCriticAgentParams, observation: Observation
    ) -> jax.Array:
        return self.policy_network.apply(
            params.preprocessor_params, params.network_params.policy, observation
        )

    def apply_value(
        self, params: ActorCriticAgentParams, observation: Observation
    ) -> jax.Array:
        return self.value_network.apply(
            params.preprocessor_params, params.network_params.value, observation
        )

    def get_acting_policy_factory(self) -> PolicyFactory[ActorCriticNetworkParams]:

        def make_policy(
            params: AgentParams[ActorCriticNetworkParams], deterministic: bool = False
        ):
            return networks_utils.policy_factory(
                policy_apply=self.apply_policy,
                parametric_action_distribution=self.parametric_action_distribution,
                params=params,
                deterministic=deterministic,
            )

        return make_policy

    def generate_training_unroll(
        self,
        params: ActorCriticAgentParams,
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
        )
        return env_state, transitions
