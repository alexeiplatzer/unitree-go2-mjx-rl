import functools
from collections.abc import Callable
from dataclasses import dataclass

import jax
from flax import linen
from flax.struct import dataclass as flax_dataclass
from jax import numpy as jnp

from quadruped_mjx_rl.models import distributions, networks_utils
from quadruped_mjx_rl.models.acting import (
    GenerateUnrollFn, proprioceptive_unroll_factory,
    vision_unroll_factory,
)
from quadruped_mjx_rl.models.architectures.configs_base import (
    ComponentNetworksArchitecture,
    ModelConfig,
    register_model_config_class,
)
from quadruped_mjx_rl.models.base_modules import ActivationFn
from quadruped_mjx_rl.models.base_modules import ModuleConfigMLP
from quadruped_mjx_rl.models.types import (
    AgentNetworkParams, AgentParams, identity_observation_preprocessor, Params, Policy,
    PolicyFactory, PreprocessObservationFn, PreprocessorParams,
)
from quadruped_mjx_rl.types import Action, Extra, Observation, ObservationSize, PRNGKey


@dataclass
class ActorCriticConfig(ModelConfig):
    policy: ModuleConfigMLP
    value: ModuleConfigMLP

    @classmethod
    def default(cls) -> "ActorCriticConfig":
        default_super = ModelConfig.default()
        return ActorCriticConfig(
            policy=default_super.policy,
            value=ModuleConfigMLP(
                layer_sizes=[256, 256, 256, 256, 256], obs_key="proprioceptive"
            ),
        )

    @property
    def vision(self) -> bool:
        return super().vision or self.value.vision

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
        vision_obs_period: int | None = None,
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
        self.policy_module = model_config.policy.create(
            activation_fn=activation,
            activate_final=False,
            extra_final_layer_size=self.parametric_action_distribution.param_size,
        )
        self.value_module = model_config.value.create(
            activation_fn=activation, activate_final=False, extra_final_layer_size=1
        )
        self.vision = model_config.vision
        self.vision_obs_period = vision_obs_period

    @staticmethod
    def agent_params_class() -> type[ActorCriticAgentParams]:
        return ActorCriticAgentParams

    def initialize(self, rng: PRNGKey) -> ActorCriticNetworkParams:
        policy_key, value_key = jax.random.split(rng)
        return ActorCriticNetworkParams(
            policy=self.policy_module.init(policy_key, self.dummy_obs),
            value=self.value_module.init(value_key, self.dummy_obs),
        )

    def apply_policy(
        self,
        preprocessor_params: PreprocessorParams,
        network_params: ActorCriticNetworkParams,
        observation: Observation,
    ) -> jax.Array:
        observation = self.preprocess_obs(preprocessor_params, observation)
        return self.policy_module.apply(network_params.policy, observation)

    def apply_value(
        self,
        preprocessor_params: PreprocessorParams,
        network_params: ActorCriticNetworkParams,
        observation: Observation,
        terminal: bool = False,
    ) -> jax.Array:
        observation = self.preprocess_obs(preprocessor_params, observation)
        return jnp.squeeze(
            self.value_module.apply(network_params.value, observation),
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

    def make_acting_unroll_fn(
        self,
        agent_params: AgentParams[AgentNetworkParams],
        *,
        deterministic: bool = False,
        accumulate_pipeline_states: bool = False,
    ) -> GenerateUnrollFn:
        return self.make_unroll_fn(
            agent_params=agent_params,
            policy_factory=self.policy_metafactory(self.apply_policy),
            deterministic=deterministic,
        )

    def make_unroll_fn(
        self,
        agent_params: AgentParams[AgentNetworkParams],
        *,
        policy_factory: PolicyFactory,
        deterministic: bool = False,
        apply_encoder_fn: Callable | None = None,
        accumulate_pipeline_states: bool = False,
    ) -> GenerateUnrollFn:
        acting_policy = policy_factory(agent_params, deterministic)
        if self.vision:
            if apply_encoder_fn is None:
                raise ValueError("Vision unrolls require an encoder function.")
            vision_encoder = functools.partial(
                apply_encoder_fn,
                preprocessor_params=agent_params.preprocessor_params,
                network_params=agent_params.network_params,
                repeat_output=False,
            )
            return vision_unroll_factory(
                policy=acting_policy,
                vision_encoder=vision_encoder,
                vision_obs_period=self.vision_obs_period,
                accumulate_pipeline_states=accumulate_pipeline_states,
            )
        else:
            return proprioceptive_unroll_factory(
                acting_policy, accumulate_pipeline_states=accumulate_pipeline_states
            )

