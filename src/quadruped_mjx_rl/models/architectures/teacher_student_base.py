import functools
from collections.abc import Mapping
from dataclasses import dataclass, field

import jax
from flax import linen
from flax.struct import dataclass as flax_dataclass

from quadruped_mjx_rl import types
from quadruped_mjx_rl.models import distributions
from quadruped_mjx_rl.models.architectures.actor_critic_base import (
    ActorCriticConfig,
    ActorCriticNetworkParams,
    ActorCriticAgent,
)
from quadruped_mjx_rl.models.architectures.configs_base import (
    AgentModel,
    ComponentNetworksArchitecture, register_model_config_class,
)
from quadruped_mjx_rl.models.base_modules import ActivationFn, MLP
from quadruped_mjx_rl.models.networks_utils import (
    AgentParams,
    FeedForwardNetwork,
    make_network,
    policy_factory,
    RecurrentHiddenState,
    RecurrentNetwork,
)


@dataclass
class TeacherStudentConfig(ActorCriticConfig):
    @dataclass
    class ModulesConfig(ActorCriticConfig.ModulesConfig):
        teacher_obs_key: str = "environment_privileged"
        student_obs_key: str = "proprioceptive_history"
        common_obs_key: str = "proprioceptive"
        latent_obs_key: str = "latent"
        encoder_dense: list[int] = field(default_factory=lambda: [256, 256])
        student_dense: list[int] = field(default_factory=lambda: [256, 256])

    modules: ModulesConfig = field(default_factory=ModulesConfig)
    latent_size: int = 16

    @classmethod
    def config_class_key(cls) -> str:
        return "TeacherStudent"

    @classmethod
    def get_model_class(cls) -> type["TeacherStudentAgent"]:
        return TeacherStudentAgent


register_model_config_class(TeacherStudentConfig)


@flax_dataclass
class TeacherStudentNetworkParams(ActorCriticNetworkParams):
    """Contains training state for the learner."""

    teacher_encoder: types.Params
    student_encoder: types.Params


@flax_dataclass
class TeacherStudentAgentParams(AgentParams[TeacherStudentNetworkParams]):
    """Contains training state for the full agent."""

    def restore_params(
        self,
        restore_params: "TeacherStudentAgentParams",
        restore_value: bool = False,
    ):
        value_params = (
            restore_params.network_params.value if restore_value else self.network_params.value
        )
        return self.replace(
            network_params=TeacherStudentNetworkParams(
                policy=restore_params.network_params.policy,
                value=value_params,
                teacher_encoder=restore_params.network_params.teacher_encoder,
                student_encoder=restore_params.network_params.student_encoder,
            ),
            preprocessor_params=restore_params.preprocessor_params,
        )


class TeacherStudentNetworks(ComponentNetworksArchitecture[TeacherStudentNetworkParams]):
    def __init__(
        self,
        *,
        model_config: TeacherStudentConfig,
        observation_size: types.ObservationSize,
        output_size: int,
        preprocess_observations_fn: types.PreprocessObservationFn = (
            types.identity_observation_preprocessor
        ),
        activation: ActivationFn = linen.swish,
    ):
        """Make teacher-student network with preprocessor."""

        teacher_encoder_module = MLP(
            layer_sizes=model_config.modules.encoder_dense + [model_config.latent_size],
            activation=activation,
            activate_final=True,
        )
        self._teacher_encoder_network = make_network(
            module=teacher_encoder_module,
            obs_size=observation_size,
            preprocess_observations_fn=preprocess_observations_fn,
            preprocess_obs_keys=(model_config.modules.teacher_obs_key,),
            apply_to_obs_keys=(model_config.modules.teacher_obs_key,),
            squeeze_output=False,
        )
        student_encoder_module = MLP(
            layer_sizes=model_config.modules.student_dense + [model_config.latent_size],
            activation=activation,
            activate_final=True,
        )
        self._student_encoder_network = make_network(
            module=student_encoder_module,
            obs_size=observation_size,
            preprocess_observations_fn=preprocess_observations_fn,
            preprocess_obs_keys=(model_config.modules.student_obs_key,),
            apply_to_obs_keys=(model_config.modules.student_obs_key,),
            squeeze_output=False,
        )

        self._latent_obs_key = model_config.modules.latent_obs_key
        policy_module = MLP(
            layer_sizes=model_config.modules.policy + [output_size],
            activation=activation,
            activate_final=False,
        )
        self._policy_network = make_network(
            module=policy_module,
            obs_size=observation_size,
            preprocess_observations_fn=preprocess_observations_fn,
            preprocess_obs_keys=(model_config.modules.common_obs_key,),
            apply_to_obs_keys=(model_config.modules.common_obs_key, self._latent_obs_key),
            squeeze_output=False,
            concatenate_inputs=True,
        )
        value_module = MLP(
            layer_sizes=model_config.modules.value + [1],
            activation=activation,
            activate_final=False,
        )
        self._value_network = make_network(
            module=value_module,
            obs_size=observation_size,
            preprocess_observations_fn=preprocess_observations_fn,
            preprocess_obs_keys=(model_config.modules.common_obs_key,),
            apply_to_obs_keys=(model_config.modules.common_obs_key, self._latent_obs_key),
            squeeze_output=True,
            concatenate_inputs=True,
        )

    def initialize(self, rng: types.PRNGKey) -> TeacherStudentNetworkParams:
        policy_key, value_key, teacher_key, student_key = jax.random.split(rng, 4)
        return TeacherStudentNetworkParams(
            policy=self._policy_network.init(policy_key),
            value=self._value_network.init(value_key),
            teacher_encoder=self._teacher_encoder_network.init(teacher_key),
            student_encoder=self._student_encoder_network.init(student_key),
        )

    def apply_teacher_encoder_module(
        self, params: TeacherStudentAgentParams, observation: types.Observation
    ) -> jax.Array:
        return self._teacher_encoder_network.apply(
            params.preprocessor_params, params.network_params.teacher_encoder, observation
        )

    def apply_student_encoder_module(
        self, params: TeacherStudentAgentParams, observation: types.Observation
    ) -> jax.Array:
        return self._student_encoder_network.apply(
            params.preprocessor_params, params.network_params.student_encoder, observation
        )

    def apply_policy_module(
        self,
        params: TeacherStudentAgentParams,
        observation: types.Observation,
        latent_encoding: jax.Array,
    ) -> jax.Array:
        observation = observation | {self._latent_obs_key: latent_encoding}
        return self._policy_network.apply(
            params.preprocessor_params, params.network_params.policy, observation
        )

    def apply_value_module(
        self,
        params: TeacherStudentAgentParams,
        observation: types.Observation,
        latent_encoding: jax.Array,
    ) -> jax.Array:
        observation = observation | {self._latent_obs_key: latent_encoding}
        return self._value_network.apply(
            params.preprocessor_params, params.network_params.policy, observation
        )


class TeacherStudentAgent(
    ActorCriticAgent, AgentModel[TeacherStudentNetworkParams]
):

    @staticmethod
    def agent_params_class() -> type[TeacherStudentAgentParams]:
        return TeacherStudentAgentParams

    @staticmethod
    def networks_class() -> type[TeacherStudentNetworks]:
        return TeacherStudentNetworks

    def apply_rollout_value(
        self, params: TeacherStudentAgentParams, observation: types.Observation
    ) -> jax.Array:
        latent_encoding = self.networks.apply_teacher_encoder_module(params, observation)
        return self.networks.apply_value_module(params, observation, latent_encoding)

    def apply_rollout_policy(
        self,
        params: TeacherStudentAgentParams,
        observation: types.Observation,
    ) -> jax.Array:
        latent_encoding = self.networks.apply_teacher_encoder_module(params, observation)
        return self.networks.apply_policy_module(params, observation, latent_encoding)

    def apply_student_policy(
        self,
        params: TeacherStudentAgentParams,
        observation: types.Observation,
    ) -> jax.Array:
        latent_encoding = self.networks.apply_student_encoder_module(params, observation)
        return self.networks.apply_policy_module(params, observation, latent_encoding)


    def policy_metafactory(self):
        """Creates params and inference function for the Teacher and Student agents."""

        def make_teacher_policy(params: TeacherStudentAgentParams, deterministic: bool = False):
            return policy_factory(
                policy_apply=self.apply_rollout_policy,
                parametric_action_distribution=self.parametric_action_distribution,
                params=params,
                deterministic=deterministic,
            )

        def make_student_policy(params: TeacherStudentAgentParams, deterministic: bool = False):
            return policy_factory(
                policy_apply=self.apply_student_policy,
                parametric_action_distribution=self.parametric_action_distribution,
                params=params,
                deterministic=deterministic,
                recurrent=isinstance(self.networks._student_encoder_network, RecurrentNetwork),
            )

        return make_teacher_policy, make_student_policy

