from collections.abc import Mapping
from dataclasses import dataclass, field

import jax
from jax import numpy as jnp
from flax import linen

from quadruped_mjx_rl import types
from quadruped_mjx_rl.models import distributions
from quadruped_mjx_rl.models.architectures.actor_critic_base import (
    ActorCriticAgent,
    ActorCriticConfig,
)
from quadruped_mjx_rl.models.architectures.configs_base import (
    AgentModel, ComponentNetworksArchitecture,
    register_model_config_class,
)
from quadruped_mjx_rl.models.architectures.teacher_student_base import (
    TeacherStudentAgentParams, TeacherStudentConfig,
    TeacherStudentAgent, TeacherStudentNetworkParams,
)
from quadruped_mjx_rl.models.architectures.teacher_student_vision import (
    TeacherStudentVisionConfig, TeacherStudentVisionNetworks,
)
from quadruped_mjx_rl.models.base_modules import ActivationFn, CNN, MLP, LSTM, MixedModeRNN
from quadruped_mjx_rl.models.networks_utils import make_network, RecurrentAgentState, policy_factory
from quadruped_mjx_rl.types import RecurrentHiddenState


@dataclass
class TeacherStudentRecurrentConfig(TeacherStudentVisionConfig):
    @dataclass
    class ModulesConfig(TeacherStudentVisionConfig.ModulesConfig):
        teacher_obs_key: str = "privileged_terrain_map"
        student_visual_obs_key: str = "pixels/frontal_ego/rgb_adjusted"
        student_proprioceptive_obs_key: str = "proprioceptive"
        common_obs_key: str = "proprioceptive"
        latent_obs_key: str = "latent"
        encoder_convolutional: list[int] = field(default_factory=lambda: [16, 16, 16])
        student_convolutional: list[int] = field(default_factory=lambda: [32, 32, 32])
        student_vision_dense: list[int] = field(default_factory=lambda: [128, 128])
        student_vision_latent_size: int = 64
        student_proprio_dense: list[int] = field(default_factory=lambda: [64, 32])
        student_recurrent_size: int = 16
        student_recurrent_backpropagation_steps: int = 64,
        student_dense: list[int] = field(default_factory=lambda: [16])

    modules: ModulesConfig = field(default_factory=ModulesConfig)

    @classmethod
    def config_class_key(cls) -> str:
        return "TeacherStudentRecurrent"

    @classmethod
    def get_model_class(cls) -> type["TeacherStudentRecurrentNetworks"]:
        return TeacherStudentRecurrentNetworks


register_model_config_class(TeacherStudentRecurrentConfig)


class TeacherStudentRecurrentNetworks(ComponentNetworksArchitecture[TeacherStudentNetworkParams]):
    def __init__(
        self,
        *,
        model_config: TeacherStudentRecurrentConfig,
        observation_size: types.ObservationSize,
        output_size: int,
        preprocess_observations_fn: types.PreprocessObservationFn = (
            types.identity_observation_preprocessor
        ),
        activation: ActivationFn = linen.swish,
    ):
        """Make teacher-student network with preprocessor."""

        teacher_encoder_module = CNN(
            num_filters=model_config.modules.encoder_convolutional,
            dense_layer_sizes=model_config.modules.encoder_dense + [model_config.latent_size],
            activation=activation,
            activate_final=True,
        )
        self._teacher_encoder_network = make_network(
            module=teacher_encoder_module,
            obs_size=observation_size,
            preprocess_observations_fn=preprocess_observations_fn,
            preprocess_obs_keys=(),
            apply_to_obs_keys=(model_config.modules.teacher_obs_key,),
            squeeze_output=False,
        )

        student_vision_encoder_module = CNN(
            num_filters=model_config.modules.student_convolutional,
            dense_layer_sizes=model_config.modules.student_vision_dense + [
                model_config.modules.student_vision_latent_size],
            activation=activation,
            activate_final=True,
        )
        student_proprioceptive_encoder_module = MLP(
            layer_sizes=model_config.modules.student_proprio_dense,
            activation=activation,
            activate_final=True,
        )
        student_recurrent_cell_module = LSTM(
            recurrent_layer_size=model_config.modules.student_recurrent_size,
            dense_layer_sizes=model_config.modules.student_dense + [model_config.latent_size],
        )
        self.dummy_agent_state = RecurrentAgentState(
            recurrent_carry=(
                jnp.zeros((1, model_config.modules.student_recurrent_size)),
                jnp.zeros((1, model_config.modules.student_recurrent_size)),
            ),
            recurrent_buffer=jnp.zeros(
                (
                    1,
                    model_config.modules.student_recurrent_backpropagation_steps,
                    model_config.modules.student_recurrent_size,
                )
            ),
            done_buffer=jnp.ones(
                (1, model_config.modules.student_recurrent_backpropagation_steps, 1)
            ),
        )
        student_encoder_module = MixedModeRNN(
            convolutional_module=student_vision_encoder_module,
            proprioceptive_preprocessing_module=student_proprioceptive_encoder_module,
            recurrent_module=student_recurrent_cell_module,
        )
        self._student_encoder_network = make_network(
            module=student_encoder_module,
            obs_size=observation_size,
            preprocess_observations_fn=preprocess_observations_fn,
            preprocess_obs_keys=(model_config.modules.student_proprioceptive_obs_key,),
            apply_to_obs_keys=(model_config.modules.student_visual_obs_key,
                model_config.modules.student_proprioceptive_obs_key),
            squeeze_output=False,
            concatenate_inputs=False,
            recurrent=True,
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
        student_params_key, student_dummy_key = jax.random.split(student_key, 2)
        dummy_done = jnp.zeros(())
        return TeacherStudentNetworkParams(
            policy=self._policy_network.init(policy_key),
            value=self._value_network.init(value_key),
            teacher_encoder=self._teacher_encoder_network.init(teacher_key),
            student_encoder=self._student_encoder_network.init(
                student_params_key, dummy_done, self.dummy_agent_state, student_dummy_key
            ),
        )

    def apply_teacher_encoder_module(
        self, params: TeacherStudentAgentParams, observation: types.Observation
    ) -> jax.Array:
        return self._teacher_encoder_network.apply(
            params.preprocessor_params, params.network_params.teacher_encoder, observation
        )

    def init_student_carry(self, init_carry_key: types.PRNGKey) -> jax.Array:
        return self._student_encoder_network.init_recurrent_carry(init_carry_key)

    def apply_student_encoder_module(
        self,
        params: TeacherStudentAgentParams,
        observation: types.Observation,
        done: jax.Array,
        recurrent_agent_state: RecurrentAgentState,
        reinitialization_key: types.PRNGKey,
    ) -> tuple[jax.Array, RecurrentAgentState]:
        return self._student_encoder_network.apply(
            params.preprocessor_params,
            params.network_params.student_encoder,
            observation,
            done,
            recurrent_agent_state,
            reinitialization_key,
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
            params.preprocessor_params, params.network_params.value, observation
        )


class TeacherStudentRecurrentAgent(TeacherStudentAgent):

    @staticmethod
    def agent_params_class() -> type[TeacherStudentAgentParams]:
        return TeacherStudentAgentParams

    @staticmethod
    def networks_class() -> type[TeacherStudentRecurrentNetworks]:
        return TeacherStudentRecurrentNetworks

    def apply_student_policy(
        self,
        params: TeacherStudentAgentParams,
        observation: types.Observation,
    ) -> jax.Array:
        raise NotImplementedError("Use apply_recurrent_student_policy instead.")

    def apply_recurrent_student_policy(
        self,
        params: TeacherStudentAgentParams,
        observation: types.Observation,
        done: jax.Array,
        recurrent_carry: RecurrentHiddenState,
        key: types.PRNGKey,
    ) -> tuple[jax.Array, RecurrentHiddenState]:
        recurrent_agent_state = RecurrentAgentState(
            recurrent_carry=recurrent_carry,
            recurrent_buffer=self.networks.dummy_agent_state.recurrent_buffer,
            done_buffer=self.networks.dummy_agent_state.done_buffer,
        )
        latent_encoding, recurrent_agent_state = self.networks.apply_student_encoder_module(
            params=params,
            observation=observation,
            done=done,
            recurrent_agent_state=recurrent_agent_state,
            key=key,
        )
        policy_logits = self.networks.apply_policy_module(params, observation, latent_encoding)
        return policy_logits, recurrent_agent_state.recurrent_carry

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
                policy_apply=self.apply_recurrent_student_policy,
                parametric_action_distribution=self.parametric_action_distribution,
                params=params,
                deterministic=deterministic,
                recurrent=True,
            )

        return make_teacher_policy, make_student_policy
