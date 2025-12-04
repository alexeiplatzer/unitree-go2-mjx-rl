from dataclasses import dataclass, field

import jax
from jax import numpy as jnp
from flax import linen

import quadruped_mjx_rl.models.types
from quadruped_mjx_rl import types
from quadruped_mjx_rl.models.architectures.actor_critic_base import (
    ActorCriticNetworks,
)
from quadruped_mjx_rl.models.architectures.configs_base import (
    ComponentNetworksArchitecture,
    register_model_config_class,
)
from quadruped_mjx_rl.models.architectures.teacher_student_base import (
    TeacherStudentAgentParams,
    TeacherStudentNetworkParams,
    TeacherStudentVisionConfig
)
from quadruped_mjx_rl.models.base_modules import (
    ActivationFn,
    ModuleConfigCNN,
    ModuleConfigLSTM,
    ModuleConfigMixedModeRNN,
    ModuleConfigMLP,
)
from quadruped_mjx_rl.models.networks_utils import (
    make_network,
    policy_factory,
)
from quadruped_mjx_rl.models.types import RecurrentAgentState, RecurrentCarry


@dataclass
class TeacherStudentRecurrentConfig(TeacherStudentVisionConfig):

    encoder_obs_key: str = "privileged_terrain_map"
    student_obs_key: str = "pixels/frontal_ego/rgb_adjusted"
    student_proprio_obs_key: str = "proprioceptive"
    encoder: ModuleConfigCNN = field(
        default_factory=lambda: ModuleConfigCNN(
            filter_sizes=[16, 16, 16], dense=ModuleConfigMLP(layer_sizes=[256, 256])
        )
    )
    student: ModuleConfigMixedModeRNN = field(
        default_factory=lambda: ModuleConfigMixedModeRNN(
            convolutional=ModuleConfigCNN(
                filter_sizes=[32, 32, 32], dense=ModuleConfigMLP(layer_sizes=[128, 128, 64])
            ),
            proprioceptive_preprocessing=ModuleConfigMLP(layer_sizes=[64, 64]),
            recurrent=ModuleConfigLSTM(
                recurrent_size=16, dense=ModuleConfigMLP(layer_sizes=[16])
            ),
        )
    )
    student_recurrent_backpropagation_steps: int = 64

    @classmethod
    def config_class_key(cls) -> str:
        return "TeacherStudentRecurrent"

    @classmethod
    def get_model_class(cls) -> type["TeacherStudentRecurrentNetworks"]:
        return TeacherStudentRecurrentNetworks


register_model_config_class(TeacherStudentRecurrentConfig)


class TeacherStudentRecurrentNetworks(
    ComponentNetworksArchitecture[TeacherStudentNetworkParams]
):
    def __init__(
        self,
        *,
        model_config: TeacherStudentRecurrentConfig,
        observation_size: types.ObservationSize,
        output_size: int,
        preprocess_observations_fn: quadruped_mjx_rl.models.types.PreprocessObservationFn = (
            quadruped_mjx_rl.models.types.identity_observation_preprocessor
        ),
        activation: ActivationFn = linen.swish,
    ):
        """Make teacher-student network with preprocessor."""

        teacher_encoder_module = model_config.encoder.create(
            activation_fn=activation,
            activate_final=True,
            extra_final_layer_size=model_config.latent_encoding_size,
        )
        self._teacher_encoder_network = make_network(
            module=teacher_encoder_module,
            obs_size=observation_size,
            preprocess_observations_fn=preprocess_observations_fn,
            preprocess_obs_keys=(),
            apply_to_obs_keys=(model_config.encoder_obs_key,),
        )

        self.dummy_agent_state = RecurrentAgentState(
            recurrent_carry=(
                jnp.zeros((1, model_config.student.recurrent.recurrent_size)),
                jnp.zeros((1, model_config.student.recurrent.recurrent_size)),
            ),
            recurrent_buffer=jnp.zeros(
                (
                    1,
                    model_config.student_recurrent_backpropagation_steps,
                    model_config.student.recurrent.recurrent_size,
                )
            ),
            done_buffer=jnp.ones((1, model_config.student_recurrent_backpropagation_steps, 1)),
        )
        student_encoder_module = model_config.student.create(
            activation_fn=activation,
            activate_final=True,
            extra_final_layer_size=model_config.latent_encoding_size,
        )
        self._student_encoder_network = make_network(
            module=student_encoder_module,
            obs_size=observation_size,
            preprocess_observations_fn=preprocess_observations_fn,
            preprocess_obs_keys=(model_config.student_proprio_obs_key,),
            apply_to_obs_keys=(
                model_config.student_obs_key,
                model_config.student_proprio_obs_key,
            ),
            concatenate_inputs=False,
            recurrent=True,
        )

        self._latent_obs_key = model_config.latent_obs_key
        self.actor_critic = ActorCriticNetworks(
            model_config=model_config,
            observation_size=observation_size,
            action_size=output_size,
            preprocess_observations_fn=preprocess_observations_fn,
            activation=activation,
            extra_input_keys=(self._latent_obs_key,),
        )

    def initialize(self, rng: types.PRNGKey) -> TeacherStudentNetworkParams:
        policy_key, value_key, teacher_key, student_key = jax.random.split(rng, 4)
        student_params_key, student_dummy_key = jax.random.split(student_key, 2)
        dummy_done = jnp.zeros(())
        return TeacherStudentNetworkParams(
            policy=self.actor_critic.policy_network.init(policy_key),
            value=self.actor_critic.value_network.init(value_key),
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
        return self._student_encoder_network.apply_differentiable(
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
        return self.actor_critic.policy_network.apply(
            params.preprocessor_params, params.network_params.policy, observation
        )

    def apply_value_module(
        self,
        params: TeacherStudentAgentParams,
        observation: types.Observation,
        latent_encoding: jax.Array,
    ) -> jax.Array:
        observation = observation | {self._latent_obs_key: latent_encoding}
        return self.actor_critic.value_network.apply(
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
        recurrent_carry: RecurrentCarry,
        key: types.PRNGKey,
    ) -> tuple[jax.Array, RecurrentCarry]:
        recurrent_agent_state = RecurrentAgentState(
            recurrent_carry=recurrent_carry,
            recurrent_buffer=self.networks.dummy_agent_state.recurrent_buffer,
            done_buffer=self.networks.dummy_agent_state.done_buffer,
        )
        latent_encoding, recurrent_agent_state = self.networks.apply_student_encoder(
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
