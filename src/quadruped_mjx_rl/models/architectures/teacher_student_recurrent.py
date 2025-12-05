from dataclasses import dataclass, field

import jax
from flax import linen
from jax import numpy as jnp

from quadruped_mjx_rl.models.acting import GenerateUnrollFn
from quadruped_mjx_rl.models.architectures.actor_critic_enriched import (
    ActorCriticEnrichedNetworks,
)
from quadruped_mjx_rl.models.architectures.configs_base import (
    ComponentNetworksArchitecture,
    register_model_config_class,
)
from quadruped_mjx_rl.models.architectures.teacher_student_base import (
    TeacherStudentAgentParams,
    TeacherStudentNetworkParams,
    TeacherStudentVisionConfig,
)
from quadruped_mjx_rl.models.base_modules import (
    ActivationFn,
    ModuleConfigCNN,
    ModuleConfigLSTM,
    ModuleConfigMixedModeRNN,
    ModuleConfigMLP,
)
from quadruped_mjx_rl.models.types import (
    identity_observation_preprocessor,
    PreprocessObservationFn,
    PreprocessorParams,
)
from quadruped_mjx_rl.models.types import RecurrentAgentState
from quadruped_mjx_rl.types import Observation, ObservationSize, PRNGKey


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
    ActorCriticEnrichedNetworks, ComponentNetworksArchitecture[TeacherStudentNetworkParams]
):
    def __init__(
        self,
        *,
        model_config: TeacherStudentRecurrentConfig,
        observation_size: ObservationSize,
        action_size: int,
        preprocess_observations_fn: PreprocessObservationFn = identity_observation_preprocessor,
        activation: ActivationFn = linen.swish,
    ):
        """Make teacher-student network with preprocessor."""
        self.student_encoder_obs_key = model_config.student_obs_key
        self.student_proprio_obs_key = model_config.student_proprio_obs_key
        self.student_encoder_module = model_config.encoder.create(
            activation_fn=activation,
            activate_final=True,
            extra_final_layer_size=model_config.latent_encoding_size,
        )

        recurrent_input_size = (
            model_config.student.convolutional.dense.layer_sizes[-1]
            + model_config.student.proprioceptive_preprocessing.layer_sizes[-1]
        )
        recurrent_output_size = model_config.student.recurrent.recurrent_size
        self.dummy_agent_state = RecurrentAgentState(
            recurrent_carry=(
                jnp.zeros((1, recurrent_output_size)),
                jnp.zeros((1, recurrent_output_size)),
            ),
            recurrent_buffer=jnp.zeros(
                (
                    1,
                    model_config.student_recurrent_backpropagation_steps,
                    recurrent_input_size,
                )
            ),
            done_buffer=jnp.ones((1, model_config.student_recurrent_backpropagation_steps, 1)),
        )

        super().__init__(
            model_config=model_config,
            observation_size=observation_size,
            action_size=action_size,
            preprocess_observations_fn=preprocess_observations_fn,
            activation=activation,
        )

    @staticmethod
    def agent_params_class() -> type[TeacherStudentAgentParams]:
        return TeacherStudentAgentParams

    def initialize(self, rng: PRNGKey) -> TeacherStudentNetworkParams:
        policy_key, value_key, teacher_key, student_key = jax.random.split(rng, 4)
        student_params_key, student_dummy_key = jax.random.split(student_key, 2)
        dummy_done = jnp.zeros((1, 1))
        return TeacherStudentNetworkParams(
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
                teacher_key, self.dummy_obs[self.acting_encoder_obs_key]
            ),
            student_encoder=self.student_encoder_module.init(
                student_params_key,
                self.dummy_obs[self.student_encoder_obs_key],
                self.dummy_obs[self.student_proprio_obs_key],
                dummy_done,
                self.dummy_agent_state.recurrent_carry,
                self.dummy_agent_state.recurrent_buffer,
                self.dummy_agent_state.done_buffer,
                student_dummy_key,
            ),
        )

    def init_student_carry(self, init_carry_key: PRNGKey) -> jax.Array:
        return self.student_encoder_module.initialize_carry(init_carry_key)

    def apply_student_encoder(
        self,
        preprocessor_params: PreprocessorParams,
        network_params: TeacherStudentNetworkParams,
        observation: Observation,
        done: jax.Array,
        recurrent_agent_state: RecurrentAgentState,
        reinitialization_key: PRNGKey,
    ) -> tuple[jax.Array, RecurrentAgentState]:
        observation = self.preprocess_obs(preprocessor_params, observation)
        encoding, recurrent_carry, recurrent_buffer, done_buffer = (
            self.student_encoder_module.apply(
                network_params.student_encoder,
                observation[self.student_encoder_obs_key],
                observation[self.student_proprio_obs_key],
                done,
                recurrent_agent_state.recurrent_carry,
                recurrent_agent_state.recurrent_buffer,
                recurrent_agent_state.done_buffer,
                reinitialization_key,
            )
        )
        return encoding, RecurrentAgentState(recurrent_carry, recurrent_buffer, done_buffer)

    def recurrent_unroll_factory(
        self,
        preprocessor_params: PreprocessorParams,
        network_params: TeacherStudentNetworkParams,
    ) -> GenerateUnrollFn:
        pass  # TODO