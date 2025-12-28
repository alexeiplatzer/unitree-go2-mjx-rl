import functools
from collections.abc import Sequence
from dataclasses import dataclass, field

import jax
from flax import linen
from jax import numpy as jnp

from quadruped_mjx_rl.physics_pipeline import Env, State
from quadruped_mjx_rl.models.acting import GenerateUnrollFn, recurrent_unroll_factory
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
    RecurrentCarry,
    RecurrentAgentState,
)
from quadruped_mjx_rl.models.networks_utils import process_policy_logits
from quadruped_mjx_rl.types import Observation, ObservationSize, PRNGKey, Transition


@dataclass
class TeacherStudentRecurrentConfig(TeacherStudentVisionConfig):

    policy_obs_key: str = "proprioceptive_history"
    value_obs_key: str = "proprioceptive_history"
    encoder_obs_key: str = "privileged_terrain_map"
    student_obs_key: str = "pixels/frontal_ego/rgb_adjusted"
    student_proprio_obs_key: str = "proprioceptive"
    encoder: ModuleConfigCNN = field(
        default_factory=lambda: ModuleConfigCNN(
            filter_sizes=[16, 32, 32], dense=ModuleConfigMLP(layer_sizes=[256])
        )
    )
    student: ModuleConfigMixedModeRNN = field(
        default_factory=lambda: ModuleConfigMixedModeRNN(
            convolutional=ModuleConfigCNN(
                filter_sizes=[16, 32, 32], dense=ModuleConfigMLP(layer_sizes=[128, 64])
            ),
            proprioceptive_preprocessing=ModuleConfigMLP(layer_sizes=[128, 64]),
            recurrent=ModuleConfigLSTM(
                recurrent_size=128, dense=ModuleConfigMLP(layer_sizes=[128])
            ),
        )
    )
    student_recurrent_backpropagation_steps: int = 64
    student_encoder_supersteps: int = 4

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
        self.student_encoder_module = model_config.student.create(
            activation_fn=activation,
            activate_final=True,
            extra_final_layer_size=model_config.latent_encoding_size,
        )

        self.recurrent_input_size = (
            model_config.student.convolutional.dense.layer_sizes[-1]
            + model_config.student.proprioceptive_preprocessing.layer_sizes[-1]
        )
        self.recurrent_output_size = model_config.student.recurrent.recurrent_size
        self.buffers_length = model_config.student_recurrent_backpropagation_steps

        super().__init__(
            model_config=model_config,
            observation_size=observation_size,
            action_size=action_size,
            preprocess_observations_fn=preprocess_observations_fn,
            activation=activation,
        )
        self.student_encoder_supersteps = model_config.student_encoder_supersteps

    @staticmethod
    def agent_params_class() -> type[TeacherStudentAgentParams]:
        return TeacherStudentAgentParams

    def initialize(self, rng: PRNGKey) -> TeacherStudentNetworkParams:
        policy_key, value_key, teacher_key, student_key = jax.random.split(rng, 4)
        student_params_key, student_dummy_key, student_agent_key = jax.random.split(
            student_key, 3
        )
        dummy_agent_state = jax.tree_util.tree_map(
            lambda x: jnp.expand_dims(x, axis=0), self.init_agent_state(jax.random.PRNGKey(0))
        )
        dummy_done = jnp.zeros((1,))
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
                jnp.repeat(
                    jnp.expand_dims(self.dummy_obs[self.student_encoder_obs_key], 1),
                    self.student_encoder_supersteps,
                    axis=1,
                ),
                jnp.repeat(
                    jnp.expand_dims(self.dummy_obs[self.student_proprio_obs_key], 1),
                    self.student_encoder_supersteps * self.encoder_supersteps,
                    axis=1,
                ),
                jnp.repeat(
                    jnp.expand_dims(dummy_done, 1),
                    self.student_encoder_supersteps * self.encoder_supersteps,
                    axis=1,
                ),
                dummy_agent_state.recurrent_carry,
                dummy_agent_state.recurrent_buffer,
                dummy_agent_state.done_buffer,
                jnp.expand_dims(student_dummy_key, 0),
            ),
        )

    def init_student_carry(self, init_carry_key: PRNGKey) -> RecurrentCarry:
        return self.student_encoder_module.initialize_carry(init_carry_key)

    def init_agent_state(self, key: PRNGKey) -> RecurrentAgentState:
        recurrent_carry = self.init_student_carry(key)
        return RecurrentAgentState(
            recurrent_carry=recurrent_carry,
            recurrent_buffer=jnp.zeros((self.buffers_length, self.recurrent_input_size)),
            done_buffer=jnp.ones((self.buffers_length,)),
        )

    def apply_student_encoder(
        self,
        preprocessor_params: PreprocessorParams,
        network_params: TeacherStudentNetworkParams,
        observation: Observation,
        done: jax.Array,
        recurrent_agent_state: RecurrentAgentState,
        key: PRNGKey,
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
                key,
            )
        )
        return encoding, RecurrentAgentState(recurrent_carry, recurrent_buffer, done_buffer)

    def apply_student_acting_encoder(
        self,
        preprocessor_params: PreprocessorParams,
        network_params: TeacherStudentNetworkParams,
        observation: Observation,
        done: jax.Array,
        recurrent_carry: RecurrentCarry,
        key: PRNGKey,
    ) -> tuple[jax.Array, RecurrentCarry]:
        observation = self.preprocess_obs(preprocessor_params, observation)
        encoding, recurrent_carry = self.student_encoder_module.apply(
            network_params.student_encoder,
            observation[self.student_encoder_obs_key],
            observation[self.student_proprio_obs_key],
            done,
            recurrent_carry,
            key,
            method="encode",
        )
        return encoding, recurrent_carry

    def make_student_unroll_fn(
        self,
        agent_params: TeacherStudentAgentParams,
        *,
        deterministic: bool = False,
        accumulate_pipeline_states: bool = False,
    ) -> GenerateUnrollFn:
        policy = self.policy_metafactory(self.apply_policy_with_latents)(
            agent_params, deterministic
        )
        recurrent_encoder = functools.partial(
            self.apply_student_acting_encoder,
            preprocessor_params=agent_params.preprocessor_params,
            network_params=agent_params.network_params,
        )
        if not self.vision:
            vision_substeps = 0
            proprio_substeps = self.student_encoder_supersteps
        else:
            vision_substeps = self.student_encoder_supersteps
            proprio_substeps = self.encoder_supersteps

        return recurrent_unroll_factory(
            policy=policy,
            recurrent_encoder=recurrent_encoder,
            encoding_size=self.dummy_latent.shape[-1],
            init_carry_fn=self.init_student_carry,
            vision_substeps=vision_substeps,
            proprio_substeps=proprio_substeps,
            accumulate_pipeline_states=accumulate_pipeline_states,
        )
