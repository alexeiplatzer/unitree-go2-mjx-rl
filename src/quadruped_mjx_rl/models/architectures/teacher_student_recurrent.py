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
    TeacherStudentConfig, TeacherStudentNetworkParams,
)
from quadruped_mjx_rl.models.base_modules import (
    ActivationFn,
    ModuleConfigCNN,
    ModuleConfigLSTM,
    ModuleConfigMixedModeCNN, ModuleConfigMixedModeRNN,
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
class TeacherStudentRecurrentConfig(TeacherStudentConfig):
    student: ModuleConfigMixedModeRNN
    student_recurrent_backpropagation_steps: int

    @property
    def recurrent(self) -> bool:
        return True

    @classmethod
    def default(cls) -> "TeacherStudentRecurrentConfig":
        default_super = TeacherStudentConfig.default_mixed()
        return TeacherStudentRecurrentConfig(
            policy=ModuleConfigMLP(
                layer_sizes=default_super.policy.layer_sizes, obs_key="proprioceptive_history"
            ),
            value=ModuleConfigMLP(
                layer_sizes=default_super.value.layer_sizes, obs_key="proprioceptive_history"
            ),
            encoder=ModuleConfigMixedModeCNN(
                vision_preprocessing=ModuleConfigCNN(
                    filter_sizes=default_super.encoder.vision_preprocessing.filter_sizes,
                    dense=default_super.encoder.vision_preprocessing.dense,
                    obs_key="privileged_terrain_map"
                ),
                joint_processing=default_super.encoder.joint_processing,
            ),
            student=ModuleConfigMixedModeRNN(
                convolutional=ModuleConfigCNN(
                    filter_sizes=[16, 32, 32],
                    dense=ModuleConfigMLP(layer_sizes=[128, 64]),
                    obs_key="pixels/ego_frontal/rgb_adjusted"
                ),
                proprioceptive_preprocessing=ModuleConfigMLP(
                    layer_sizes=[128, 64], obs_key="proprioceptive"
                ),
                recurrent=ModuleConfigLSTM(
                    recurrent_size=128, dense=ModuleConfigMLP(layer_sizes=[128])
                ),
            ),
            latent_encoding_size=128,
            student_recurrent_backpropagation_steps=64,
        )

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
        vision_obs_period: int | None = None,
        preprocess_observations_fn: PreprocessObservationFn = identity_observation_preprocessor,
        activation: ActivationFn = linen.swish,
    ):
        """Make teacher-student network with preprocessor."""
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
            vision_obs_period=vision_obs_period,
            preprocess_observations_fn=preprocess_observations_fn,
            activation=activation,
        )
        self.dummy_done = jnp.zeros((1,))
        self.recurrent_period = None

    @staticmethod
    def agent_params_class() -> type[TeacherStudentAgentParams]:
        return TeacherStudentAgentParams

    def set_recurrent_period(self, unroll_length: int):
        self.recurrent_period = unroll_length
        self.dummy_obs[self.student_encoder_module.convolutional_module.obs_key] = jnp.repeat(
            jnp.expand_dims(
                self.dummy_obs[self.student_encoder_module.convolutional_module.obs_key], 1
            ),
            self.recurrent_period // self.vision_obs_period,
            axis=1,
        )
        self.dummy_obs[
            self.student_encoder_module.proprioceptive_preprocessing_module.obs_key
        ] = jnp.repeat(
            jnp.expand_dims(
                self.dummy_obs[
                    self.student_encoder_module.proprioceptive_preprocessing_module.obs_key
                ],
                1,
            ),
            unroll_length,
            axis=1,
        )
        self.dummy_done = jnp.repeat(jnp.expand_dims(self.dummy_done, 1), unroll_length, axis=1)

    def initialize(self, rng: PRNGKey) -> TeacherStudentNetworkParams:
        if self.recurrent_period is None:
            raise ValueError("Recurrent period must be set before initialization.")
        policy_key, value_key, teacher_key, student_key = jax.random.split(rng, 4)
        student_params_key, student_dummy_key, student_agent_key = jax.random.split(
            student_key, 3
        )
        dummy_agent_state = jax.tree_util.tree_map(
            lambda x: jnp.expand_dims(x, axis=0), self.init_agent_state(student_agent_key)
        )
        return TeacherStudentNetworkParams(
            policy=self.policy_module.init(policy_key, self.dummy_obs, self.dummy_latent),
            value=self.value_module.init(value_key, self.dummy_obs, self.dummy_latent),
            acting_encoder=self.acting_encoder_module.init(teacher_key, self.dummy_obs),
            student_encoder=self.student_encoder_module.init(
                student_params_key,
                self.dummy_obs,
                self.dummy_done,
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
                observation,
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
            observation,
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
        return recurrent_unroll_factory(
            policy=policy,
            recurrent_encoder=recurrent_encoder,
            encoding_size=self.dummy_latent.shape[-1],
            init_carry_fn=self.init_student_carry,
            recurrent_obs_period=self.recurrent_period,
            vision_obs_period=self.vision_obs_period,
            accumulate_pipeline_states=accumulate_pipeline_states,
        )
