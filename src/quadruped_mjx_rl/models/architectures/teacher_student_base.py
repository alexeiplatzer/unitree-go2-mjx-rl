from dataclasses import dataclass

import jax
from flax import linen
from flax.struct import dataclass as flax_dataclass
from jax import numpy as jnp

from quadruped_mjx_rl.models.acting import GenerateUnrollFn
from quadruped_mjx_rl.models.architectures.actor_critic_enriched import (
    ActorCriticEnrichedConfig,
    ActorCriticEnrichedNetworkParams,
    ActorCriticEnrichedNetworks,
)
from quadruped_mjx_rl.models.architectures.configs_base import (
    ComponentNetworksArchitecture,
    register_model_config_class,
)
from quadruped_mjx_rl.models.base_modules import (
    ActivationFn, ModuleConfigCNN,
    ModuleConfigMixedModeCNN, ModuleConfigMLP,
)
from quadruped_mjx_rl.models.types import (
    AgentParams, identity_observation_preprocessor, Params, PreprocessObservationFn,
    PreprocessorParams,
)
from quadruped_mjx_rl.types import Observation, ObservationSize, PRNGKey


@dataclass
class TeacherStudentConfig(ActorCriticEnrichedConfig):
    encoder: ModuleConfigMLP | ModuleConfigCNN | ModuleConfigMixedModeCNN
    student: ModuleConfigMLP | ModuleConfigCNN | ModuleConfigMixedModeCNN

    @classmethod
    def default(cls) -> "TeacherStudentConfig":
        default_super = ActorCriticEnrichedConfig.default()
        return TeacherStudentConfig(
            policy=default_super.policy,
            value=default_super.value,
            encoder=ModuleConfigMLP(layer_sizes=[256], obs_key="environment_privileged"),
            student=ModuleConfigMLP(layer_sizes=[512], obs_key="proprioceptive_history"),
            latent_encoding_size=default_super.latent_encoding_size,
        )

    @classmethod
    def default_vision(cls) -> "TeacherStudentConfig":
        default_super = TeacherStudentConfig.default()
        return TeacherStudentConfig(
            policy=default_super.policy,
            value=default_super.value,
            encoder=ModuleConfigCNN(
                filter_sizes=[16, 24, 32],
                dense=ModuleConfigMLP(layer_sizes=[256]),
                obs_key="pixels/frontal_ego/depth",
            ),
            student=ModuleConfigCNN(
                filter_sizes=[16, 24, 32],
                dense=ModuleConfigMLP(layer_sizes=[256]),
                obs_key="pixels/frontal_ego/rgb_adjusted",
            ),
            latent_encoding_size=default_super.latent_encoding_size,
        )

    @classmethod
    def default_mixed(cls) -> "TeacherStudentConfig":
        default_super = TeacherStudentConfig.default_vision()
        return TeacherStudentConfig(
            policy=default_super.policy,
            value=default_super.value,
            encoder=ModuleConfigMixedModeCNN(
                vision_preprocessing=ModuleConfigCNN(
                    filter_sizes=[16, 24, 32],
                    dense=ModuleConfigMLP(layer_sizes=[256]),
                    obs_key="pixels/frontal_ego/depth",
                ),
                joint_processing=ModuleConfigMLP(layer_sizes=[256], obs_key="goal_direction"),
            ),
            student=default_super.student,
            latent_encoding_size=default_super.latent_encoding_size,
        )

    @property
    def vision(self) -> bool:
        return super().vision or self.student.vision

    @classmethod
    def config_class_key(cls) -> str:
        return "TeacherStudent"

    @classmethod
    def get_model_class(cls) -> type["TeacherStudentNetworks"]:
        return TeacherStudentNetworks


register_model_config_class(TeacherStudentConfig)


@flax_dataclass
class TeacherStudentNetworkParams(ActorCriticEnrichedNetworkParams):
    """Contains training state for the learner."""

    student_encoder: Params


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
                acting_encoder=restore_params.network_params.acting_encoder,
                student_encoder=restore_params.network_params.student_encoder,
            ),
            preprocessor_params=restore_params.preprocessor_params,
        )


class TeacherStudentNetworks(
    ActorCriticEnrichedNetworks, ComponentNetworksArchitecture[TeacherStudentNetworkParams]
):
    def __init__(
        self,
        *,
        model_config: TeacherStudentConfig,
        observation_size: ObservationSize,
        action_size: int,
        vision_obs_period: int | None = None,
        preprocess_observations_fn: PreprocessObservationFn = identity_observation_preprocessor,
        activation: ActivationFn = linen.swish,
    ):
        """Make teacher-student network with preprocessor."""
        self.student_encoder_module = model_config.encoder.create(
            activation_fn=activation,
            activate_final=True,
            extra_final_layer_size=model_config.latent_encoding_size,
        )
        super().__init__(
            model_config=model_config,
            observation_size=observation_size,
            action_size=action_size,
            vision_obs_period=vision_obs_period,
            preprocess_observations_fn=preprocess_observations_fn,
            activation=activation,
        )

    @staticmethod
    def agent_params_class() -> type[TeacherStudentAgentParams]:
        return TeacherStudentAgentParams

    def initialize(self, rng: PRNGKey) -> TeacherStudentNetworkParams:
        policy_key, value_key, teacher_key, student_key = jax.random.split(rng, 4)
        return TeacherStudentNetworkParams(
            policy=self.policy_module.init(policy_key, self.dummy_obs, self.dummy_latent),
            value=self.value_module.init(value_key, self.dummy_obs, self.dummy_latent),
            acting_encoder=self.acting_encoder_module.init(teacher_key, self.dummy_obs),
            student_encoder=self.student_encoder_module.init(student_key, self.dummy_obs),
        )

    def apply_student_encoder(
        self,
        preprocessor_params: PreprocessorParams,
        network_params: TeacherStudentNetworkParams,
        observation: Observation,
        repeat_output: bool = True,
    ) -> jax.Array:
        observation = self.preprocess_obs(preprocessor_params, observation)
        latent_encoding = self.student_encoder_module.apply(
            network_params.student_encoder, observation
        )
        if repeat_output:
            latent_encoding = jnp.repeat(latent_encoding, self.vision_obs_period, axis=0)
        return latent_encoding

    def apply_student_policy(
        self,
        preprocessor_params: PreprocessorParams,
        network_params: TeacherStudentNetworkParams,
        observation: Observation,
    ) -> jax.Array:
        latent_encoding = self.apply_student_encoder(
            preprocessor_params, network_params, observation
        )
        return self.apply_policy_with_latents(
            preprocessor_params, network_params, observation, latent_encoding
        )

    def make_student_unroll_fn(
        self,
        agent_params: TeacherStudentAgentParams,
        *,
        deterministic: bool = False,
        accumulate_pipeline_states: bool = False,
    ) -> GenerateUnrollFn:
        if self.vision:
            policy_factory = self.policy_metafactory(self.apply_policy_with_latents)
            encoder_factory = self.apply_student_encoder
        else:
            policy_factory = self.policy_metafactory(self.apply_student_policy)
            encoder_factory = None
        return self.make_unroll_fn(
            agent_params=agent_params,
            policy_factory=policy_factory,
            apply_encoder_fn=encoder_factory,
            deterministic=deterministic,
            accumulate_pipeline_states=accumulate_pipeline_states,
        )

