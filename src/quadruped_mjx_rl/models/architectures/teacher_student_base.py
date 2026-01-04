from dataclasses import dataclass, field

import jax
from flax import linen
from flax.struct import dataclass as flax_dataclass
from jax import numpy as jnp

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
    ActivationFn, MixedModeCNN, ModuleConfigCNN,
    ModuleConfigMixedModeCNN, ModuleConfigMLP,
)
from quadruped_mjx_rl.models.types import (
    identity_observation_preprocessor,
    Params,
    AgentParams,
    PolicyFactory,
    PreprocessObservationFn,
    PreprocessorParams,
)
from quadruped_mjx_rl.types import Observation, ObservationSize, PRNGKey


@dataclass
class TeacherStudentConfig(ActorCriticEnrichedConfig):
    encoder: ModuleConfigMLP
    student: ModuleConfigMLP

    @classmethod
    def default(cls) -> "TeacherStudentConfig":
        default_super = ActorCriticEnrichedConfig.default()
        return TeacherStudentConfig(
            policy=default_super.policy,
            value=default_super.value,
            encoder=ModuleConfigMLP(layer_sizes=[128, 256], obs_key="environment_privileged"),
            student=ModuleConfigMLP(layer_sizes=[512, 256], obs_key="proprioceptive_history"),
            latent_encoding_size=default_super.latent_encoding_size,
            encoder_supersteps=default_super.encoder_supersteps,
        )

    @classmethod
    def config_class_key(cls) -> str:
        return "TeacherStudent"

    @classmethod
    def get_model_class(cls) -> type["TeacherStudentNetworks"]:
        return TeacherStudentNetworks


@dataclass
class TeacherStudentVisionConfig(TeacherStudentConfig):
    encoder: ModuleConfigCNN
    student: ModuleConfigCNN

    @classmethod
    def default(cls) -> "TeacherStudentVisionConfig":
        default_super = TeacherStudentConfig.default()
        return TeacherStudentVisionConfig(
            policy=default_super.policy,
            value=default_super.value,
            encoder=ModuleConfigCNN(
                filter_sizes=[16, 32, 64],
                dense=ModuleConfigMLP(layer_sizes=[256]),
                obs_key="pixels/terrain/depth",
            ),
            student=ModuleConfigCNN(
                filter_sizes=[16, 32, 64],
                dense=ModuleConfigMLP(layer_sizes=[256]),
                obs_key="pixels/frontal_ego/rgb_adjusted",
            ),
            latent_encoding_size=default_super.latent_encoding_size,
            encoder_supersteps=default_super.encoder_supersteps,
        )

    @classmethod
    def config_class_key(cls) -> str:
        return "TeacherStudentVision"

    @classmethod
    def get_model_class(cls) -> type["TeacherStudentNetworks"]:
        return TeacherStudentNetworks


@dataclass
class TeacherStudentMixedModeConfig(TeacherStudentVisionConfig):
    encoder: ModuleConfigMixedModeCNN

    @classmethod
    def default(cls) -> "TeacherStudentMixedModeConfig":
        default_super = TeacherStudentVisionConfig.default()
        return TeacherStudentMixedModeConfig(
            policy=default_super.policy,
            value=default_super.value,
            encoder=ModuleConfigMixedModeCNN(
                vision_preprocessing=ModuleConfigCNN(
                    filter_sizes=[16, 32, 32],
                    dense=ModuleConfigMLP(layer_sizes=[256]),
                    obs_key="pixels/terrain/depth",
                ),
                joint_processing=ModuleConfigMLP(layer_sizes=[256], obs_key="goalwards_xy"),
            ),
            student=default_super.student,
            latent_encoding_size=default_super.latent_encoding_size,
            encoder_supersteps=default_super.encoder_supersteps,
        )

    @classmethod
    def config_class_key(cls) -> str:
        return "TeacherStudentMixedMode"

    @classmethod
    def get_model_class(cls) -> type["TeacherStudentNetworks"]:
        return TeacherStudentNetworks


register_model_config_class(TeacherStudentConfig)
register_model_config_class(TeacherStudentVisionConfig)
register_model_config_class(TeacherStudentMixedModeConfig)


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
            latent_encoding = jnp.repeat(latent_encoding, self.encoder_supersteps, axis=0)
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

    def get_student_policy_factory(self) -> PolicyFactory:
        return self.policy_metafactory(self.apply_student_policy)
