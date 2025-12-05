from dataclasses import dataclass, field

import jax
from flax import linen
from flax.struct import dataclass as flax_dataclass
from jax import numpy as jnp

from quadruped_mjx_rl.models import (
    AgentParams,
)
from quadruped_mjx_rl.models.architectures.actor_critic_enriched import (
    ActorCriticEnrichedConfig,
    ActorCriticEnrichedNetworkParams,
    ActorCriticEnrichedNetworks,
)
from quadruped_mjx_rl.models.architectures.configs_base import (
    ComponentNetworksArchitecture,
    register_model_config_class,
)
from quadruped_mjx_rl.models.base_modules import ActivationFn, ModuleConfigCNN, ModuleConfigMLP
from quadruped_mjx_rl.models.types import (
    identity_observation_preprocessor,
    Params,
    PreprocessObservationFn,
)
from quadruped_mjx_rl.types import Observation, ObservationSize, PRNGKey


@dataclass
class TeacherStudentConfig(ActorCriticEnrichedConfig):
    encoder_obs_key: str = "environment_privileged"
    student_obs_key: str = "proprioceptive_history"
    encoder: ModuleConfigMLP = field(
        default_factory=lambda: ModuleConfigMLP(layer_sizes=[256, 256])
    )
    student: ModuleConfigMLP = field(
        default_factory=lambda: ModuleConfigMLP(layer_sizes=[256, 256])
    )

    @classmethod
    def config_class_key(cls) -> str:
        return "TeacherStudent"

    @classmethod
    def get_model_class(cls) -> type["TeacherStudentNetworks"]:
        return TeacherStudentNetworks


@dataclass
class TeacherStudentVisionConfig(TeacherStudentConfig):

    encoder_obs_key: str = "pixels/terrain/depth"
    student_obs_key: str = "pixels/frontal_ego/rgb_adjusted"
    encoder: ModuleConfigCNN = field(
        default_factory=lambda: ModuleConfigCNN(
            filter_sizes=[32, 64, 64], dense=ModuleConfigMLP(layer_sizes=[256, 256])
        )
    )
    student: ModuleConfigCNN = field(
        default_factory=lambda: ModuleConfigCNN(
            filter_sizes=[32, 64, 64], dense=ModuleConfigMLP(layer_sizes=[256, 256])
        )
    )

    @classmethod
    def config_class_key(cls) -> str:
        return "TeacherStudentVision"

    @classmethod
    def get_model_class(cls) -> type["TeacherStudentNetworks"]:
        return TeacherStudentNetworks


register_model_config_class(TeacherStudentConfig)
register_model_config_class(TeacherStudentVisionConfig)


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
        self.student_encoder_obs_key = model_config.student_obs_key
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
                student_key, self.dummy_obs[self.student_encoder_obs_key]
            ),
        )

    def apply_student_encoder(
        self, params: TeacherStudentAgentParams, observation: Observation
    ) -> jax.Array:
        observation = self.preprocess_obs(params.preprocessor_params, observation)
        return self.student_encoder_module.apply(
            params.network_params.student_encoder, observation[self.student_encoder_obs_key]
        )
