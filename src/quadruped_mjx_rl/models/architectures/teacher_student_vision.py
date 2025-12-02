from collections.abc import Mapping
from dataclasses import dataclass, field

import jax
from jax import numpy as jnp
from flax import linen

from quadruped_mjx_rl import types
from quadruped_mjx_rl.models import distributions
from quadruped_mjx_rl.models.networks_utils import FeedForwardNetwork
from quadruped_mjx_rl.models.architectures.actor_critic_base import (
    ActorCriticConfig,
    ActorCriticNetworkParams,
)
from quadruped_mjx_rl.models.architectures.configs_base import (
    register_model_config_class,
    ComponentNetworksArchitecture,
    ModuleConfigMLP,
)
from quadruped_mjx_rl.models.architectures.teacher_student_base import (
    TeacherStudentConfig,
    TeacherStudentAgent,
    TeacherStudentNetworkParams,
    TeacherStudentAgentParams, TeacherStudentNetworks,
)
from quadruped_mjx_rl.models.base_modules import ActivationFn, CNN, MLP
from quadruped_mjx_rl.models.networks_utils import make_network


@dataclass
class ModuleConfigCNN:
    filter_sizes: list[int]
    dense: ModuleConfigMLP


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
    def get_model_class(cls) -> type["TeacherStudentVisionAgent"]:
        return TeacherStudentVisionAgent


register_model_config_class(TeacherStudentVisionConfig)


class TeacherStudentVisionNetworks(ComponentNetworksArchitecture[TeacherStudentNetworkParams]):
    def __init__(
        self,
        *,
        model_config: TeacherStudentVisionConfig,
        observation_size: types.ObservationSize,
        output_size: int,
        preprocess_observations_fn: types.PreprocessObservationFn = (
            types.identity_observation_preprocessor
        ),
        activation: ActivationFn = linen.swish,
    ):
        """Make teacher-student network with preprocessor."""
        # Visual observations are not preprocessed
        teacher_preprocess_keys = ()
        student_preprocess_keys = ()
        teacher_encoder_module = CNN(
            num_filters=model_config.encoder.filter_sizes,
            dense_layer_sizes=model_config.encoder.dense.layer_sizes + [model_config.latent_encoding_size],
            activation=activation,
            activate_final=True,
        )
        self._teacher_encoder_network = make_network(
            module=teacher_encoder_module,
            obs_size=observation_size,
            preprocess_observations_fn=preprocess_observations_fn,
            preprocess_obs_keys=teacher_preprocess_keys,
            apply_to_obs_keys=(model_config.encoder_obs_key,),
            squeeze_output=False,
        )
        student_encoder_module = CNN(
            num_filters=model_config.student.filter_sizes,
            dense_layer_sizes=model_config.student.dense.layer_sizes + [model_config.latent_encoding_size],
            activation=activation,
            activate_final=True,
        )
        self._student_encoder_network = make_network(
            module=student_encoder_module,
            obs_size=observation_size,
            preprocess_observations_fn=preprocess_observations_fn,
            preprocess_obs_keys=student_preprocess_keys,
            apply_to_obs_keys=(model_config.student_obs_key,),
            squeeze_output=False,
        )

        self._latent_obs_key = model_config.latent_obs_key
        policy_module = MLP(
            layer_sizes=model_config.policy.layer_sizes + [output_size],
            activation=activation,
            activate_final=False,
        )
        self._policy_network = make_network(
            module=policy_module,
            obs_size=observation_size,
            preprocess_observations_fn=preprocess_observations_fn,
            preprocess_obs_keys=(model_config.common_obs_key,),
            apply_to_obs_keys=(model_config.common_obs_key, self._latent_obs_key),
            squeeze_output=False,
            concatenate_inputs=True,
        )
        value_module = MLP(
            layer_sizes=model_config.value.layer_sizes + [1],
            activation=activation,
            activate_final=False,
        )
        self._value_network = make_network(
            module=value_module,
            obs_size=observation_size,
            preprocess_observations_fn=preprocess_observations_fn,
            preprocess_obs_keys=(model_config.common_obs_key,),
            apply_to_obs_keys=(model_config.common_obs_key, self._latent_obs_key),
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


class TeacherStudentVisionAgent(TeacherStudentAgent):

    @staticmethod
    def networks_class() -> type[TeacherStudentVisionNetworks]:
        return TeacherStudentVisionNetworks
