from collections.abc import Mapping

import jax
from flax import linen
from flax.struct import dataclass as flax_dataclass

from quadruped_mjx_rl import types
from quadruped_mjx_rl.models import distributions
from quadruped_mjx_rl.models.architectures.raw_actor_critic import (
    ActorCriticNetworkParams,
    ActorCriticNetworks,
)
from quadruped_mjx_rl.models.configs import TeacherStudentConfig, TeacherStudentVisionConfig
from quadruped_mjx_rl.models.modules import ActivationFn, CNN, MLP
from quadruped_mjx_rl.models.networks import (
    AgentParams,
    ComponentNetworkArchitecture,
    FeedForwardNetwork,
    make_network,
    policy_factory,
)


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


@flax_dataclass
class TeacherStudentNetworks(
    ActorCriticNetworks, ComponentNetworkArchitecture[TeacherStudentNetworkParams]
):
    teacher_encoder_network: FeedForwardNetwork
    student_encoder_network: FeedForwardNetwork

    def agent_params_class(self):
        return TeacherStudentAgentParams

    def initialize(self, rng: types.PRNGKey) -> TeacherStudentNetworkParams:
        policy_key, value_key, teacher_key, student_key = jax.random.split(rng, 4)
        return TeacherStudentNetworkParams(
            policy=self.policy_network.init(policy_key),
            value=self.value_network.init(value_key),
            teacher_encoder=self.teacher_encoder_network.init(teacher_key),
            student_encoder=self.student_encoder_network.init(student_key),
        )

    def teacher_policy_apply(
        self, params: TeacherStudentAgentParams, observation: types.Observation
    ) -> jax.Array:
        latent_vector = self.teacher_encoder_network.apply(
            params.preprocessor_params, params.network_params.teacher_encoder, observation
        )
        return self.policy_network.apply(
            params.preprocessor_params, params.network_params.policy, observation, latent_vector
        )

    def student_policy_apply(
        self, params: TeacherStudentAgentParams, observation: types.Observation
    ):
        latent_vector = self.student_encoder_network.apply(
            params.preprocessor_params, params.network_params.student_encoder, observation
        )
        return self.policy_network.apply(
            params.preprocessor_params, params.network_params.policy, observation, latent_vector
        )

    def value_apply(
        self, params: TeacherStudentAgentParams, observation: types.Observation
    ) -> jax.Array:
        latent_vector = self.teacher_encoder_network.apply(
            params.preprocessor_params, params.network_params.teacher_encoder, observation
        )
        return self.value_network.apply(
            params.preprocessor_params, params.network_params.value, observation, latent_vector
        )

    def policy_apply(self, params: TeacherStudentAgentParams, observation: types.Observation):
        return self.teacher_policy_apply(params, observation)

    def policy_metafactory(self):
        """Creates params and inference function for the Teacher and Student agents."""

        def make_teacher_policy(params: TeacherStudentAgentParams, deterministic: bool = False):
            return policy_factory(
                policy_apply=self.teacher_policy_apply,
                parametric_action_distribution=self.parametric_action_distribution,
                params=params,
                deterministic=deterministic,
            )

        def make_student_policy(params: TeacherStudentAgentParams, deterministic: bool = False):
            return policy_factory(
                policy_apply=self.student_policy_apply,
                parametric_action_distribution=self.parametric_action_distribution,
                params=params,
                deterministic=deterministic,
            )

        return make_teacher_policy, make_student_policy


def make_teacher_student_networks(
    *,
    observation_size: types.ObservationSize,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = (
        types.identity_observation_preprocessor
    ),
    teacher_obs_key: str = "privileged_state",
    student_obs_key: str = "state_history",
    common_obs_key: str = "state",
    latent_obs_key: str = "latent",
    model_config: TeacherStudentConfig = TeacherStudentConfig(),
    activation: ActivationFn = linen.swish,
):
    """Make teacher-student network with preprocessor."""

    # Sanity check
    if not isinstance(observation_size, Mapping):
        raise TypeError(
            f"Environment observations must be a dictionary (Mapping),"
            f" got {type(observation_size)}"
        )
    if teacher_obs_key not in observation_size:
        raise ValueError(
            f"Teacher observation key {teacher_obs_key} must be in environment observations."
        )
    if student_obs_key not in observation_size:
        raise ValueError(
            f"Student observation key {student_obs_key} must be in environment observations."
        )
    if common_obs_key not in observation_size:
        raise ValueError(
            f"Common observation key {common_obs_key} must be in environment observations."
        )

    observation_size = dict(observation_size) | {latent_obs_key: (model_config.latent_size,)}

    parametric_action_distribution = distributions.NormalTanhDistribution(
        event_size=action_size
    )

    if isinstance(model_config, TeacherStudentVisionConfig):
        teacher_encoder_module = CNN(
            num_filters=model_config.modules.encoder_convolutional,
            kernel_sizes=[(3, 3)] * len(model_config.modules.encoder_convolutional),
            strides=[(1, 1)] * len(model_config.modules.encoder_convolutional),
            dense_layer_sizes=model_config.modules.encoder_dense + [model_config.latent_size],
            activation=activation,
            activate_final=True,
        )
        student_encoder_module = CNN(
            num_filters=model_config.modules.adapter_convolutional,
            kernel_sizes=[(3, 3)] * len(model_config.modules.adapter_convolutional),
            strides=[(1, 1)] * len(model_config.modules.adapter_convolutional),
            dense_layer_sizes=model_config.modules.adapter_dense + [model_config.latent_size],
            activation=activation,
            activate_final=True,
        )
        # Visual observations are not preprocessed
        teacher_preprocess_keys = ()
        student_preprocess_keys = ()
    elif isinstance(model_config, TeacherStudentConfig):
        teacher_encoder_module = MLP(
            layer_sizes=model_config.modules.encoder + [model_config.latent_size],
            activation=activation,
            activate_final=True,
        )
        student_encoder_module = MLP(
            layer_sizes=model_config.modules.adapter + [model_config.latent_size],
            activation=activation,
            activate_final=True,
        )
        teacher_preprocess_keys = (teacher_obs_key,)
        student_preprocess_keys = (student_obs_key,)
    else:
        raise TypeError("Model configuration must be a TeacherStudentConfig instance.")

    teacher_encoder_network = make_network(
        module=teacher_encoder_module,
        obs_size=observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        preprocess_obs_keys=teacher_preprocess_keys,
        apply_to_obs_keys=(teacher_obs_key,),
        squeeze_output=False,
    )

    student_encoder_network = make_network(
        module=student_encoder_module,
        obs_size=observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        preprocess_obs_keys=student_preprocess_keys,
        apply_to_obs_keys=(student_obs_key,),
        squeeze_output=False,
    )

    policy_module = MLP(
        layer_sizes=model_config.modules.policy + [parametric_action_distribution.param_size],
        activation=activation,
        activate_final=False,
    )

    value_module = MLP(
        layer_sizes=model_config.modules.value + [1],
        activation=activation,
        activate_final=False,
    )

    policy_network = make_network(
        module=policy_module,
        obs_size=observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        preprocess_obs_keys=(common_obs_key,),
        apply_to_obs_keys=(common_obs_key, latent_obs_key),
        squeeze_output=False,
    )

    value_network = make_network(
        module=value_module,
        obs_size=observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        preprocess_obs_keys=(common_obs_key,),
        apply_to_obs_keys=(common_obs_key, latent_obs_key),
        squeeze_output=True,
    )

    policy_raw_apply = policy_network.apply
    value_raw_apply = value_network.apply

    def policy_apply(
        processor_params: types.PreprocessorParams,
        params: types.Params,
        obs: dict[str, types.ndarray],
        latent_encoding: types.ndarray,
    ):
        obs = obs | {latent_obs_key: latent_encoding}
        return policy_raw_apply(processor_params=processor_params, params=params, obs=obs)

    def value_apply(
        processor_params: types.PreprocessorParams,
        params: types.Params,
        obs: dict[str, types.ndarray],
        latent_encoding: types.ndarray,
    ):
        obs = obs | {latent_obs_key: latent_encoding}
        return value_raw_apply(processor_params=processor_params, params=params, obs=obs)

    policy_network.apply = policy_apply
    value_network.apply = value_apply

    return TeacherStudentNetworks(
        teacher_encoder_network=teacher_encoder_network,
        student_encoder_network=student_encoder_network,
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )
