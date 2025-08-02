# Typing
from collections.abc import Sequence, Mapping
from quadruped_mjx_rl import types

# Supporting
import functools

# Math
from flax.struct import dataclass as flax_dataclass
from flax import linen
from quadruped_mjx_rl.models import distributions

# Definitions
from quadruped_mjx_rl.models.configs import TeacherStudentConfig, TeacherStudentVisionConfig
from quadruped_mjx_rl.models.modules import ActivationFn, MLP, HeadMLP, CNN
from quadruped_mjx_rl.models.networks import make_network, FeedForwardNetwork, AgentParams
from quadruped_mjx_rl.models.architectures.raw_actor_critic import (
    ActorCriticNetworks,
    ActorCriticNetworkParams,
)


@flax_dataclass
class TeacherStudentNetworks(ActorCriticNetworks):
    teacher_encoder_network: FeedForwardNetwork
    student_encoder_network: FeedForwardNetwork


@flax_dataclass
class TeacherStudentNetworkParams(ActorCriticNetworkParams):
    """Contains training state for the learner."""

    teacher_encoder: types.Params
    student_encoder: types.Params


@flax_dataclass
class TeacherStudentAgentParams(AgentParams[TeacherStudentNetworkParams]):
    """Contains training state for the full agent."""


def make_teacher_student_inference_fns(teacher_student_networks: TeacherStudentNetworks):
    """Creates params and inference function for the Teacher and Student agents."""

    def make_policies(
        params: TeacherStudentAgentParams, deterministic: bool = False
    ) -> tuple[types.Policy, types.Policy]:
        teacher_encoder_network = teacher_student_networks.teacher_encoder_network
        student_encoder_network = teacher_student_networks.student_encoder_network
        policy_network = teacher_student_networks.policy_network
        parametric_action_distribution = teacher_student_networks.parametric_action_distribution

        def teacher_policy(
            observations: types.Observation, key_sample: types.PRNGKey
        ) -> tuple[types.Action, types.Extra]:
            normalizer_params = params.preprocessor_params
            encoder_params = params.network_params.teacher_encoder
            policy_params = params.network_params.policy

            latent_vector = teacher_encoder_network.apply(
                normalizer_params, encoder_params, observations
            )
            logits = policy_network.apply(
                normalizer_params,
                policy_params,
                observations,
                latent_vector,
            )
            if deterministic:
                return teacher_student_networks.parametric_action_distribution.mode(logits), {}
            raw_actions = parametric_action_distribution.sample_no_postprocessing(
                logits, key_sample
            )
            log_prob = parametric_action_distribution.log_prob(logits, raw_actions)
            postprocessed_actions = parametric_action_distribution.postprocess(raw_actions)
            return postprocessed_actions, {
                "log_prob": log_prob,
                "raw_action": raw_actions,
            }

        def student_policy(
            observations: types.Observation, key_sample: types.PRNGKey
        ) -> tuple[types.Action, types.Extra]:
            normalizer_params = params.preprocessor_params
            encoder_params = params.network_params.student_encoder
            policy_params = params.network_params.policy

            latent_vector = student_encoder_network.apply(
                normalizer_params, encoder_params, observations
            )
            logits = policy_network.apply(
                normalizer_params,
                policy_params,
                observations,
                latent_vector,
            )
            if deterministic:
                return teacher_student_networks.parametric_action_distribution.mode(logits), {}
            raw_actions = parametric_action_distribution.sample_no_postprocessing(
                logits, key_sample
            )
            log_prob = parametric_action_distribution.log_prob(logits, raw_actions)
            postprocessed_actions = parametric_action_distribution.postprocess(raw_actions)
            return postprocessed_actions, {
                "log_prob": log_prob,
                "raw_action": raw_actions,
            }

        return teacher_policy, student_policy

    return make_policies


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

    observation_size = dict(observation_size) | {latent_obs_key: model_config.latent_size}

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

    policy_module = HeadMLP(
        layer_sizes=model_config.modules.policy + [parametric_action_distribution.param_size],
        activation=activation,
        activate_final=False,
    )

    value_module = HeadMLP(
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
