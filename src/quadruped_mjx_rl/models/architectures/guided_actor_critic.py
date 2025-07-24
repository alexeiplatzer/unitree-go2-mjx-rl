
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
from quadruped_mjx_rl.models.networks import make_network, FeedForwardNetwork, normalizer_select


@flax_dataclass
class TeacherNetworks:
    encoder_network: FeedForwardNetwork
    policy_network: FeedForwardNetwork
    value_network: FeedForwardNetwork
    parametric_action_distribution: distributions.ParametricDistribution


@flax_dataclass
class StudentNetworks:
    encoder_network: FeedForwardNetwork


@flax_dataclass
class TeacherNetworkParams:
    """Contains training state for the learner."""

    encoder: types.Params
    policy: types.Params
    value: types.Params


@flax_dataclass
class StudentNetworkParams:
    encoder: types.Params


def make_teacher_inference_fn(teacher_networks: TeacherNetworks):
    """Creates params and inference function for the Teacher agent."""

    def make_policy(params: types.Params, deterministic: bool = False) -> types.Policy:
        encoder_network = teacher_networks.encoder_network
        policy_network = teacher_networks.policy_network
        parametric_action_distribution = teacher_networks.parametric_action_distribution

        def policy(
            observations: types.Observation, key_sample: types.PRNGKey
        ) -> tuple[types.Action, types.Extra]:
            normalizer_params = params[0]
            encoder_params = params[1]
            policy_params = params[2]
            latent_vector = encoder_network.apply(
                normalizer_params, encoder_params, observations
            )
            logits = policy_network.apply(
                normalizer_params,
                policy_params,
                observations | {"latent": latent_vector},
            )
            if deterministic:
                return teacher_networks.parametric_action_distribution.mode(logits), {}
            raw_actions = parametric_action_distribution.sample_no_postprocessing(
                logits, key_sample
            )
            log_prob = parametric_action_distribution.log_prob(logits, raw_actions)
            postprocessed_actions = parametric_action_distribution.postprocess(raw_actions)
            return postprocessed_actions, {
                "log_prob": log_prob,
                "raw_action": raw_actions,
            }

        return policy

    return make_policy


def make_student_inference_fn(
    teacher_networks: TeacherNetworks, student_networks: StudentNetworks
):
    """Creates params and inference function for the Student agent."""

    def make_policy(
        teacher_student_params: types.Params,
        deterministic: bool = False,
    ) -> types.Policy:
        encoder_network = student_networks.encoder_network
        policy_network = teacher_networks.policy_network
        parametric_action_distribution = teacher_networks.parametric_action_distribution

        def policy(
            observations: types.Observation, key_sample: types.PRNGKey
        ) -> tuple[types.Action, types.Extra]:
            # TODO this indices might be false? maybe refactoring makes sense to avoid them
            normalizer_params = teacher_student_params[0]
            encoder_params = teacher_student_params[1]
            policy_params = teacher_student_params[2]
            latent_vector = encoder_network.apply(
                normalizer_params,
                encoder_params,
                observations,
            )
            logits = policy_network.apply(
                normalizer_params,
                policy_params,
                observations | {"latent": latent_vector},
            )
            if deterministic:
                return teacher_networks.parametric_action_distribution.mode(logits), {}
            raw_actions = parametric_action_distribution.sample_no_postprocessing(
                logits, key_sample
            )
            log_prob = parametric_action_distribution.log_prob(logits, raw_actions)
            postprocessed_actions = parametric_action_distribution.postprocess(raw_actions)
            return postprocessed_actions, {
                "log_prob": log_prob,
                "raw_action": raw_actions,
            }

        return policy

    return make_policy


def make_teacher_networks(
    observation_size: types.ObservationSize,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = (
        types.identity_observation_preprocessor
    ),
    policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,
    value_hidden_layer_sizes: Sequence[int] = (128,) * 5,
    encoder_convolutional_layer_sizes: Sequence[int] | None = None,
    encoder_hidden_layer_sizes: Sequence[int] = (128,) * 2,
    latent_representation_size: int = 32,
    activation: ActivationFn = linen.swish,
    policy_obs_key: str = "state",
    value_obs_key: str = "state",
    encoder_obs_key: str = "privileged_state",
    latent_obs_key: str = "latent",
) -> TeacherNetworks:
    """Make Teacher networks with preprocessor."""

    if not isinstance(observation_size, Mapping):
        raise TypeError(
            f"Environment observations must be a dictionary (Mapping),"
            f" got {type(observation_size)}"
        )
    # required_keys = {"state", "privileged_state", "state_history"}
    # if not required_keys.issubset(observation_size.keys()):
    #     raise ValueError(
    #         f"Environment observation dict missing required keys. "
    #         f"Expected: {required_keys}, Got: {observation_size.keys()}"
    #     )

    observation_size |= {"latent": latent_representation_size}

    parametric_action_distribution = distributions.NormalTanhDistribution(
        event_size=action_size
    )

    if encoder_convolutional_layer_sizes is not None:
        encoder_module = CNN(
            num_filters=list(encoder_convolutional_layer_sizes),
            kernel_sizes=[(3, 3)] * len(encoder_convolutional_layer_sizes),
            strides=[(1, 1)] * len(encoder_convolutional_layer_sizes),
            dense_layer_sizes=list(encoder_hidden_layer_sizes) + [latent_representation_size],
            activation=activation,
            activate_final=True,
        )
        encoder_preprocess_keys = ()
    else:
        encoder_module = MLP(
            layer_sizes=list(encoder_hidden_layer_sizes) + [latent_representation_size],
            activation=activation,
            activate_final=True,
        )
        encoder_preprocess_keys = (encoder_obs_key,)
    encoder_network = make_network(
        module=encoder_module,
        obs_size=observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        preprocess_obs_keys=encoder_preprocess_keys,
        apply_to_obs_keys=(encoder_obs_key,),
        squeeze_output=False,
    )

    policy_module = MLP(
        layer_sizes=(
            list(policy_hidden_layer_sizes) + [parametric_action_distribution.param_size]
        ),
        activation=activation,
        activate_final=False,
    )
    policy_network = make_network(
        module=policy_module,
        obs_size=observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        preprocess_obs_keys=(policy_obs_key,),
        apply_to_obs_keys=(policy_obs_key, latent_obs_key),
        squeeze_output=False,
    )

    value_module = MLP(
        layer_sizes=list(value_hidden_layer_sizes) + [1],
        activation=activation,
        activate_final=False,
    )
    value_network = make_network(
        module=value_module,
        obs_size=observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        preprocess_obs_keys=(value_obs_key,),
        apply_to_obs_keys=(value_obs_key, latent_obs_key),
        squeeze_output=True,
    )

    return TeacherNetworks(
        encoder_network=encoder_network,
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )


def make_student_networks(
    observation_size: types.ObservationSize,
    latent_representation_size: int = 32,
    preprocess_observations_fn: types.PreprocessObservationFn = (
        types.identity_observation_preprocessor
    ),
    adapter_convolutional_layer_sizes: Sequence[int] | None = None,
    adapter_hidden_layer_sizes: Sequence[int] = (128,) * 2,
    activation: ActivationFn = linen.swish,
    encoder_obs_key: str = "state_history",
) -> StudentNetworks:
    """Make Student networks with preprocessor."""
    if adapter_convolutional_layer_sizes is not None:
        encoder_module = CNN(
            num_filters=list(adapter_convolutional_layer_sizes),
            kernel_sizes=[(3, 3)] * len(adapter_convolutional_layer_sizes),
            strides=[(1, 1)] * len(adapter_convolutional_layer_sizes),
            dense_layer_sizes=list(adapter_hidden_layer_sizes) + [latent_representation_size],
            activation=activation,
            activate_final=True,
        )
        preprocess_obs_keys = ()
    else:
        encoder_module = MLP(
            layer_sizes=list(adapter_hidden_layer_sizes) + [latent_representation_size],
            activation=activation,
            activate_final=True,
        )
        preprocess_obs_keys = (encoder_obs_key,)
    encoder_network = make_network(
        module=encoder_module,
        obs_size=observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        preprocess_obs_keys=(),
        apply_to_obs_keys=preprocess_obs_keys,
        squeeze_output=False,
    )

    return StudentNetworks(
        encoder_network=encoder_network,
    )


# TODO: potential alternative refactoring
def _make_teacher_student_network(
    observation_size: types.ObservationSize,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = (
        types.identity_observation_preprocessor
    ),
    teacher_obs_key: str = "privileged_state",
    student_obs_key: str = "state_history",
    common_obs_key: str = "state",
    model_config: TeacherStudentConfig = TeacherStudentConfig(),
    activation: ActivationFn = linen.swish,
):
    """Make teacher student network with preprocessor."""

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

    parametric_action_distribution = distributions.NormalTanhDistribution(
        event_size=action_size
    )

    def mapped_preprocess(
        obs: types.Observation, processor_params: types.PreprocessorParams, obs_key: str
    ) -> types.Observation:
        return preprocess_observations_fn(
            obs[obs_key], normalizer_select(processor_params, obs_key)
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
        teacher_preprocess = lambda obs, p: obs[teacher_obs_key]
        student_preprocess = lambda obs, p: obs[student_obs_key]
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
        teacher_preprocess = functools.partial(mapped_preprocess, obs_key=teacher_obs_key)
        student_preprocess = functools.partial(mapped_preprocess, obs_key=student_obs_key)
    else:
        raise TypeError("Model configuration must be a TeacherStudentConfig instance.")

    teacher_encoder_network = make_network(
        module=teacher_encoder_module,
        obs_size=observation_size[teacher_obs_key],
        preprocess_observations_fn=teacher_preprocess,
    )

    student_encoder_network = make_network(
        module=student_encoder_module,
        obs_size=observation_size[student_obs_key],
        preprocess_observations_fn=student_preprocess,
    )

    def head_preprocessor(obs, p):
        return {**obs, common_obs_key: mapped_preprocess(obs, p, common_obs_key)}

    policy_module = HeadMLP(
        layer_sizes=model_config.modules.policy + [parametric_action_distribution.param_size],
        activation=activation,
        obs_keys=[common_obs_key, "latent"],
    )

    value_module = HeadMLP(
        layer_sizes=model_config.modules.value + [1],
        activation=activation,
        obs_keys=[common_obs_key, "latent"],
    )

    policy_network = make_network(
        module=policy_module,
        obs_size=observation_size,
        preprocess_observations_fn=head_preprocessor,  # TODO: maybe glue them here, simplify upstream
    )

    value_network = make_network(
        module=value_module,
        obs_size=observation_size[common_obs_key] + model_config.latent_size,
        preprocess_observations_fn=head_preprocessor,
    )

    return None
