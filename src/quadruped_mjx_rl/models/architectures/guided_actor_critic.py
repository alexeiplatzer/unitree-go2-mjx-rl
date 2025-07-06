
# Typing
from collections.abc import Sequence, Mapping
from quadruped_mjx_rl import types

# Math
from flax.struct import dataclass as flax_dataclass
from flax import linen
from quadruped_mjx_rl.models import modules, networks, distributions


@flax_dataclass
class TeacherNetworks:
    encoder_network: networks.FeedForwardNetwork
    policy_network: networks.FeedForwardNetwork
    value_network: networks.FeedForwardNetwork
    parametric_action_distribution: distributions.ParametricDistribution


@flax_dataclass
class StudentNetworks:
    encoder_network: networks.FeedForwardNetwork


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
    activation: modules.ActivationFn = linen.swish,
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
    required_keys = {"state", "privileged_state", "state_history"}
    if not required_keys.issubset(observation_size.keys()):
        raise ValueError(
            f"Environment observation dict missing required keys. "
            f"Expected: {required_keys}, Got: {observation_size.keys()}"
        )

    observation_size |= {"latent": latent_representation_size}

    parametric_action_distribution = distributions.NormalTanhDistribution(event_size=action_size)
    if encoder_convolutional_layer_sizes is not None:
        encoder_module = modules.CNN(
            num_filters=list(encoder_convolutional_layer_sizes),
            kernel_sizes=[(3,)] * len(encoder_convolutional_layer_sizes),
            strides=[(1,)] * len(encoder_convolutional_layer_sizes),
            dense_layer_sizes=list(encoder_hidden_layer_sizes),
            activation=activation,
        )
    else:
        encoder_module = modules.MLP(
            layer_sizes=list(encoder_hidden_layer_sizes) + [latent_representation_size],
            activation=activation,
        )
    encoder_network = networks.make_network(
        module=encoder_module,
        obs_size=observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        obs_keys=encoder_obs_key,
    )
    policy_module = modules.MLP(
        layer_sizes=(
            list(policy_hidden_layer_sizes) + [parametric_action_distribution.param_size]
        ),
        activation=activation,
    )
    policy_network = networks.make_network(
        module=policy_module,
        obs_size=observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        obs_keys=(policy_obs_key, latent_obs_key),
    )
    value_module = modules.MLP(
        layer_sizes=list(value_hidden_layer_sizes) + [1],
        activation=activation,
    )
    value_network = networks.make_network(
        module=value_module,
        obs_size=observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        obs_keys=(value_obs_key, latent_obs_key),
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
    activation: modules.ActivationFn = linen.swish,
    encoder_obs_key: str = "state_history",
) -> StudentNetworks:
    """Make Student networks with preprocessor."""
    if adapter_convolutional_layer_sizes is not None:
        encoder_module = modules.CNN(
            num_filters=list(adapter_convolutional_layer_sizes),
            kernel_sizes=[(3,)] * len(adapter_convolutional_layer_sizes),
            strides=[(1,)] * len(adapter_convolutional_layer_sizes),
            dense_layer_sizes=list(adapter_hidden_layer_sizes),
            activation=activation,
        )
    else:
        encoder_module = modules.MLP(
            layer_sizes=list(adapter_hidden_layer_sizes) + [latent_representation_size],
            activation=activation,
        )
    encoder_network = networks.make_network(
        module=encoder_module,
        obs_size=observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        obs_keys=encoder_obs_key,
    )

    return StudentNetworks(
        encoder_network=encoder_network,
    )
