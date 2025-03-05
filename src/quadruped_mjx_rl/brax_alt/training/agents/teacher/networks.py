from typing import Sequence, Tuple

from brax.training import distribution
from ... import networks
from brax.training import types
from brax.training.types import PRNGKey
import flax
from flax import linen
from jax import numpy as jnp


@flax.struct.dataclass
class TeacherNetworks:
    encoder_network: networks.FeedForwardNetwork
    policy_network: networks.FeedForwardNetwork
    value_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution


@flax.struct.dataclass
class StudentNetworks:
    encoder_network: networks.FeedForwardNetwork


def make_teacher_inference_fn(teacher_networks: TeacherNetworks):
    """Creates params and inference function for the Teacher agent."""

    def make_policy(params: types.Params, deterministic: bool = False) -> types.Policy:
        encoder_network = teacher_networks.encoder_network
        policy_network = teacher_networks.policy_network
        parametric_action_distribution = teacher_networks.parametric_action_distribution

        def policy(
            observations: types.Observation, key_sample: PRNGKey
        ) -> Tuple[types.Action, types.Extra]:
            normalizer_params = params[0]
            encoder_params = params[1]
            policy_params = params[2]
            latent_vector = encoder_network.apply(
                normalizer_params, encoder_params, observations
            )
            logits = policy_network.apply(
                normalizer_params,
                policy_params,
                observations,
                latent_vector,
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
        teacher_params: types.Params,
        student_params: types.Params,
        deterministic: bool = False,
    ) -> types.Policy:
        encoder_network = student_networks.encoder_network
        policy_network = teacher_networks.policy_network
        parametric_action_distribution = teacher_networks.parametric_action_distribution

        def policy(
            observations: types.Observation, key_sample: PRNGKey
        ) -> Tuple[types.Action, types.Extra]:
            normalizer_params = teacher_params[0]
            encoder_params = student_params[0]
            policy_params = teacher_params[2]
            latent_vector = encoder_network.apply(
                normalizer_params,
                encoder_params,
                observations,
            )
            logits = policy_network.apply(
                normalizer_params,
                policy_params,
                observations,
                latent_vector,
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
    observation_size: int,
    privileged_observation_size: int,
    action_size: int,
    latent_representation_size: int = 32,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,
    value_hidden_layer_sizes: Sequence[int] = (128,) * 5,
    encoder_hidden_layer_sizes: Sequence[int] = (128,) * 2,
    activation: networks.ActivationFn = linen.swish,
    policy_obs_key: str = "state",
    value_obs_key: str = "state",
    encoder_obs_key: str = "privileged_state",
) -> TeacherNetworks:
    """Make Teacher networks with preprocessor."""
    parametric_action_distribution = distribution.NormalTanhDistribution(event_size=action_size)
    encoder_network = networks.make_encoder_network(
        latent_representation_size,
        privileged_observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=encoder_hidden_layer_sizes,
        activation=activation,
        obs_key=encoder_obs_key,
    )
    policy_network = networks.make_policy_network(
        parametric_action_distribution.param_size,
        observation_size + latent_representation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=policy_hidden_layer_sizes,
        activation=activation,
        obs_key=policy_obs_key,
    )
    value_network = networks.make_value_network(
        observation_size + latent_representation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=value_hidden_layer_sizes,
        activation=activation,
        obs_key=value_obs_key,
    )

    return TeacherNetworks(
        encoder_network=encoder_network,
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )


def make_student_networks(
    observation_size: int,
    # privileged_observation_size: int,
    # action_size: int,
    latent_representation_size: int = 32,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    # policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,
    # value_hidden_layer_sizes: Sequence[int] = (128,) * 5,
    adapter_hidden_layer_sizes: Sequence[int] = (128,) * 2,
    activation: networks.ActivationFn = linen.swish,
    # policy_obs_key: str = 'state',
    # value_obs_key: str = 'state',
    encoder_obs_key: str = "state_history",
) -> StudentNetworks:
    """Make Student networks with preprocessor."""
    # parametric_action_distribution = distribution.NormalTanhDistribution(
    #     event_size=action_size
    # )
    encoder_network = networks.make_encoder_network(
        latent_representation_size,
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=adapter_hidden_layer_sizes,
        activation=activation,
        obs_key=encoder_obs_key,
    )

    return StudentNetworks(
        encoder_network=encoder_network,
    )
