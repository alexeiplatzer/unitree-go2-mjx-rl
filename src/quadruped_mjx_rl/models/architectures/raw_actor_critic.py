
# Typing
from collections.abc import Sequence
from quadruped_mjx_rl import types

# Math
from flax.struct import dataclass as flax_dataclass
from flax import linen
from quadruped_mjx_rl.models import modules, networks, distributions


@flax_dataclass
class ActorCriticNetworks:
    policy_network: networks.FeedForwardNetwork
    value_network: networks.FeedForwardNetwork
    parametric_action_distribution: distributions.ParametricDistribution


@flax_dataclass
class ActorCriticNetworkParams:
    policy: types.Params
    value: types.Params


def make_inference_fn(actor_critic_networks: ActorCriticNetworks):
    """Creates params and inference function for the PPO agent."""

    def make_policy(params: types.Params, deterministic: bool = False) -> types.Policy:
        policy_network = actor_critic_networks.policy_network
        parametric_action_distribution = actor_critic_networks.parametric_action_distribution

        def policy(
            observations: types.Observation, key_sample: types.PRNGKey
        ) -> tuple[types.Action, types.Extra]:
            param_subset = (params[0], params[1])  # normalizer and policy params
            logits = policy_network.apply(*param_subset, observations)
            if deterministic:
                return actor_critic_networks.parametric_action_distribution.mode(logits), {}
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


def make_networks(
    observation_size: types.ObservationSize,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = (
        types.identity_observation_preprocessor
    ),
    policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,
    value_hidden_layer_sizes: Sequence[int] = (256,) * 5,
    activation: modules.ActivationFn = linen.swish,
    policy_obs_key: str = "state",
    value_obs_key: str = "state",
) -> ActorCriticNetworks:
    """Make Actor Critic networks with preprocessor."""
    parametric_action_distribution = distributions.NormalTanhDistribution(event_size=action_size)
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
        preprocess_obs_keys=(policy_obs_key,),
        apply_to_obs_keys=(policy_obs_key,),
    )
    value_module = modules.MLP(
        layer_sizes=list(value_hidden_layer_sizes) + [1],
        activation=activation,
    )
    value_network = networks.make_network(
        module=value_module,
        obs_size=observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        preprocess_obs_keys=(value_obs_key,),
        apply_to_obs_keys=(value_obs_key,),
        squeeze_output=True,
    )

    return ActorCriticNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )
