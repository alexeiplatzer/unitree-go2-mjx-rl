# Typing
from collections.abc import Sequence
from quadruped_mjx_rl import types
from dataclasses import dataclass

# Math
from flax.struct import dataclass as flax_dataclass
from flax import linen
from quadruped_mjx_rl.models import modules, networks, distributions, configs


@flax_dataclass
class ActorCriticNetworks:
    policy_network: networks.FeedForwardNetwork
    value_network: networks.FeedForwardNetwork
    parametric_action_distribution: distributions.ParametricDistribution


@flax_dataclass
class ActorCriticNetworkParams:
    policy: types.Params
    value: types.Params


@flax_dataclass
class ActorCriticAgentParams(networks.AgentParams[ActorCriticNetworkParams]):
    """Full usable parameters for an actor critic architecture."""


@dataclass
class OptimizerConfig:
    learning_rate: float = 0.0004
    max_grad_norm: float | None = None


def make_inference_fn(actor_critic_networks: ActorCriticNetworks):
    """Creates params and inference function for the PPO agent."""

    def make_policy(
        params: ActorCriticAgentParams, deterministic: bool = False
    ) -> types.Policy:
        policy_network = actor_critic_networks.policy_network
        parametric_action_distribution = actor_critic_networks.parametric_action_distribution

        def policy(
            observations: types.Observation, key_sample: types.PRNGKey
        ) -> tuple[types.Action, types.Extra]:
            normalizer_params = params.preprocessor_params
            policy_params = params.network_params.policy
            logits = policy_network.apply(normalizer_params, policy_params, observations)
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


def make_actor_critic_networks(
    *,
    observation_size: types.ObservationSize,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = (
        types.identity_observation_preprocessor
    ),
    policy_obs_key: str = "state",
    value_obs_key: str = "state",
    model_config: configs.ActorCriticConfig = configs.ActorCriticConfig(),
    activation: modules.ActivationFn = linen.swish,
) -> ActorCriticNetworks:
    """Make Actor Critic networks with preprocessor."""
    parametric_action_distribution = distributions.NormalTanhDistribution(
        event_size=action_size
    )
    policy_module = modules.MLP(
        layer_sizes=(model_config.modules.policy + [parametric_action_distribution.param_size]),
        activation=activation,
        activate_final=False,
    )
    policy_network = networks.make_network(
        module=policy_module,
        obs_size=observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        preprocess_obs_keys=(policy_obs_key,),
        apply_to_obs_keys=(policy_obs_key,),
        squeeze_output=False,
    )
    value_module = modules.MLP(
        layer_sizes=model_config.modules.value + [1],
        activation=activation,
        activate_final=False,
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
