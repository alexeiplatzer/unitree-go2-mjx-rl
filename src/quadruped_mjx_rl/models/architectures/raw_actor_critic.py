import jax
from flax import linen
from flax.struct import dataclass as flax_dataclass

from quadruped_mjx_rl import types
from quadruped_mjx_rl.models import configs, distributions, modules, networks
from quadruped_mjx_rl.models.networks import ComponentNetworkArchitecture
from quadruped_mjx_rl.types import Observation, PRNGKey


@flax_dataclass
class ActorCriticNetworkParams:
    policy: types.Params
    value: types.Params


@flax_dataclass
class ActorCriticAgentParams(networks.AgentParams[ActorCriticNetworkParams]):
    """Full usable parameters for an actor critic architecture."""

    def restore_params(
        self,
        restore_params: "ActorCriticAgentParams",
        restore_value: bool = False,
    ):
        value_params = (
            restore_params.network_params.value if restore_value else self.network_params.value
        )
        return self.replace(
            network_params=ActorCriticNetworkParams(
                policy=restore_params.network_params.policy,
                value=value_params,
            ),
            preprocessor_params=restore_params.preprocessor_params,
        )


@flax_dataclass
class ActorCriticNetworks(ComponentNetworkArchitecture[ActorCriticNetworkParams]):
    policy_network: networks.FeedForwardNetwork
    value_network: networks.FeedForwardNetwork
    parametric_action_distribution: distributions.ParametricDistribution

    def agent_params_class(self):
        return ActorCriticAgentParams

    def initialize(self, rng: PRNGKey) -> ActorCriticNetworkParams:
        policy_key, value_key = jax.random.split(rng)
        return ActorCriticNetworkParams(
            policy=self.policy_network.init(policy_key),
            value=self.value_network.init(value_key),
        )

    def policy_apply(
        self, params: ActorCriticAgentParams, observation: Observation
    ) -> jax.Array:
        return self.policy_network.apply(
            params.preprocessor_params, params.network_params.policy, observation
        )

    def value_apply(
        self, params: ActorCriticAgentParams, observation: Observation
    ) -> jax.Array:
        return self.value_network.apply(
            params.preprocessor_params, params.network_params.value, observation
        )

    def policy_metafactory(self):
        """Creates params and inference function for the PPO agent."""

        def make_policy(
            params: ActorCriticAgentParams, deterministic: bool = False
        ) -> types.Policy:
            return networks.policy_factory(
                policy_apply=self.policy_apply,
                parametric_action_distribution=self.parametric_action_distribution,
                params=params,
                deterministic=deterministic,
            )

        return (make_policy,)


def make_actor_critic_networks(
    *,
    observation_size: types.ObservationSize,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = (
        types.identity_observation_preprocessor
    ),
    policy_obs_key: str = "proprioceptive",
    value_obs_key: str = "proprioceptive",
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
