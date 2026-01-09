import functools
from typing import Protocol

from quadruped_mjx_rl.models.architectures import (
    ModelConfig,
)
from quadruped_mjx_rl.models.architectures.configs_base import ComponentNetworksArchitecture
from quadruped_mjx_rl.models.types import (
    AgentNetworkParams,
    identity_observation_preprocessor,
    PreprocessObservationFn,
)
from quadruped_mjx_rl.types import ObservationSize


class NetworkFactory(Protocol[AgentNetworkParams]):
    def __call__(
        self,
        observation_size: ObservationSize,
        action_size: int,
        vision_obs_period: int | None = None,
        preprocess_observations_fn: PreprocessObservationFn = identity_observation_preprocessor,
    ) -> ComponentNetworksArchitecture[AgentNetworkParams]:
        pass


def get_networks_factory(
    model_config: ModelConfig,
) -> NetworkFactory:
    """Checks the model type from the configuration and returns the appropriate factory."""
    model_class = type(model_config).get_model_class()
    return functools.partial(model_class, model_config=model_config)
