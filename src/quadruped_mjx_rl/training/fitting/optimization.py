import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Protocol

import jax
import optax
from flax.struct import dataclass as flax_dataclass
from jax import numpy as jnp

from quadruped_mjx_rl.models.networks import (
    AgentNetworkParams,
    AgentParams,
    ComponentNetworkArchitecture,
)
from quadruped_mjx_rl.running_statistics import RunningStatisticsState
from quadruped_mjx_rl.training import training_utils
from quadruped_mjx_rl.training.configs import HyperparamsPPO, OptimizerConfig
from quadruped_mjx_rl.training.gradients import gradient_update_fn
from quadruped_mjx_rl.types import Metrics, PreprocessorParams, PRNGKey, Transition


@flax_dataclass
class OptimizerState:
    optimizer_state: optax.OptState


def make_optimizer(learning_rate: float, max_grad_norm: float | None = None):
    opt = optax.adam(learning_rate=learning_rate)
    return (
        opt
        if max_grad_norm is None
        else optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            opt,
        )
    )


class LossFn(Protocol[AgentNetworkParams]):
    def __call__(
        self,
        network: ComponentNetworkArchitecture[AgentNetworkParams],
        network_params: AgentNetworkParams,
        preprocessor_params: PreprocessorParams,
        data: Transition,
        rng: PRNGKey,
        hyperparams: HyperparamsPPO,
    ) -> tuple[jnp.ndarray, Metrics]:
        pass


class Fitter(ABC, Generic[AgentNetworkParams]):
    @abstractmethod
    def __init__(
        self,
        optimizer_config: OptimizerConfig,
        network: ComponentNetworkArchitecture[AgentNetworkParams],
        main_loss_fn: LossFn[AgentNetworkParams],
        algorithm_hyperparams: HyperparamsPPO,
    ):
        pass

    @abstractmethod
    def optimizer_init(self, network_params: AgentNetworkParams) -> OptimizerState:
        pass

    @abstractmethod
    def minibatch_step(
        self,
        carry: tuple[OptimizerState, AgentParams[AgentNetworkParams], PRNGKey],
        data: Transition,
        normalizer_params: RunningStatisticsState,
    ) -> tuple[tuple[OptimizerState, AgentParams[AgentNetworkParams], PRNGKey], dict[str, ...]]:
        pass


class SimpleFitter(Fitter):
    def __init__(
        self,
        optimizer_config: OptimizerConfig,
        network,
        main_loss_fn,
        algorithm_hyperparams,
    ):
        self.optimizer = make_optimizer(
            optimizer_config.learning_rate, optimizer_config.max_grad_norm
        )
        loss_fn = functools.partial(
            main_loss_fn,
            network=network,
            hyperparams=algorithm_hyperparams,
        )
        self.gradient_update_fn = gradient_update_fn(
            loss_fn=loss_fn,
            optimizer=self.optimizer,
            pmap_axis_name=training_utils.PMAP_AXIS_NAME,
            has_aux=True,
        )

    def optimizer_init(self, network_params):
        return OptimizerState(optimizer_state=self.optimizer.init(network_params))

    def minibatch_step(self, carry, data, normalizer_params):
        optimizer_state, network_params, key = carry
        key, key_loss = jax.random.split(key)
        (loss, metrics), params, raw_optimizer_state = self.gradient_update_fn(
            network_params,
            normalizer_params,
            data,
            key_loss,
            optimizer_state=optimizer_state.optimizer_state,
        )
        optimizer_state = OptimizerState(optimizer_state=raw_optimizer_state)
        return (optimizer_state, params, key), metrics

