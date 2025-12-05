import functools
from abc import ABC, abstractmethod
from typing import Generic, Protocol, Any
from collections.abc import Callable
import logging

import jax
import optax
from flax.struct import dataclass as flax_dataclass
from jax import numpy as jnp

from quadruped_mjx_rl.models import AgentParams
from quadruped_mjx_rl.models.architectures.configs_base import ComponentNetworksArchitecture
from quadruped_mjx_rl.running_statistics import RunningStatisticsState
from quadruped_mjx_rl.training import training_utils
from quadruped_mjx_rl.training.configs import OptimizerConfig
from quadruped_mjx_rl.training.algorithms.ppo import HyperparamsPPO
from quadruped_mjx_rl.training.gradients import gradient_update_fn
from quadruped_mjx_rl.training.evaluator import Evaluator
from quadruped_mjx_rl.types import Metrics, PRNGKey, Transition
from quadruped_mjx_rl.models.types import AgentNetworkParams, PolicyFactory, PreprocessorParams


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
        network_params: AgentNetworkParams,
        preprocessor_params: PreprocessorParams,
        data: Transition,
        rng: PRNGKey,
        network: ComponentNetworksArchitecture[AgentNetworkParams],
        hyperparams: HyperparamsPPO,
    ) -> tuple[jax.Array, Metrics]:
        pass


class EvalFn(Protocol[AgentNetworkParams]):
    def __call__(
        self,
        current_step: int,
        params: AgentParams[AgentNetworkParams],
        training_metrics: Metrics,
    ) -> None:
        pass


class Fitter(ABC, Generic[AgentNetworkParams]):
    @abstractmethod
    @property
    def network(self) -> ComponentNetworksArchitecture[AgentNetworkParams]:
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
    ) -> tuple[tuple[OptimizerState, AgentParams[AgentNetworkParams], PRNGKey], dict[str, Any]]:
        pass

    @abstractmethod
    def make_evaluation_fn(
        self,
        rng: PRNGKey,
        evaluator_factory: Callable[[PRNGKey, PolicyFactory], Evaluator],
        progress_fn_factory: Callable[[], tuple[Callable[[int, Metrics], None], list[float]]],
        deterministic_eval: bool = True,
    ) -> tuple[EvalFn[AgentNetworkParams], list[float]]:
        pass


class SimpleFitter(Fitter[AgentNetworkParams]):
    def __init__(
        self,
        optimizer_config: OptimizerConfig,
        network: ComponentNetworksArchitecture[AgentNetworkParams],
        main_loss_fn: LossFn[AgentNetworkParams],
        algorithm_hyperparams: HyperparamsPPO,
    ):
        self._network = network
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

    @property
    def network(self) -> ComponentNetworksArchitecture[AgentNetworkParams]:
        return self._network

    def optimizer_init(self, network_params: AgentNetworkParams) -> OptimizerState:
        return OptimizerState(optimizer_state=self.optimizer.init(network_params))

    def minibatch_step(
        self,
        carry: tuple[OptimizerState, AgentParams[AgentNetworkParams], PRNGKey],
        data: Transition,
        normalizer_params: RunningStatisticsState,
    ) -> tuple[tuple[OptimizerState, AgentParams[AgentNetworkParams], PRNGKey], dict[str, Any]]:
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

    def make_evaluation_fn(
        self,
        rng: PRNGKey,
        evaluator_factory: Callable[[PRNGKey, PolicyFactory], Evaluator],
        progress_fn_factory: Callable[[], tuple[Callable[[int, Metrics], None], list[float]]],
        deterministic_eval: bool = True,
    ) -> tuple[EvalFn[AgentNetworkParams], list[float]]:
        main_policy_factory = functools.partial(
            self.network.get_acting_policy_factory(), deterministic=deterministic_eval
        )
        simple_evaluator = evaluator_factory(rng, main_policy_factory)
        progress_fn, times = progress_fn_factory()

        def evaluation_fn(
            current_step: int,
            params: AgentParams[AgentNetworkParams],
            training_metrics: Metrics,
        ) -> None:
            evaluator_metrics = simple_evaluator.run_evaluation(
                params, training_metrics=training_metrics
            )
            logging.info("eval/episode_reward: %s" % evaluator_metrics["eval/episode_reward"])
            logging.info(
                "eval/episode_reward_std: %s" % evaluator_metrics["eval/episode_reward_std"]
            )
            logging.info("current_step: %s" % current_step)
            progress_fn(current_step, evaluator_metrics)

        return evaluation_fn, times
