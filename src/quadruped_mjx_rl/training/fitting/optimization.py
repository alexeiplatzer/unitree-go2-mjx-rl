import functools
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar

import jax
import optax
from flax.struct import dataclass as flax_dataclass

from quadruped_mjx_rl.models.acting import GenerateUnrollFn
from quadruped_mjx_rl.physics_pipeline import Env
from quadruped_mjx_rl.models import AgentParams
from quadruped_mjx_rl.models.architectures.configs_base import ComponentNetworksArchitecture
from quadruped_mjx_rl.models.types import AgentNetworkParams, PreprocessorParams
from quadruped_mjx_rl.running_statistics import RunningStatisticsState
from quadruped_mjx_rl.training import training_utils
from quadruped_mjx_rl.training.algorithms.ppo import HyperparamsPPO
from quadruped_mjx_rl.training.configs import (
    OptimizerConfig,
    TrainingConfig,
)
from quadruped_mjx_rl.training.progress_plotting import make_progress_fn
from quadruped_mjx_rl.training.evaluator import Evaluator
from quadruped_mjx_rl.training.gradients import gradient_update_fn
from quadruped_mjx_rl.types import Metrics, PRNGKey, Transition


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


NetworkArchitecture = TypeVar("NetworkArchitecture", bound=ComponentNetworksArchitecture)


class LossFn(Protocol[AgentNetworkParams, NetworkArchitecture]):
    def __call__(
        self,
        network_params: AgentNetworkParams,
        preprocessor_params: PreprocessorParams,
        data: Transition,
        rng: PRNGKey,
        network: NetworkArchitecture,
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
    def __init__(
        self,
        optimizer_config: OptimizerConfig,
        network: ComponentNetworksArchitecture[AgentNetworkParams],
        main_loss_fn: LossFn[
            AgentNetworkParams, ComponentNetworksArchitecture[AgentNetworkParams]
        ],
        algorithm_hyperparams: HyperparamsPPO,
    ):
        pass

    @property
    @abstractmethod
    def network(self) -> ComponentNetworksArchitecture[AgentNetworkParams]:
        pass

    @abstractmethod
    def optimizer_init(self, network_params: AgentNetworkParams) -> OptimizerState:
        pass

    @abstractmethod
    def minibatch_step(
        self,
        carry: tuple[OptimizerState, AgentNetworkParams, PRNGKey],
        data: Transition,
        normalizer_params: RunningStatisticsState,
    ) -> tuple[tuple[OptimizerState, AgentNetworkParams, PRNGKey], dict[str, Any]]:
        pass

    @abstractmethod
    def make_evaluation_fn(
        self,
        eval_env: Env,
        eval_key: PRNGKey,
        training_config: TrainingConfig,
        show_outputs: bool = False,
        run_in_cell: bool = True,
        save_plots_path: Path | None = None,
    ) -> tuple[EvalFn[AgentNetworkParams], list[float]]:
        pass


class SimpleFitter(Fitter[AgentNetworkParams]):
    def __init__(
        self,
        optimizer_config: OptimizerConfig,
        network: ComponentNetworksArchitecture[AgentNetworkParams],
        main_loss_fn: LossFn[
            AgentNetworkParams, ComponentNetworksArchitecture[AgentNetworkParams]
        ],
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
        carry: tuple[OptimizerState, AgentNetworkParams, PRNGKey],
        data: Transition,
        normalizer_params: RunningStatisticsState,
    ) -> tuple[tuple[OptimizerState, AgentNetworkParams, PRNGKey], dict[str, Any]]:
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
        eval_env: Env,
        eval_key: PRNGKey,
        training_config: TrainingConfig,
        show_outputs: bool = False,
        run_in_cell: bool = True,
        save_plots_path: Path | None = None,
    ) -> tuple[EvalFn[AgentNetworkParams], list[float]]:
        acting_policy_eval_fn, times = self._evaluation_factory(
            eval_env=eval_env,
            eval_key=eval_key,
            training_config=training_config,
            make_unroll_fn=functools.partial(
                self.network.make_acting_unroll_fn,
                deterministic=training_config.deterministic_eval,
            ),
            show_outputs=show_outputs,
            run_in_cell=run_in_cell,
            save_plots_path=save_plots_path,
        )

        def evaluation_fn(
            current_step: int,
            params: AgentParams[AgentNetworkParams],
            training_metrics: Metrics,
        ) -> None:
            logging.info(f"current_step: {current_step}")
            acting_policy_eval_fn(current_step, params, training_metrics)

        return evaluation_fn, times

    @staticmethod
    def _evaluation_factory(
        eval_env: Env,
        eval_key: PRNGKey,
        training_config: TrainingConfig,
        make_unroll_fn: Callable[[AgentParams[AgentNetworkParams]], GenerateUnrollFn],
        show_outputs: bool = False,
        run_in_cell: bool = True,
        save_plots_path: Path | None = None,
        name: str = "",
        color: str = "blue",
    ) -> tuple[EvalFn[AgentNetworkParams], list[float]]:
        """Creates the evaluation function, given an unroll function for an agent
        and the relevant evaluation configs"""
        evaluator = Evaluator(
            eval_env=eval_env,
            key=eval_key,
            num_eval_envs=training_config.num_eval_envs,
            episode_length=training_config.episode_length,
            action_repeat=training_config.action_repeat,
            unroll_factory=make_unroll_fn,
        )

        reward_plots_path = save_plots_path / f"{name}_evaluation_results" if save_plots_path else None
        name = name + " " if name else ""
        progress_fn, times = make_progress_fn(
            show_outputs=show_outputs,
            run_in_cell=run_in_cell,
            save_plots_path=reward_plots_path,
            num_timesteps=training_config.num_timesteps,
            title=name + "Evaluation results",
            color=color,
            label_key="episode reward",
            data_key="eval/episode_reward",
            data_err_key="eval/episode_reward_std",
            data_max=60 * training_config.episode_length // 1024,  # TODO: dynamically adjust to expected reward
            data_min=-10,
        )
        energy_plots_path = save_plots_path / f"{name}_energy_evaluation" if save_plots_path else None
        energy_progress_fn, _ = make_progress_fn(
            show_outputs=show_outputs,
            run_in_cell=run_in_cell,
            save_plots_path=energy_plots_path,
            num_timesteps=training_config.num_timesteps,
            title=name + "Energy per distance",
            color="orange",
            label_key="energy per meter",
            data_key="eval/episode_energy_per_meter",
            data_err_key="eval/episode_energy_per_meter_std",
            data_max=2000,
            data_min=0,
        )

        def evaluation_fn(
            current_step: int,
            params: AgentParams[AgentNetworkParams],
            training_metrics: Metrics,
        ) -> None:
            evaluator_metrics = evaluator.run_evaluation(
                params, training_metrics=training_metrics
            )
            logging.info(f"All {name}Evaluation metrics: {evaluator_metrics}")
            logging.info(f"{name}Episode reward mean: {evaluator_metrics['eval/episode_reward']}")
            logging.info(f"{name}Episode reward std: {evaluator_metrics['eval/episode_reward_std']}")
            logging.info(f"{name}Episode energy per meter mean: {evaluator_metrics['eval/episode_energy_per_meter']}")
            logging.info(f"{name}Episode energy per meter std: {evaluator_metrics['eval/episode_energy_per_meter_std']}")
            logging.info(f"{name}Episode average foot slip mean: {evaluator_metrics['eval/episode_foot_slip']}")
            logging.info(f"{name}Episode average foot slip std: {evaluator_metrics['eval/episode_foot_slip_std']}")
            if "eval/episode_average_tracking_error" in evaluator_metrics:
                logging.info(f"{name}Average tracking error mean: {evaluator_metrics['eval/episode_average_tracking_error']}")
                logging.info(f"{name}Average tracking error std: {evaluator_metrics['eval/episode_average_tracking_error_std']}")
            progress_fn(current_step, evaluator_metrics)
            energy_progress_fn(current_step, evaluator_metrics)

        return evaluation_fn, times
