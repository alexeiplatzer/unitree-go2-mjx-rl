import functools
import logging
from pathlib import Path
from typing import Any

import jax
import optax
from jax import numpy as jnp

from quadruped_mjx_rl.physics_pipeline import Env
from quadruped_mjx_rl.models.architectures.teacher_student_recurrent import (
    TeacherStudentAgentParams,
    TeacherStudentNetworkParams,
    TeacherStudentRecurrentNetworks,
)
from quadruped_mjx_rl.models.types import PreprocessorParams, RecurrentAgentState
from quadruped_mjx_rl.running_statistics import RunningStatisticsState
from quadruped_mjx_rl.training import gradients, training_utils
from quadruped_mjx_rl.training.algorithms.ppo import HyperparamsPPO
from quadruped_mjx_rl.training.configs import (
    TeacherStudentOptimizerConfig, TrainingConfig,
)
from quadruped_mjx_rl.training.progress_plotting import make_progress_fn
from quadruped_mjx_rl.training.evaluator import Evaluator
from quadruped_mjx_rl.training.fitting.optimization import EvalFn
from quadruped_mjx_rl.training.fitting.optimization import LossFn, make_optimizer, SimpleFitter
from quadruped_mjx_rl.training.fitting.teacher_student import TeacherStudentOptimizerState
from quadruped_mjx_rl.types import Metrics, PRNGKey, Transition


class RecurrentStudentFitter(SimpleFitter[TeacherStudentNetworkParams]):
    def __init__(
        self,
        optimizer_config: TeacherStudentOptimizerConfig,
        network: TeacherStudentRecurrentNetworks,
        main_loss_fn: LossFn[TeacherStudentNetworkParams, TeacherStudentRecurrentNetworks],
        algorithm_hyperparams: HyperparamsPPO,
    ):
        super().__init__(
            optimizer_config=optimizer_config,
            network=network,
            main_loss_fn=main_loss_fn,
            algorithm_hyperparams=algorithm_hyperparams,
        )
        self.student_optimizer = make_optimizer(
            optimizer_config.student_learning_rate, optimizer_config.max_grad_norm
        )
        student_loss_fn = functools.partial(compute_student_recurrent_loss, network=network)
        self.student_gradient_update_fn = gradients.gradient_update_fn(
            loss_fn=student_loss_fn,
            optimizer=self.student_optimizer,
            pmap_axis_name=training_utils.PMAP_AXIS_NAME,
            has_aux=True,
        )

    @property
    def network(self) -> TeacherStudentRecurrentNetworks:
        return self._network

    def optimizer_init(
        self, network_params: TeacherStudentNetworkParams
    ) -> TeacherStudentOptimizerState:
        return TeacherStudentOptimizerState(
            optimizer_state=self.optimizer.init(network_params),
            student_optimizer_state=self.student_optimizer.init(network_params),
        )

    def minibatch_step(
        self,
        carry: tuple[
            TeacherStudentOptimizerState,
            TeacherStudentNetworkParams,
            PRNGKey,
        ],
        data: tuple[Transition, RecurrentAgentState],
        normalizer_params: RunningStatisticsState,
    ) -> tuple[
        tuple[
            TeacherStudentOptimizerState,
            TeacherStudentNetworkParams,
            PRNGKey,
        ],
        tuple[RecurrentAgentState, dict[str, Any]],
    ]:
        optimizer_state, network_params, key = carry
        key, teacher_key, student_key = jax.random.split(key, 3)
        transitions, agent_state = data
        ((teacher_loss, teacher_metrics), teacher_update_params, teacher_optimizer_state) = (
            self.gradient_update_fn(
                network_params,
                normalizer_params,
                transitions,
                teacher_key,
                optimizer_state=optimizer_state.optimizer_state,
            )
        )
        (
            (student_loss, (agent_state, student_metrics)),
            student_update_params,
            student_optimizer_state,
        ) = self.student_gradient_update_fn(
            network_params,
            normalizer_params,
            agent_state,
            transitions,
            student_key,
            optimizer_state=optimizer_state.student_optimizer_state,
        )
        network_params_updated = TeacherStudentNetworkParams(
            policy=teacher_update_params.policy,
            value=teacher_update_params.value,
            acting_encoder=teacher_update_params.acting_encoder,
            student_encoder=student_update_params.student_encoder,
        )
        optimizer_state = TeacherStudentOptimizerState(
            optimizer_state=teacher_optimizer_state,
            student_optimizer_state=student_optimizer_state,
        )
        metrics = teacher_metrics | student_metrics
        return (optimizer_state, network_params_updated, key), (agent_state, metrics)

    def make_evaluation_fn(
        self,
        eval_env: Env,
        eval_key: PRNGKey,
        training_config: TrainingConfig,
        show_outputs: bool = True,
        run_in_cell: bool = True,
        save_plots_path: Path | None = None,
    ) -> tuple[EvalFn[TeacherStudentNetworkParams], list[float]]:
        teacher_eval_key, student_eval_key = jax.random.split(eval_key, 2)
        teacher_evaluator = Evaluator(
            eval_env=eval_env,
            key=teacher_eval_key,
            num_eval_envs=training_config.num_eval_envs,
            episode_length=training_config.episode_length,
            action_repeat=training_config.action_repeat,
            unroll_factory=functools.partial(
                self.network.make_acting_unroll_fn,
                deterministic=training_config.deterministic_eval,
            ),
        )
        student_evaluator = Evaluator(
            eval_env=eval_env,
            key=student_eval_key,
            num_eval_envs=training_config.num_eval_envs,
            episode_length=training_config.episode_length,
            action_repeat=training_config.action_repeat,
            unroll_factory=functools.partial(
                self.network.make_student_unroll_fn,
                deterministic=training_config.deterministic_eval,
            ),
        )

        data_key = "eval/episode_reward"
        data_err_key = "eval/episode_reward_std"

        teacher_progress_fn, times = make_progress_fn(
            show_outputs=show_outputs,
            run_in_cell=run_in_cell,
            save_plots_path=save_plots_path / "teacher_evaluation" if save_plots_path else None,
            num_timesteps=training_config.num_timesteps,
            title="Teacher evaluation results",
            color="blue",
            label_key="episode reward",
            data_key=data_key,
            data_err_key=data_err_key,
            data_max=150,
            data_min=-10,
        )

        student_progress_fn, _ = make_progress_fn(
            show_outputs=show_outputs,
            run_in_cell=run_in_cell,
            save_plots_path=save_plots_path / "student_evaluation" if save_plots_path else None,
            num_timesteps=training_config.num_timesteps,
            title="Student evaluation results",
            color="green",
            label_key="episode reward",
            data_key=data_key,
            data_err_key=data_err_key,
            data_max=150,
            data_min=-10,
        )

        convergence_key = "training/student_total_loss"
        convergence_err_key = ""
        convergence_progress_fn, _ = make_progress_fn(
            show_outputs=show_outputs,
            run_in_cell=run_in_cell,
            save_plots_path=(
                save_plots_path / "student_convergence" if save_plots_path else None
            ),
            num_timesteps=training_config.num_timesteps,
            title="Student encoder convergence",
            color="red",
            label_key="episode MSE",
            data_key=convergence_key,
            data_err_key=convergence_err_key,
            data_max=1,
            data_min=0,  # TODO no idea what are appropriate values here, check in practice
        )

        teacher_eval_fn = self._evaluation_factory(
            data_key, data_err_key, teacher_evaluator, teacher_progress_fn, "Teacher"
        )
        student_eval_fn = self._evaluation_factory(
            data_key, data_err_key, student_evaluator, student_progress_fn, "Student"
        )

        def evaluation_fn(current_step, params, training_metrics):
            logging.info(f"current_step: {current_step}")
            teacher_eval_fn(current_step, params, training_metrics)
            student_eval_fn(current_step, params, training_metrics)
            if convergence_key in training_metrics:
                logging.info(f"student absolute loss: {training_metrics[convergence_key]}")
                convergence_progress_fn(current_step, training_metrics)
            else:
                logging.info("student absolute loss: not known yet.")

        return evaluation_fn, times


def compute_student_recurrent_loss(
    network_params: TeacherStudentNetworkParams,
    preprocessor_params: PreprocessorParams,
    agent_state: RecurrentAgentState,
    data: Transition,
    rng: PRNGKey,
    network: TeacherStudentRecurrentNetworks,
) -> tuple[jax.Array, tuple[RecurrentAgentState, Metrics]]:
    """Computes Adaptation module loss."""

    done = 1 - data.discount

    student_latent_vector, agent_state = network.apply_student_encoder(
        preprocessor_params=preprocessor_params,
        network_params=network_params,
        observation=data.observation,
        done=done,
        recurrent_agent_state=agent_state,
        key=jax.random.split(rng, done.shape[0]),
    )

    # Put the time dimension first.
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)

    teacher_latent_vector = network.apply_acting_encoder(
        preprocessor_params, network_params, data.observation
    )

    # # match the shapes
    # teacher_substeps = teacher_latent_vector.shape[0]
    # student_substeps = student_latent_vector.shape[0]
    # if teacher_substeps > student_substeps:
    #     student_latent_vector = jnp.repeat(
    #         student_latent_vector, teacher_substeps // student_substeps, axis=0
    #     )
    # elif teacher_substeps < student_substeps:
    #     teacher_latent_vector = jnp.repeat(
    #         teacher_latent_vector, student_substeps // teacher_substeps, axis=0
    #     )

    teacher_latent_vector = jax.lax.stop_gradient(teacher_latent_vector)
    total_loss = optax.squared_error(teacher_latent_vector[-1] - student_latent_vector).mean()

    return total_loss, (agent_state, {"student_total_loss": total_loss})
