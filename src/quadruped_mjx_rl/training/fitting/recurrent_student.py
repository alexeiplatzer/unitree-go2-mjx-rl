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
    TeacherStudentOptimizerConfig,
    TrainingWithRecurrentStudentConfig,
    TrainingWithVisionConfig,
)
from quadruped_mjx_rl.training.evaluation import make_progress_fn
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
            TeacherStudentAgentParams,
            RecurrentAgentState,
            PRNGKey,
        ],
        data: Transition,
        normalizer_params: RunningStatisticsState,
    ) -> tuple[
        tuple[
            TeacherStudentOptimizerState,
            TeacherStudentAgentParams,
            RecurrentAgentState,
            PRNGKey,
        ],
        dict[str, Any],
    ]:
        optimizer_state, network_params, agent_state, key = carry
        key, teacher_key, student_key = jax.random.split(key, 3)
        transitions, agent_state = data
        ((teacher_loss, teacher_metrics), network_params, teacher_optimizer_state) = (
            self.gradient_update_fn(
                network_params,
                normalizer_params,
                transitions,
                teacher_key,
                optimizer_state=optimizer_state.optimizer_state,
            )
        )
        (
            (student_loss, agent_state, student_metrics),
            network_params,
            student_optimizer_state,
        ) = self.student_gradient_update_fn(
            network_params,
            normalizer_params,
            agent_state,
            data,
            student_key,
            optimizer_state=optimizer_state.student_optimizer_state,
        )
        optimizer_state = TeacherStudentOptimizerState(
            optimizer_state=teacher_optimizer_state,
            student_optimizer_state=student_optimizer_state,
        )
        metrics = teacher_metrics | student_metrics
        return (optimizer_state, network_params, agent_state, key), metrics

    def make_evaluation_fn(
        self,
        eval_env: Env,
        eval_key: PRNGKey,
        training_config: TrainingWithRecurrentStudentConfig,
        show_outputs: bool = True,
        run_in_cell: bool = True,
        save_plots_path: Path | None = None,
    ) -> tuple[EvalFn[TeacherStudentNetworkParams], list[float]]:
        teacher_eval_key, student_eval_key = jax.random.split(eval_key, 2)
        proprio_steps_per_vision_step = (
            training_config.proprio_steps_per_vision_step
            if isinstance(training_config, TrainingWithVisionConfig)
            else 1
        )
        teacher_evaluator = Evaluator(
            eval_env=eval_env,
            key=teacher_eval_key,
            num_eval_envs=training_config.num_eval_envs,
            episode_length=training_config.episode_length,
            action_repeat=training_config.action_repeat,
            unroll_factory=lambda params: self.network.make_unroll_fn(
                agent_params=params,
                deterministic=training_config.deterministic_eval,
                vision=training_config.use_vision,
                proprio_steps_per_vision_step=proprio_steps_per_vision_step,
                policy_factory=self.network.get_acting_policy_factory(),
            ),
        )
        student_evaluator = Evaluator(
            eval_env=eval_env,
            key=student_eval_key,
            num_eval_envs=training_config.num_eval_envs,
            episode_length=training_config.episode_length,
            action_repeat=training_config.action_repeat,
            unroll_factory=lambda params: self.network.recurrent_unroll_factory(
                params=params,
                deterministic=training_config.deterministic_eval,
                vision=training_config.use_vision,
                vision_substeps=training_config.vision_steps_per_recurrent_step,
                proprio_substeps=proprio_steps_per_vision_step,
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
            data_max=40,
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
            data_max=40,
            data_min=-10,
        )

        convergence_key = "student_total_loss"
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
            data_max=100,
            data_min=-100,  # TODO no idea what are appropriate values here, check in practice
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
            logging.info(
                f"student absolute loss: "
                f"{training_metrics.get('training/student_total_loss', 'not known yet')}",
            )

        return evaluation_fn, times


def compute_student_recurrent_loss(
    network_params: TeacherStudentNetworkParams,
    preprocessor_params: PreprocessorParams,
    agent_state: RecurrentAgentState,
    data: Transition,
    rng: PRNGKey,
    network: TeacherStudentRecurrentNetworks,
) -> tuple[jax.Array, RecurrentAgentState, Metrics]:
    """Computes Adaptation module loss."""

    # Put the time dimension first.
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)

    done = 1 - data.discount

    teacher_latent_vector = network.apply_acting_encoder(
        preprocessor_params, network_params, data.observation
    )

    student_latent_vector, agent_state = network.apply_student_encoder(
        preprocessor_params=preprocessor_params,
        network_params=network_params,
        observation=data.observation,
        done=done,
        recurrent_agent_state=agent_state,
        reinitialization_key=rng,
    )

    teacher_latent_vector = jax.lax.stop_gradient(teacher_latent_vector)
    total_loss = optax.squared_error(teacher_latent_vector - student_latent_vector).mean()

    return total_loss, agent_state, {"student_total_loss": total_loss}
