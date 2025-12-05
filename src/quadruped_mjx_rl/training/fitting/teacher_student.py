"""Custom fitter for the teacher-student actor-critic architecture"""

import functools
from typing import Any, Callable
import logging

import jax
import optax
from flax.struct import dataclass as flax_dataclass
from jax import numpy as jnp

import quadruped_mjx_rl.training.algorithms.ppo
from quadruped_mjx_rl.models.architectures.teacher_student_base import (
    TeacherStudentAgentParams,
    TeacherStudentNetworkParams,
    TeacherStudentNetworks,
)
from quadruped_mjx_rl.models.types import PolicyFactory
from quadruped_mjx_rl.running_statistics import RunningStatisticsState
from quadruped_mjx_rl.training import gradients, training_utils
from quadruped_mjx_rl.training.evaluator import Evaluator
from quadruped_mjx_rl.training.fitting.optimization import (
    LossFn,
    Fitter,
    EvalFn,
    SimpleFitter,
    OptimizerState,
    make_optimizer,
)
from quadruped_mjx_rl.types import Metrics, PRNGKey, Transition
from quadruped_mjx_rl.training.configs import TeacherStudentOptimizerConfig
from quadruped_mjx_rl.training.algorithms.ppo import HyperparamsPPO


@flax_dataclass
class TeacherStudentOptimizerState(OptimizerState):
    student_optimizer_state: optax.OptState


class TeacherStudentFitter(Fitter[TeacherStudentNetworkParams]):
    def __init__(
        self,
        optimizer_config: TeacherStudentOptimizerConfig,
        network: TeacherStudentNetworks,
        main_loss_fn: LossFn[TeacherStudentNetworkParams],
        algorithm_hyperparams: HyperparamsPPO,
    ):
        self._network = network
        self.teacher_optimizer = make_optimizer(
            optimizer_config.learning_rate, optimizer_config.max_grad_norm
        )
        self.student_optimizer = make_optimizer(
            optimizer_config.student_learning_rate, optimizer_config.max_grad_norm
        )
        teacher_loss_fn = functools.partial(
            main_loss_fn,
            network=network,
            hyperparams=algorithm_hyperparams,
        )
        student_loss_fn = functools.partial(compute_student_loss, network=network)
        self.teacher_gradient_update_fn = gradients.gradient_update_fn(
            loss_fn=teacher_loss_fn,
            optimizer=self.teacher_optimizer,
            pmap_axis_name=training_utils.PMAP_AXIS_NAME,
            has_aux=True,
        )
        self.student_gradient_update_fn = gradients.gradient_update_fn(
            loss_fn=student_loss_fn,
            optimizer=self.student_optimizer,
            pmap_axis_name=training_utils.PMAP_AXIS_NAME,
            has_aux=True,
        )

    @property
    def network(self) -> TeacherStudentNetworks:
        return self._network

    def optimizer_init(
        self, network_params: TeacherStudentNetworkParams
    ) -> TeacherStudentOptimizerState:
        return TeacherStudentOptimizerState(
            optimizer_state=self.teacher_optimizer.init(network_params),
            student_optimizer_state=self.student_optimizer.init(network_params),
        )

    def minibatch_step(
        self,
        carry: tuple[TeacherStudentOptimizerState, TeacherStudentAgentParams, PRNGKey],
        data: Transition,
        normalizer_params: RunningStatisticsState,
    ) -> tuple[
        tuple[TeacherStudentOptimizerState, TeacherStudentAgentParams, PRNGKey], dict[str, Any]
    ]:
        optimizer_state, network_params, key = carry
        key, teacher_key = jax.random.split(key, 2)
        ((teacher_loss, teacher_metrics), network_params, teacher_optimizer_state) = (
            self.teacher_gradient_update_fn(
                network_params,
                normalizer_params,
                data,
                teacher_key,
                optimizer_state=optimizer_state.optimizer_state,
            )
        )
        ((student_loss, student_metrics), network_params, student_optimizer_state) = (
            self.student_gradient_update_fn(
                network_params,
                normalizer_params,
                data,
                optimizer_state=optimizer_state.student_optimizer_state,
            )
        )
        optimizer_state = TeacherStudentOptimizerState(
            optimizer_state=teacher_optimizer_state,
            student_optimizer_state=student_optimizer_state,
        )
        metrics = teacher_metrics | student_metrics
        return (optimizer_state, network_params, key), metrics

    def make_evaluation_fn(
        self,
        rng: PRNGKey,
        evaluator_factory: Callable[[PRNGKey, PolicyFactory], Evaluator],
        progress_fn_factory: Callable[..., Callable[[int, Metrics], None]],
        deterministic_eval: bool = True,
    ) -> tuple[EvalFn[TeacherStudentNetworkParams], list[float]]:
        teacher_policy_factory = self.network.get_acting_policy_factory()
        student_policy_factory = self.network.get_student_policy_factory()
        teacher_eval_key, student_eval_key = jax.random.split(rng, 2)
        teacher_progress_fn = functools.partial(
            progress_fn_factory, title="Teacher policy evaluation results"
        )
        student_progress_fn = functools.partial(
            progress_fn_factory, title="Student policy evaluation results", color="green"
        )
        encoder_convergence_progress_fn, conv_times = progress_fn_factory(
            title="Student encoder convergence",
            data_key="training/student_total_loss",
            data_err_key=None,
            label_key="student_total_loss",
            color="red",
        )
        teacher_eval_fn, teacher_times = SimpleFitter.make_evaluation_fn(
            self,
            teacher_eval_key,
            lambda k, _: evaluator_factory(k, teacher_policy_factory),
            teacher_progress_fn,
        )
        student_eval_fn, student_times = SimpleFitter.make_evaluation_fn(
            self,
            student_eval_key,
            lambda k, _: evaluator_factory(k, student_policy_factory),
            student_progress_fn,
        )

        def evaluation_fn(current_step, params, training_metrics):
            teacher_eval_fn(current_step, params, training_metrics)
            student_eval_fn(current_step, params, training_metrics)
            logging.info(
                "student absolute loss: %s",
                training_metrics.get("training/student_total_loss", "not known yet"),
            )
            if training_metrics:
                encoder_convergence_progress_fn(current_step, training_metrics)

        return evaluation_fn, teacher_times


def compute_student_loss(
    network_params: TeacherStudentNetworkParams,
    preprocessor_params: quadruped_mjx_rl.models.types.PreprocessorParams,
    data: Transition,
    network: TeacherStudentNetworks,
) -> tuple[jax.Array, Metrics]:
    """Computes Adaptation module loss."""

    # Put the time dimension first.
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)
    teacher_latent_vector = network.apply_acting_encoder(
        preprocessor_params, network_params.acting_encoder, data.observation
    )
    student_latent_vector = network.apply_student_encoder(
        preprocessor_params, network_params.student_encoder, data.observation
    )
    teacher_latent_vector = jax.lax.stop_gradient(teacher_latent_vector)
    total_loss = optax.squared_error(teacher_latent_vector - student_latent_vector).mean()

    return total_loss, {"student_total_loss": total_loss}
