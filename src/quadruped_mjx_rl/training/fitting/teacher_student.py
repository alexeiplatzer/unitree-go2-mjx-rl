"""Custom fitter for the teacher-student actor-critic architecture"""
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from typing import Protocol
import functools

from flax.struct import dataclass as flax_dataclass
import optax
import jax
from jax import numpy as jnp

from quadruped_mjx_rl.training.gradients import gradient_update_fn
from quadruped_mjx_rl.types import Transition, Params, PRNGKey, Metrics
from quadruped_mjx_rl.running_statistics import RunningStatisticsState
from quadruped_mjx_rl.training import gradients, training_utils
from quadruped_mjx_rl.training.fitting import optimization
from quadruped_mjx_rl.models.architectures.guided_actor_critic import (
    TeacherStudentNetworks,
    TeacherStudentNetworkParams,
    TeacherStudentAgentParams,
)


@dataclass
class TeacherStudentOptimizerConfig(optimization.OptimizerConfig):
    student_learning_rate: float = 0.0004


@flax_dataclass
class TeacherStudentOptimizerState(optimization.OptimizerState):
    student_optimizer_state: optax.OptState


class TeacherStudentFitter(optimization.Fitter[TeacherStudentNetworkParams]):
    def __init__(
        self,
        optimizer_config: TeacherStudentOptimizerConfig,
        network: TeacherStudentNetworks,
        main_loss_fn: optimization.LossFn[TeacherStudentNetworkParams],
        algorithm_hyperparams: optimization.HyperparamsPPO,
    ):
        self.teacher_optimizer = optimization.make_optimizer(
            optimizer_config.learning_rate, optimizer_config.max_grad_norm
        )
        self.student_optimizer = optimization.make_optimizer(
            optimizer_config.student_learning_rate, optimizer_config.max_grad_norm
        )
        teacher_loss_fn = functools.partial(
            main_loss_fn,
            network=network,
            hyperparams=algorithm_hyperparams,
        )
        student_loss_fn = functools.partial(
            compute_student_loss,
            network,
        )
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

    def optimizer_init(self, network_params):
        return TeacherStudentOptimizerState(
            optimizer_state=self.teacher_optimizer.init(network_params),
            student_optimizer_state=self.student_optimizer.init(network_params),
        )

    def minibatch_step(
        self,
        carry: tuple[TeacherStudentOptimizerState, TeacherStudentAgentParams, PRNGKey],
        data,
        normalizer_params,
    ):
        optimizer_state, network_params, key = carry
        key, teacher_key, student_key = jax.random.split(key, 3)
        (
            (teacher_loss, teacher_metrics), params, teacher_optimizer_state
        ) = self.teacher_gradient_update_fn(
            network_params,
            normalizer_params,
            data,
            teacher_key,
            optimizer_state=optimizer_state.optimizer_state,
        )
        (
            (student_loss, student_metrics), params, student_optimizer_state
        ) = self.student_gradient_update_fn(
            network_params,
            normalizer_params,
            data,
            student_key,
            optimizer_state=optimizer_state.student_optimizer_state
        )
        optimizer_state = TeacherStudentOptimizerState(
            optimizer_state=teacher_optimizer_state,
            student_optimizer_state=student_optimizer_state,
        )
        metrics = teacher_metrics | student_metrics
        return (optimizer_state, params, key), metrics


def compute_student_loss(
    networks: TeacherStudentNetworks,
    network_params: TeacherStudentNetworkParams,
    preprocessor_params: optimization.PreprocessorParams,
    data: Transition,
) -> tuple[jnp.ndarray, Metrics]:
    """Computes Adaptation module loss."""

    encoder_apply = networks.teacher_encoder_network.apply
    adapter_apply = networks.student_encoder_network.apply

    # Put the time dimension first.
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)
    teacher_latent_vector = encoder_apply(
        preprocessor_params, network_params.teacher_encoder, data.observation
    )
    student_latent_vector = adapter_apply(
        preprocessor_params, network_params.student_encoder, data.observation
    )
    total_loss = optax.squared_error(teacher_latent_vector - student_latent_vector).mean()

    return total_loss, {"student_total_loss": total_loss}

