"""Custom fitter for the teacher-student actor-critic architecture"""

import functools
from dataclasses import dataclass
from typing import Callable
import logging

import jax
import optax
from flax.struct import dataclass as flax_dataclass
from jax import numpy as jnp

from quadruped_mjx_rl.models.architectures.teacher_student_base import (
    TeacherStudentAgentParams,
    TeacherStudentNetworkParams,
    TeacherStudentNetworks,
    ActorCriticNetworks,
    FeedForwardNetwork,
)
from quadruped_mjx_rl.models.networks_utils import AgentNetworkParams, PolicyFactory
from quadruped_mjx_rl.training import gradients, training_utils
from quadruped_mjx_rl.training.acting import Evaluator
from quadruped_mjx_rl.training.fitting import optimization
from quadruped_mjx_rl.training.fitting.optimization import EvalFn, SimpleFitter
from quadruped_mjx_rl.types import Metrics, PRNGKey, Transition, RecurrentTransition
from quadruped_mjx_rl.training.configs import TeacherStudentOptimizerConfig
from quadruped_mjx_rl.training.fitting.teacher_student import (
    TeacherStudentFitter, compute_student_loss
)


def compute_student_recurrent_loss(
    network_params: TeacherStudentNetworkParams,
    preprocessor_params: optimization.PreprocessorParams,
    data: RecurrentTransition,
    network: TeacherStudentNetworks,
) -> tuple[jnp.ndarray, Metrics]:
    """Computes Adaptation module loss."""

    encoder_apply = network.teacher_encoder_network.apply
    adapter_apply = network.student_encoder_network.apply

    # Put the time dimension first.
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)
    teacher_latent_vector = encoder_apply(
        preprocessor_params, network_params.teacher_encoder, data.observation
    )
    student_latent_vector = data.recurrent_encoding
    teacher_latent_vector = jax.lax.stop_gradient(teacher_latent_vector)
    total_loss = optax.squared_error(teacher_latent_vector - student_latent_vector).mean()

    return total_loss, {"student_total_loss": total_loss}
