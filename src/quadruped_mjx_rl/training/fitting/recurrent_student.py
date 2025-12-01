import functools
from typing import Callable

import jax
from jax import numpy as jnp

from quadruped_mjx_rl.models.architectures.teacher_student_base import (
    TeacherStudentAgentParams,
    TeacherStudentNetworkParams,
    TeacherStudentAgent,
)
from quadruped_mjx_rl.models.networks_utils import (
    RecurrentAgentState, PolicyFactory, RecurrentNetwork,
)
from quadruped_mjx_rl.training import gradients, training_utils
from quadruped_mjx_rl.training.acting import Evaluator
from quadruped_mjx_rl.training.fitting import optimization
from quadruped_mjx_rl.training.fitting.optimization import EvalFn
from quadruped_mjx_rl.types import Metrics, PRNGKey, Transition, Observation
from quadruped_mjx_rl.training.configs import TeacherStudentOptimizerConfig
from quadruped_mjx_rl.training.fitting.teacher_student import TeacherStudentOptimizerState


class RecurrentStudentFitter(optimization.Fitter[TeacherStudentNetworkParams]):
    def __init__(
        self,
        optimizer_config: TeacherStudentOptimizerConfig,
        network: TeacherStudentAgent,
        main_loss_fn: optimization.LossFn[TeacherStudentNetworkParams],
        algorithm_hyperparams: optimization.HyperparamsPPO,
    ):
        self._network = network
        if not isinstance(network._student_encoder_network, RecurrentNetwork):
            raise ValueError("Student encoder network must be a recurrent network!")
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
            compute_student_recurrent_loss, network=network
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

    @property
    def network(self):
        return self._network

    def optimizer_init(self, network_params):
        return TeacherStudentOptimizerState(
            optimizer_state=self.teacher_optimizer.init(network_params),
            student_optimizer_state=self.student_optimizer.init(network_params),
        )

    def minibatch_step(
        self,
        carry: tuple[TeacherStudentOptimizerState, TeacherStudentAgentParams, PRNGKey],
        data: tuple[tuple[Transition, Observation], RecurrentAgentState],
        normalizer_params,
        recurrent_buffer=None,
    ):
        optimizer_state, network_params, key = carry
        key, teacher_key, student_key = jax.random.split(key, 3)
        (transitions, vision_obs), agent_state = data
        ((teacher_loss, teacher_metrics), network_params, teacher_optimizer_state) = (
            self.teacher_gradient_update_fn(
                network_params,
                normalizer_params,
                transitions,  # TODO: add ppo support for visual obs in teacher
                teacher_key,
                optimizer_state=optimizer_state.optimizer_state,
            )
        )
        ((student_loss, student_metrics), network_params, student_optimizer_state) = (
            self.student_gradient_update_fn(
                network_params,
                normalizer_params,
                data,
                student_key,
                recurrent_buffer,
                optimizer_state=optimizer_state.student_optimizer_state,
            )
        )
        # TODO: update the recurrent buffer
        optimizer_state = TeacherStudentOptimizerState(
            optimizer_state=teacher_optimizer_state,
            student_optimizer_state=student_optimizer_state,
        )
        metrics = teacher_metrics | student_metrics
        return (optimizer_state, network_params, key), metrics

    def update_agent_state(
        self,
        agent_params: TeacherStudentAgentParams,
        agent_state: RecurrentAgentState,
        transitions: Transition,
        vision_obs: Observation,
        init_carry_key: PRNGKey,
    ):
        done_anywhere = jnp.any(1 - transitions.discount, axis=-1)
        preencoded_latents = self.network._student_encoder_network.apply(agent_params, vision_obs, agent_state.recurrent_carry)
        return agent_state

    def make_evaluation_fn(
        self,
        rng: PRNGKey,
        evaluator_factory: Callable[[PRNGKey, PolicyFactory], Evaluator],
        progress_fn_factory: Callable[..., Callable[[int, Metrics], None]],
        deterministic_eval: bool = True,
    ) -> tuple[EvalFn[TeacherStudentNetworkParams], list[float]]:
        # TODO
        pass


def compute_student_recurrent_loss(*args, **kwargs):
    # TODO
    pass



