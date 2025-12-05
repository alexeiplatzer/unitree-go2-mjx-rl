import functools
from typing import Any, Callable

import jax
import optax
from jax import numpy as jnp

import quadruped_mjx_rl.training.algorithms.ppo
from quadruped_mjx_rl.models.architectures.teacher_student_recurrent import (
    TeacherStudentAgentParams,
    TeacherStudentNetworkParams,
    TeacherStudentRecurrentNetworks,
)
from quadruped_mjx_rl.models.types import PolicyFactory, RecurrentAgentState
from quadruped_mjx_rl.running_statistics import RunningStatisticsState
from quadruped_mjx_rl.training import gradients, training_utils
from quadruped_mjx_rl.training.evaluator import Evaluator
from quadruped_mjx_rl.training.fitting.optimization import Fitter, LossFn, make_optimizer
from quadruped_mjx_rl.training.fitting.optimization import EvalFn
from quadruped_mjx_rl.types import Metrics, PRNGKey, Transition, Observation
from quadruped_mjx_rl.training.configs import TeacherStudentOptimizerConfig
from quadruped_mjx_rl.training.fitting.teacher_student import TeacherStudentOptimizerState


class RecurrentStudentFitter(Fitter[TeacherStudentNetworkParams]):
    def __init__(
        self,
        optimizer_config: TeacherStudentOptimizerConfig,
        network: TeacherStudentRecurrentNetworks,
        main_loss_fn: LossFn[TeacherStudentNetworkParams],
        algorithm_hyperparams: quadruped_mjx_rl.training.algorithms.ppo.HyperparamsPPO,
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
        student_loss_fn = functools.partial(compute_student_recurrent_loss, network=network)
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
    def network(self) -> TeacherStudentRecurrentNetworks:
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
            self.teacher_gradient_update_fn(
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
        rng: PRNGKey,
        evaluator_factory: Callable[[PRNGKey, PolicyFactory], Evaluator],
        progress_fn_factory: Callable[..., Callable[[int, Metrics], None]],
        deterministic_eval: bool = True,
    ) -> tuple[EvalFn[TeacherStudentNetworkParams], list[float]]:
        # TODO
        pass


def compute_student_recurrent_loss(
    network_params: TeacherStudentNetworkParams,
    preprocessor_params: quadruped_mjx_rl.models.types.PreprocessorParams,
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
        preprocessor_params, network_params.acting_encoder, data.observation
    )

    student_latent_vector, agent_state = network.apply_student_encoder(
        preprocessor_params=preprocessor_params,
        network_params=network_params.student_encoder,
        observation=data.observation,
        done=done,
        recurrent_agent_state=agent_state,
        reinitialization_key=rng,
    )

    teacher_latent_vector = jax.lax.stop_gradient(teacher_latent_vector)
    total_loss = optax.squared_error(teacher_latent_vector - student_latent_vector).mean()

    return total_loss, agent_state, {"student_total_loss": total_loss}
