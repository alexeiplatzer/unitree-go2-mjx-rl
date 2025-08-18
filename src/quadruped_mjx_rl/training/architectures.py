import functools
from dataclasses import asdict

import jax
import optax
from flax.struct import dataclass as flax_dataclass

from quadruped_mjx_rl.models.architectures import OptimizerConfig
from quadruped_mjx_rl.training.configs import HyperparamsPPO
from quadruped_mjx_rl.training import gradients, training_utils


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


def make_actor_critic_step(
    optimizer_config: OptimizerConfig, main_loss_fn, hyperparams_ppo: HyperparamsPPO
):
    optimizer = make_optimizer(optimizer_config.learning_rate, optimizer_config.max_grad_norm)
    loss_fn = functools.partial(
        main_loss_fn,
        **asdict(hyperparams_ppo),
    )
    gradient_update_fn = gradients.gradient_update_fn(
        loss_fn=loss_fn,
        optimizer=optimizer,
        pmap_axis_name=training_utils.PMAP_AXIS_NAME,
        has_aux=True,
    )

    def optimizer_init(params):
        return optimizer.init(params)

    def minibatch_step(carry, normalizer_params, data):
        optimizer_state, network_params, key = carry
        key, key_loss = jax.random.split(key)
        (loss, metrics), params, optimizer_state = gradient_update_fn(
            network_params,
            normalizer_params,
            data,
            key_loss,
            optimizer_state=optimizer_state,
        )
        return (optimizer_state, params, key), metrics

    return optimizer_init, minibatch_step
