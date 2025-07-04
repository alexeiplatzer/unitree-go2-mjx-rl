"""Brax training gradient utility functions."""

from typing import Callable, Optional

import jax
import optax


def loss_and_pgrad(
    loss_fn: Callable[..., float],
    pmap_axis_name: Optional[str],
    has_aux: bool = False,
):
    g = jax.value_and_grad(loss_fn, has_aux=has_aux)

    def h(*args, **kwargs):
        value, grad = g(*args, **kwargs)
        return value, jax.lax.pmean(grad, axis_name=pmap_axis_name)

    return g if pmap_axis_name is None else h


def gradient_update_fn(
    loss_fn: Callable[..., float],
    optimizer: optax.GradientTransformation,
    pmap_axis_name: Optional[str],
    has_aux: bool = False,
):
    """Wrapper of the loss function that apply gradient updates.

      Args:
        loss_fn: The loss function.
        optimizer: The optimizer to apply gradients.
        pmap_axis_name: If relevant, the name of the pmap axis to synchronize
          gradients.
        has_aux: Whether the loss_fn has auxiliary data.

      Returns:
        A function that takes the same argument as the loss function plus the
        optimizer state. The output of this function is the loss, the new parameter,
        and the new optimizer state.
      """
    loss_and_pgrad_fn = loss_and_pgrad(
        loss_fn, pmap_axis_name=pmap_axis_name, has_aux=has_aux
    )

    def f(*args, optimizer_state):
        value, grads = loss_and_pgrad_fn(*args)
        params_update, optimizer_state = optimizer.update(grads, optimizer_state)
        params = optax.apply_updates(args[0], params_update)
        return value, params, optimizer_state

    return f
