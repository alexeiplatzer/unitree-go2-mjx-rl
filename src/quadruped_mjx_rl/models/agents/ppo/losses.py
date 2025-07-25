
import jax
import jax.numpy as jnp


def compute_gae(
    truncation: jnp.ndarray,
    termination: jnp.ndarray,
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    bootstrap_value: jnp.ndarray,
    lambda_: float = 1.0,
    discount: float = 0.99,
):
    """
    Calculates the Generalized Advantage Estimation (GAE).

    Args:
      truncation: A float32 tensor of shape [T, B] with truncation signal.
      termination: A float32 tensor of shape [T, B] with termination signal.
      rewards: A float32 tensor of shape [T, B] containing rewards generated
        by following the behavior policy.
      values: A float32 tensor of shape [T, B] with the value function estimates
        with respect to the target policy.
      bootstrap_value: A float32 of shape [B] with the value function estimate at
        time T.
      lambda_: Mix between 1-step (lambda_=0) and n-step (lambda_=1). Defaults to
        lambda_=1.
      discount: TD discount.

    Returns:
      A float32 tensor of shape [T, B]. Can be used as target to
        train a baseline (V(x_t) - vs_t)^2.
      A float32 tensor of shape [T, B] of advantages.
    """

    truncation_mask = 1 - truncation
    # Append bootstrapped value to get [v1, ..., v_t+1]
    values_t_plus_1 = jnp.concatenate([values[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0)
    deltas = rewards + discount * (1 - termination) * values_t_plus_1 - values
    deltas *= truncation_mask

    acc = jnp.zeros_like(bootstrap_value)
    vs_minus_v_xs = []

    def compute_vs_minus_v_xs(carry, target_t):
        lambda_, acc = carry
        truncation_mask, delta, termination = target_t
        acc = delta + discount * (1 - termination) * truncation_mask * lambda_ * acc
        return (lambda_, acc), (acc)

    (_, _), (vs_minus_v_xs) = jax.lax.scan(
        compute_vs_minus_v_xs,
        (lambda_, acc),
        (truncation_mask, deltas, termination),
        length=int(truncation_mask.shape[0]),
        reverse=True,
    )
    # Add V(x_s) to get v_s.
    vs = jnp.add(vs_minus_v_xs, values)

    vs_t_plus_1 = jnp.concatenate([vs[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0)
    advantages = (
        rewards + discount * (1 - termination) * vs_t_plus_1 - values
    ) * truncation_mask
    return jax.lax.stop_gradient(vs), jax.lax.stop_gradient(advantages)
