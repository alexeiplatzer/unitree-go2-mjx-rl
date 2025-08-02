from quadruped_mjx_rl import types
import optax
import jax
import jax.numpy as jnp

from quadruped_mjx_rl.models.architectures.guided_actor_critic import (
    TeacherStudentNetworks,
    TeacherStudentNetworkParams,
)
from quadruped_mjx_rl.models.agents.ppo.losses import compute_gae


def compute_teacher_loss(
    params: TeacherStudentNetworkParams,
    normalizer_params,
    data: types.Transition,
    rng: jnp.ndarray,
    teacher_student_networks: TeacherStudentNetworks,
    entropy_cost: float = 1e-4,
    discounting: float = 0.9,
    reward_scaling: float = 1.0,
    gae_lambda: float = 0.95,
    clipping_epsilon: float = 0.3,
    normalize_advantage: bool = True,
) -> tuple[jnp.ndarray, types.Metrics]:
    """Computes PPO loss.

    Args:
      params: Network parameters,
      normalizer_params: Parameters of the normalizer.
      data: Transition that with leading dimension [B, T]. Extra fields required
        are ['state_extras']['truncation'] ['policy_extras']['raw_action']
        ['policy_extras']['log_prob']
      rng: Random key
      teacher_student_networks: PPO networks.
      entropy_cost: entropy cost.
      discounting: discounting,
      reward_scaling: reward multiplier.
      gae_lambda: General advantage estimation lambda.
      clipping_epsilon: Policy loss clipping epsilon
      normalize_advantage: whether to normalize advantage estimate

    Returns:
      A tuple (loss, metrics)
    """
    parametric_action_distribution = teacher_student_networks.parametric_action_distribution
    encoder_apply = teacher_student_networks.teacher_encoder_network.apply
    policy_apply = teacher_student_networks.policy_network.apply
    value_apply = teacher_student_networks.value_network.apply

    # Put the time dimension first.
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)
    latent_vector = encoder_apply(normalizer_params, params.teacher_encoder, data.observation)
    policy_logits = policy_apply(
        normalizer_params, params.policy, data.observation, latent_vector
    )
    baseline = value_apply(normalizer_params, params.value, data.observation, latent_vector)
    terminal_obs = jax.tree_util.tree_map(lambda x: x[-1], data.next_observation)
    terminal_latent = encoder_apply(normalizer_params, params.teacher_encoder, terminal_obs)
    bootstrap_value = value_apply(
        normalizer_params, params.value, terminal_obs, terminal_latent
    )

    rewards = data.reward * reward_scaling
    truncation = data.extras["state_extras"]["truncation"]
    termination = (1 - data.discount) * (1 - truncation)

    target_action_log_probs = parametric_action_distribution.log_prob(
        policy_logits, data.extras["policy_extras"]["raw_action"]
    )
    behaviour_action_log_probs = data.extras["policy_extras"]["log_prob"]

    vs, advantages = compute_gae(
        truncation=truncation,
        termination=termination,
        rewards=rewards,
        values=baseline,
        bootstrap_value=bootstrap_value,
        lambda_=gae_lambda,
        discount=discounting,
    )
    if normalize_advantage:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    rho_s = jnp.exp(target_action_log_probs - behaviour_action_log_probs)

    surrogate_loss1 = rho_s * advantages
    surrogate_loss2 = jnp.clip(rho_s, 1 - clipping_epsilon, 1 + clipping_epsilon) * advantages

    policy_loss = -jnp.mean(jnp.minimum(surrogate_loss1, surrogate_loss2))

    # Value function loss
    v_error = vs - baseline
    v_loss = jnp.mean(v_error * v_error) * 0.5 * 0.5

    # Entropy reward
    entropy = jnp.mean(parametric_action_distribution.entropy(policy_logits, rng))
    entropy_loss = entropy_cost * -entropy

    total_loss = policy_loss + v_loss + entropy_loss
    return total_loss, {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "v_loss": v_loss,
        "entropy_loss": entropy_loss,
    }


def compute_student_loss(
    teacher_student_params: TeacherStudentNetworkParams,
    normalizer_params,
    data: types.Transition,
    teacher_student_networks: TeacherStudentNetworks,
) -> tuple[jnp.ndarray, types.Metrics]:
    """Computes Adaptation module loss."""

    encoder_apply = teacher_student_networks.teacher_encoder_network.apply
    adapter_apply = teacher_student_networks.student_encoder_network.apply

    # Put the time dimension first.
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)
    teacher_latent_vector = encoder_apply(
        normalizer_params, teacher_student_params.teacher_encoder, data.observation
    )
    student_latent_vector = adapter_apply(
        normalizer_params, teacher_student_params.student_encoder, data.observation
    )
    total_loss = optax.squared_error(teacher_latent_vector - student_latent_vector).mean()

    return total_loss, {"total_loss": total_loss}
