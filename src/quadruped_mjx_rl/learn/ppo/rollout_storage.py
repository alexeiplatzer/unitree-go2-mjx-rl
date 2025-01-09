import jax
import jax.numpy as jnp


class RolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.privileged_observations = None
            self.observation_histories = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.env_bins = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        privileged_obs_shape,
        obs_history_shape,
        actions_shape,
    ):
        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.obs_history_shape = obs_history_shape
        self.actions_shape = actions_shape

        # Core
        self.observations = jnp.zeros((num_transitions_per_env, num_envs, *obs_shape))
        self.privileged_observations = jnp.zeros(
            (num_transitions_per_env, num_envs, *privileged_obs_shape)
        )
        self.observation_histories = jnp.zeros(
            (num_transitions_per_env, num_envs, *obs_history_shape)
        )
        self.rewards = jnp.zeros((num_transitions_per_env, num_envs, 1))
        self.actions = jnp.zeros((num_transitions_per_env, num_envs, *actions_shape))
        self.dones = jnp.zeros((num_transitions_per_env, num_envs, 1))

        # For PPO
        self.actions_log_prob = jnp.zeros((num_transitions_per_env, num_envs, 1))
        self.values = jnp.zeros((num_transitions_per_env, num_envs, 1))
        self.returns = jnp.zeros((num_transitions_per_env, num_envs, 1))
        self.advantages = jnp.zeros((num_transitions_per_env, num_envs, 1))
        self.mu = jnp.zeros((num_transitions_per_env, num_envs, *actions_shape))
        self.sigma = jnp.zeros((num_transitions_per_env, num_envs, *actions_shape))
        self.env_bins = jnp.zeros((num_transitions_per_env, num_envs, 1))

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        self.step = 0

    def add_transitions(self, transition: Transition):
        # if self.step >= self.num_transitions_per_env:
        #     raise AssertionError("Rollout buffer overflow")
        self.observations = self.observations.at[self.step].set(transition.observations)
        self.privileged_observations = self.privileged_observations.at[self.step].set(
            transition.privileged_observations
        )
        self.observation_histories = self.observation_histories.at[self.step].set(
            transition.observation_histories
        )
        self.actions = self.actions.at[self.step].set(transition.actions)
        self.rewards = self.rewards.at[self.step].set(transition.rewards.reshape(-1, 1))
        self.dones = self.dones.at[self.step].set(transition.dones.reshape(-1, 1))
        self.values = self.values.at[self.step].set(transition.values)
        self.actions_log_prob = self.actions_log_prob.at[self.step].set(
            transition.actions_log_prob.reshape(-1, 1)
        )
        self.mu = self.mu.at[self.step].set(transition.action_mean)
        self.sigma = self.sigma.at[self.step].set(transition.action_sigma)
        self.env_bins = self.env_bins.at[self.step].set(transition.env_bins.reshape(-1, 1))
        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        advantage = jnp.zeros(self.num_transitions_per_env)
        next_values = jnp.concatenate([self.values[1:], jnp.expand_dims(last_values, axis=0)])
        next_is_not_terminal = 1.0 - self.dones
        delta = self.rewards + next_is_not_terminal * gamma * next_values - self.values
        discount = next_is_not_terminal * gamma * lam

        def discounted_cumsum(carry, x):
            bias, weight = x
            y = bias + weight * carry
            return y, y

        _, self.advantages = jax.lax.scan(
            discounted_cumsum,
            0.0,
            (delta, discount),
            reverse=True,
        )

        self.returns = self.advantages + self.values

        # Compute and normalize the advantages
        self.advantages = (self.advantages - self.advantages.mean()) / (
            self.advantages.std() + 1e-8
        )

    # maybe jax will not jit, TODO verify that this works or search for alternatives
    def mini_batch_generator(self, rng: jax.Array, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = jax.random.permutation(
            rng,
            num_mini_batches * mini_batch_size,
        )

        observations = self.observations.reshape(-1, *self.observations.shape[2:])
        privileged_obs = self.privileged_observations.reshape(
            -1, *self.privileged_observations.shape[2:]
        )
        obs_history = self.observation_histories.reshape(
            -1, *self.observation_histories.shape[2:]
        )
        critic_observations = observations

        actions = self.actions.reshape(-1, *self.actions.shape[2:])
        values = self.values.reshape(-1, *self.values.shape[2:])
        returns = self.returns.reshape(-1, *self.returns.shape[2:])
        old_actions_log_prob = self.actions_log_prob.reshape(
            -1, *self.actions_log_prob.shape[2:]
        )
        advantages = self.advantages.reshape(-1, *self.advantages.shape[2:])
        old_mu = self.mu.reshape(-1, *self.mu.shape[2:])
        old_sigma = self.sigma.reshape(-1, *self.sigma.shape[2:])
        old_env_bins = self.env_bins.reshape(-1, *self.env_bins.shape[2:])

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx]
                critic_observations_batch = critic_observations[batch_idx]
                privileged_obs_batch = privileged_obs[batch_idx]
                obs_history_batch = obs_history[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]
                env_bins_batch = old_env_bins[batch_idx]
                yield (
                    obs_batch,
                    critic_observations_batch,
                    privileged_obs_batch,
                    obs_history_batch,
                    actions_batch,
                    target_values_batch,
                    advantages_batch,
                    returns_batch,
                    old_actions_log_prob_batch,
                    old_mu_batch,
                    old_sigma_batch,
                    None,
                    env_bins_batch,
                )
