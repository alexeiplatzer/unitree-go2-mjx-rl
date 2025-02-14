from collections.abc import Sequence

import jax
from jax import numpy as jnp
from flax import linen as nn

from brax.training import distribution
from brax.training.networks import ActivationFn, Initializer  # types
from brax.training.networks import MLP


class TeacherStudentActorCritic(nn.Module):
    observation_size: int
    priveleged_observation_size: int
    num_history_steps: int
    action_size: int
    latent_size: int
    encoder_hidden_layer_sizes: Sequence[int]
    adapter_hidden_layer_sizes: Sequence[int]
    policy_hidden_layer_sizes: Sequence[int]
    value_hidden_layer_sizes: Sequence[int]
    activation: ActivationFn = nn.relu
    kernel_init: Initializer = nn.initializers.lecun_uniform()

    def setup(self):
        self.parametric_action_distribution = distribution.NormalTanhDistribution(
            event_size=self.action_size
        )
        self.encoder_module = MLP(
            layer_sizes=[self.priveleged_observation_size]
            + list(self.encoder_hidden_layer_sizes)
            + [self.latent_layer_size],
            activation=self.activation,
        )
        historical_observations_size = self.observation_size * self.num_history_steps
        self.adapter_module = MLP(
            layer_sizes=[historical_observations_size]
            + list(self.adapter_hidden_layer_sizes)
            + [self.latent_layer_size],
            activation=self.activation,
        )
        extended_observations_size = self.latent_layer_size + self.observation_size
        self.policy_module = MLP(
            layer_sizes=[extended_observations_size]
            + list(self.policy_hidden_layer_sizes)
            + [self.action_size],
            activation=self.activation,
        )
        self.value_module = MLP(
            layer_sizes=[extended_observations_size]
            + list(self.value_hidden_layer_sizes)
            + [1],
            activation=self.activation,
        )

    def __call__(
        self, observation: jnp.ndarray, privileged_observation: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return self.apply_teacher(observation, privileged_observation)

    def apply_teacher(
        self, observation: jnp.ndarray, privileged_observation: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        latent = self.encoder_module(privileged_observation)
        extended_observation = jnp.concatenate([latent, observation])
        policy_logits = self.policy_module(extended_observation)
        value_logits = self.value_module(extended_observation)
        return latent, policy_logits, value_logits

    def encode_student(self, observation_history: jnp.ndarray) -> jnp.ndarray:
        return self.adapter_module(observation_history)

    def encode_teacher(self, priveleged_observation: jnp.ndarray) -> jnp.ndarray:
        return self.encoder_module(priveleged_observation)

    def act(self, observation: jnp.ndarray, latent: jnp.ndarray) -> jnp.ndarray:
        return self.policy_module(jnp.concatenate([latent, observation]))

    def evaluate(self, observation: jnp.ndarray, latent: jnp.ndarray) -> jnp.ndarray:
        return self.value_module(jnp.concatenate([latent, observation]))
