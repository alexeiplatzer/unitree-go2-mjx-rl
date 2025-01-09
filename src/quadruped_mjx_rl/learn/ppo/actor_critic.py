# import torch
# import torch.nn as nn
from params_proto.proto import PrefixProto

# from torch.distributions import Normal

# import jax
from jax import numpy as jnp
from flax import linen as nn


class AC_Args(PrefixProto, cli=False):
    # policy
    init_noise_std = 1.0
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = "elu"  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    adaptation_module_branch_hidden_dims = [[256, 32]]

    env_factor_encoder_branch_input_dims = [18]
    env_factor_encoder_branch_latent_dims = [18]
    env_factor_encoder_branch_hidden_dims = [[256, 128]]


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(self, num_obs, num_privileged_obs, num_obs_history, num_actions, **kwargs):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        activation = get_activation(AC_Args.activation)

        for i, (branch_input_dim, branch_hidden_dims, branch_latent_dim) in enumerate(
            zip(
                AC_Args.env_factor_encoder_branch_input_dims,
                AC_Args.env_factor_encoder_branch_hidden_dims,
                AC_Args.env_factor_encoder_branch_latent_dims,
            )
        ):
            # Env factor encoder
            env_factor_encoder_layers = []
            env_factor_encoder_layers.append(nn.Linear(branch_input_dim, branch_hidden_dims[0]))
            env_factor_encoder_layers.append(activation)
            for l in range(len(branch_hidden_dims)):
                if l == len(branch_hidden_dims) - 1:
                    env_factor_encoder_layers.append(
                        nn.Linear(branch_hidden_dims[l], branch_latent_dim)
                    )
                else:
                    env_factor_encoder_layers.append(
                        nn.Linear(branch_hidden_dims[l], branch_hidden_dims[l + 1])
                    )
                    env_factor_encoder_layers.append(activation)
        self.env_factor_encoder = nn.Sequential(*env_factor_encoder_layers)
        self.add_module(f"encoder", self.env_factor_encoder)

        # Adaptation module
        for i, (branch_hidden_dims, branch_latent_dim) in enumerate(
            zip(
                AC_Args.adaptation_module_branch_hidden_dims,
                AC_Args.env_factor_encoder_branch_latent_dims,
            )
        ):
            adaptation_module_layers = []
            adaptation_module_layers.append(nn.Linear(num_obs_history, branch_hidden_dims[0]))
            adaptation_module_layers.append(activation)
            for l in range(len(branch_hidden_dims)):
                if l == len(branch_hidden_dims) - 1:
                    adaptation_module_layers.append(
                        nn.Linear(branch_hidden_dims[l], branch_latent_dim)
                    )
                else:
                    adaptation_module_layers.append(
                        nn.Linear(branch_hidden_dims[l], branch_hidden_dims[l + 1])
                    )
                    adaptation_module_layers.append(activation)
        self.adaptation_module = nn.Sequential(*adaptation_module_layers)
        self.add_module(f"adaptation_module", self.adaptation_module)

        total_latent_dim = int(
            torch.sum(torch.Tensor(AC_Args.env_factor_encoder_branch_latent_dims))
        )

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(total_latent_dim + num_obs, AC_Args.actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(AC_Args.actor_hidden_dims)):
            if l == len(AC_Args.actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(AC_Args.actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(
                    nn.Linear(AC_Args.actor_hidden_dims[l], AC_Args.actor_hidden_dims[l + 1])
                )
                actor_layers.append(activation)
        self.actor_body = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(
            nn.Linear(total_latent_dim + num_obs, AC_Args.critic_hidden_dims[0])
        )
        critic_layers.append(activation)
        for l in range(len(AC_Args.critic_hidden_dims)):
            if l == len(AC_Args.critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(AC_Args.critic_hidden_dims[l], 1))
            else:
                critic_layers.append(
                    nn.Linear(AC_Args.critic_hidden_dims[l], AC_Args.critic_hidden_dims[l + 1])
                )
                critic_layers.append(activation)
        self.critic_body = nn.Sequential(*critic_layers)

        print(f"Environment Factor Encoder: {self.env_factor_encoder}")
        print(f"Adaptation Module: {self.adaptation_module}")
        print(f"Actor MLP: {self.actor_body}")
        print(f"Critic MLP: {self.critic_body}")

        # Action noise
        self.std = nn.Parameter(AC_Args.init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations, privileged_observations):
        latent = self.env_factor_encoder(privileged_observations)
        mean = self.actor_body(jnp.concatenate((observations, latent), axis=-1))
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, privileged_observations, **kwargs):
        self.update_distribution(observations, privileged_observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_student(self, observations, observation_history):
        latent = self.adaptation_module(observation_history)
        actions_mean = self.actor_body(jnp.concatenate((observations, latent), axis=-1))
        return actions_mean

    def act_teacher(self, observations, privileged_info):
        latent = self.env_factor_encoder(privileged_info)
        actions_mean = self.actor_body(jnp.concatenate((observations, latent), axis=-1))
        return actions_mean

    def evaluate(self, critic_observations, privileged_observations):
        latent = self.env_factor_encoder(privileged_observations)
        value = self.critic_body(jnp.concatenate((critic_observations, latent), axis=-1))
        return value


def get_activation(act_name):
    if act_name == "elu":
        return nn.activation.relu
    elif act_name == "selu":
        return nn.activation.selu
    elif act_name == "relu":
        return nn.activation.relu
    elif act_name == "lrelu":
        return nn.activation.leaky_relu
    elif act_name == "tanh":
        return nn.activation.tanh
    elif act_name == "sigmoid":
        return nn.activation.sigmoid
    else:
        print("invalid activation function!")
        return None
