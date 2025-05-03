from dataclasses import dataclass


@dataclass
class TrainingConfig:
    num_timesteps: int = 100_000_000
    num_evals: int = 10
    reward_scaling: int = 1
    episode_length: int = 1000
    normalize_observations: bool = True
    action_repeat: int = 1
    unroll_length: int = 20
    num_minibatches: int = 32
    num_updates_per_batch: int = 4
    discounting: float = 0.97
    learning_rate: float = 0.0004
    entropy_cost: float = 0.01
    num_envs: int = 8192
    batch_size: int = 256
    training_class: str = "PPO"


@dataclass
class TrainingWithVisionConfig(TrainingConfig):
    num_timesteps: int = 1_000_000
    num_evals: int = 5
    reward_scaling: int = 1
    episode_length: int = 1000
    normalize_observations: bool = True
    action_repeat: int = 1
    unroll_length: int = 10
    num_minibatches: int = 8
    num_updates_per_batch: int = 8
    discounting: float = 0.97
    learning_rate: float = 0.0005
    entropy_cost: float = 0.005
    num_envs: int = 512
    batch_size: int = 256
    training_class: str = "PPO_Vision"

    madrona_backend: bool = True
    # wrap_env: bool = False
    num_eval_envs: int = 512
    max_grad_norm: float = 1.0
    # num_resets_per_eval: int = 1


training_config_classes = {
    "PPO": TrainingConfig,
    "PPO_Vision": TrainingWithVisionConfig,
}
