from typing import Protocol

from dataclasses import dataclass


@dataclass
class TrainingConfig(Protocol):
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
