# Typing
from dataclasses import dataclass, field

# Configurations
from quadruped_mjx_rl.config_utils import Configuration, register_config_base_class
from quadruped_mjx_rl.models.architectures import OptimizerConfig, TeacherStudentOptimizer


@dataclass
class AlgorithmHyperparams:
    pass


@dataclass
class HyperparamsPPO(AlgorithmHyperparams):
    discounting: float = 0.97
    entropy_cost: float = 0.01
    clipping_epsilon: float = 0.3
    gae_lambda: float = 0.95
    normalize_advantage: bool = True
    reward_scaling: int = 1


@dataclass
class TrainingConfig(Configuration):
    num_envs: int = 8192
    num_eval_envs: int = 8192
    num_timesteps: int = 100_000_000
    num_evals: int = 10
    episode_length: int = 1000
    unroll_length: int = 20
    normalize_observations: bool = True
    action_repeat: int = 1
    batch_size: int = 256
    num_updates_per_batch: int = 4
    num_minibatches: int = 32
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    rl_hyperparams: AlgorithmHyperparams = field(default_factory=HyperparamsPPO)

    @classmethod
    def config_base_class_key(cls) -> str:
        return "training"

    @classmethod
    def config_class_key(cls) -> str:
        return "PPO"

    @classmethod
    def _get_config_class_dict(cls) -> dict[str, type["Configuration"]]:
        return _training_config_classes


register_config_base_class(TrainingConfig)


@dataclass
class TrainingWithVisionConfig(TrainingConfig):
    madrona_backend: bool = True
    num_envs: int = 256
    num_eval_envs: int = 256
    num_timesteps: int = 1_000_000
    batch_size: int = 256
    num_updates_per_batch: int = 8
    optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(max_grad_norm=1.0)
    )

    @classmethod
    def config_class_key(cls) -> str:
        return "PPO_Vision"


_training_config_classes = {
    "default": TrainingConfig,
    "PPO": TrainingConfig,
    "PPO_Vision": TrainingWithVisionConfig,
}
