from dataclasses import dataclass, field

from quadruped_mjx_rl.config_utils import Configuration, register_config_base_class
from quadruped_mjx_rl.training.algorithms.ppo import HyperparamsPPO


@dataclass
class OptimizerConfig:
    learning_rate: float = 0.0004
    max_grad_norm: float | None = None


@dataclass
class TeacherStudentOptimizerConfig(OptimizerConfig):
    student_learning_rate: float = 0.0004


@dataclass
class TrainingConfig(Configuration):
    use_vision: bool = False
    augment_pixels: bool = False
    num_envs: int = 8192
    num_eval_envs: int = 8192
    seed: int = 0
    num_timesteps: int = 100_000_000
    log_training_metrics: bool = False
    training_metrics_steps: int | None = None
    num_evals: int = 10
    deterministic_eval: bool = False
    num_resets_per_eval: int = 0
    episode_length: int = 1000
    unroll_length: int = 20
    normalize_observations: bool = True
    action_repeat: int = 1
    batch_size: int = 256
    num_updates_per_batch: int = 4
    num_minibatches: int = 32
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    rl_hyperparams: HyperparamsPPO = field(default_factory=HyperparamsPPO)

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
    proprio_steps_per_vision_step: int = 1
    use_vision: bool = True
    augment_pixels: bool = False
    num_envs: int = 256  # Most of these int args are reduced because vision is heavier
    num_eval_envs: int = 256  # Should be the same as above as we reuse them
    num_timesteps: int = 1_000_000
    batch_size: int = 256
    num_updates_per_batch: int = 8
    optimizer: TeacherStudentOptimizerConfig = field(
        default_factory=lambda: TeacherStudentOptimizerConfig(
            max_grad_norm=1.0,
            student_learning_rate=0.001,
        )
    )

    @classmethod
    def config_class_key(cls) -> str:
        return "PPO_Vision"


class TrainingWithRecurrentStudentConfig(TrainingWithVisionConfig):
    unroll_length: int = 25
    proprio_steps_per_vision_step: int = 5
    vision_steps_per_recurrent_step: int = 5

    @classmethod
    def config_class_key(cls) -> str:
        return "PPO_RecurrentStudent"


_training_config_classes = {
    "default": TrainingConfig,
    "PPO": TrainingConfig,
    "PPO_Vision": TrainingWithVisionConfig,
    "PPO_RecurrentStudent": TrainingWithRecurrentStudentConfig,
}
