import functools
from dataclasses import dataclass, field

import jax

from quadruped_mjx_rl.config_utils import Configuration, register_config_base_class
from quadruped_mjx_rl.robotic_vision import RendererConfig, get_renderer
from quadruped_mjx_rl.training.algorithms.ppo import HyperparamsPPO


@dataclass
class OptimizerConfig:
    learning_rate: float = 0.0004
    max_grad_norm: float | None = None


@dataclass
class TeacherStudentOptimizerConfig(OptimizerConfig):
    student_learning_rate: float = 0.0004  # currently only supported for vision


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

    def check_validity(self):
        if self.batch_size * self.num_minibatches % self.num_envs != 0:
            raise ValueError(
                f"Batch size ({self.batch_size}) times number of minibatches "
                f"({self.num_minibatches}) must be divisible by number of environments "
                f"({self.num_envs})."
            )

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
    use_vision: bool = True
    augment_pixels: bool = False
    num_envs: int = 256  # Most of these int args are reduced because vision is heavier
    num_eval_envs: int = 256
    batch_size: int = 16
    num_minibatches: int = 32
    num_updates_per_batch: int = 4
    unroll_length: int = 4
    episode_length: int = 50
    num_timesteps: int = 10_000_000
    optimizer: TeacherStudentOptimizerConfig = field(
        default_factory=lambda: TeacherStudentOptimizerConfig(
            max_grad_norm=1.0,
            student_learning_rate=0.001,
        )
    )
    vision_frame_width: int = 64
    vision_frame_height: int = 64

    def get_renderer_factory(self, gpu_id: int = 0, debug: bool = False):
        return functools.partial(
            get_renderer,
            renderer_config=RendererConfig(
                gpu_id=gpu_id,
                render_batch_size=self.num_envs // jax.local_device_count(),  # TODO: configure
                render_width=self.vision_frame_width,
                render_height=self.vision_frame_height,
                use_rasterizer=False,
                enabled_geom_groups=[0, 1, 2],
            ),
            debug=debug,
        )

    def check_validity(self):
        super().check_validity()
        """Validates arguments for Madrona-MJX."""
        if self.use_vision:
            if self.num_eval_envs != self.num_envs:
                raise ValueError(
                    "Number of eval envs != number of training envs. The Madrona-MJX vision "
                    "backend requires a fixed batch size, the number of environments must be "
                    "consistent."
                )
            if self.action_repeat != 1:
                raise ValueError(
                    "Implement action_repeat using PipelineEnv's _n_frames to avoid unnecessary"
                    " rendering!"
                )

    @classmethod
    def config_class_key(cls) -> str:
        return "PPO_Vision"


@dataclass
class TrainingWithRecurrentStudentConfig(TrainingWithVisionConfig):
    unroll_length: int = 4
    num_updates_per_batch: int = 2
    num_minibatches: int = 16

    def check_validity(self):
        super().check_validity()
        # if self.num_updates_per_batch != 1:
        #     raise ValueError(
        #         "Recurrent training must have num_updates_per_batch == 1. Repeated updates on "
        #         "the same batch will break the recurrent buffers."
        #     )
        if self.batch_size * self.num_minibatches != self.num_envs:
            raise ValueError(
                "Total batch size must be equal to number of environments, for the consistency"
                "of sequential updates to the recurrent buffers."
            )
        # if (
        #     self.unroll_length
        #     % (self.vision_steps_per_recurrent_step * self.proprio_steps_per_vision_step)
        #     != 0
        # ):
        #     raise ValueError(
        #         "Unroll length must be divisible by the product of vision steps per recurrent "
        #         "step and proprio steps per vision step."
        #     )

    @classmethod
    def config_class_key(cls) -> str:
        return "PPO_RecurrentStudent"


_training_config_classes = {
    "default": TrainingConfig,
    "PPO": TrainingConfig,
    "PPO_Vision": TrainingWithVisionConfig,
    "PPO_RecurrentStudent": TrainingWithRecurrentStudentConfig,
}
