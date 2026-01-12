import functools
from dataclasses import dataclass, field

import jax
from quadruped_mjx_rl.config_utils import Configuration, register_config_base_class
from quadruped_mjx_rl.models import ModelConfig
from quadruped_mjx_rl.robotic_vision import RendererConfig, get_renderer
from quadruped_mjx_rl.training.algorithms.ppo import HyperparamsPPO


@dataclass
class OptimizerConfig:
    learning_rate: float
    max_grad_norm: float | None

    @classmethod
    def default(cls) -> "OptimizerConfig":
        return OptimizerConfig(learning_rate=0.0002, max_grad_norm=None)


@dataclass
class TeacherStudentOptimizerConfig(OptimizerConfig):
    student_learning_rate: float

    @classmethod
    def default(cls) -> "TeacherStudentOptimizerConfig":
        return TeacherStudentOptimizerConfig(
            learning_rate=0.0002, max_grad_norm=None, student_learning_rate=0.001
        )


@dataclass
class VisionConfig:
    vision_frame_width: int = 64
    vision_frame_height: int = 64
    vision_obs_period: int = 16


@dataclass
class TrainingConfig(Configuration):
    augment_pixels: bool = False
    num_envs: int = 2048
    num_eval_envs: int = 2048
    seed: int = 0
    num_timesteps: int = 2**28
    log_training_metrics: bool = False
    training_metrics_steps: int | None = None
    num_evals: int = 11
    deterministic_eval: bool = False
    num_resets_per_eval: int = 0
    episode_length: int = 2048
    unroll_length: int = 32
    normalize_observations: bool = True
    action_repeat: int = 1
    batch_size: int = 64
    num_updates_per_batch: int = 4
    num_minibatches: int = 32
    optimizer: TeacherStudentOptimizerConfig | OptimizerConfig = field(
        default_factory=OptimizerConfig.default
    )
    rl_hyperparams: HyperparamsPPO = field(default_factory=HyperparamsPPO)
    vision_config: VisionConfig | None = None

    # @classmethod
    # def default_vision(cls) -> "TrainingConfig":
    #     return TrainingConfig(
    #         # num_envs=1024,
    #         # num_eval_envs=1024,
    #         # num_timesteps=2**27,
    #         # batch_size=32,
    #         # optimizer=OptimizerConfig(max_grad_norm=1.0),
    #         vision_config=VisionConfig(),
    #     )

    def check_validity(self, model_config: ModelConfig | None = None) -> None:
        if self.batch_size * self.num_minibatches % self.num_envs != 0:
            raise ValueError(
                f"Batch size ({self.batch_size}) times number of minibatches "
                f"({self.num_minibatches}) must be divisible by number of environments "
                f"({self.num_envs})."
            )
        if model_config is not None and model_config.vision:
            self.check_validity_vision(model_config)
        if model_config is not None and model_config.recurrent:
            self.check_validity_recurrent(model_config)

    def check_validity_vision(self, model_config: ModelConfig | None = None) -> None:
        """Validates arguments for Madrona-MJX."""
        if self.vision_config is None:
            raise ValueError("Vision config must be set if vision is desired.")
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
        if self.unroll_length % self.vision_config.vision_obs_period != 0:
            raise ValueError(
                f"Unroll length ({self.unroll_length}) must be divisible by vision observation "
                f"period ({self.vision_config.vision_obs_period})."
            )

    def check_validity_recurrent(self, model_config: ModelConfig | None) -> None:
        if self.batch_size * self.num_minibatches != self.num_envs:
            raise ValueError(
                "Total batch size must be equal to number of environments, for the consistency"
                "of sequential updates to the recurrent buffers."
            )

    def get_renderer_factory(self, gpu_id: int = 0, debug: bool = False):
        self.check_validity_vision()
        return functools.partial(
            get_renderer,
            renderer_config=RendererConfig(
                gpu_id=gpu_id,
                render_batch_size=self.num_envs // jax.local_device_count(),  # TODO: configure
                render_width=self.vision_config.vision_frame_width,
                render_height=self.vision_config.vision_frame_height,
                use_rasterizer=False,
                enabled_geom_groups=[0, 1, 2],
            ),
            debug=debug,
        )

    @classmethod
    def config_base_class_key(cls) -> str:
        return "training"


register_config_base_class(TrainingConfig)
