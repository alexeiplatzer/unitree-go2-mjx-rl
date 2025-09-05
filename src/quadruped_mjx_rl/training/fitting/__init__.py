from quadruped_mjx_rl.models import ActorCriticConfig, ModelConfig, TeacherStudentConfig
from quadruped_mjx_rl.training.fitting.optimization import (
    Fitter,
    OptimizerConfig,
    OptimizerState,
    SimpleFitter,
)
from quadruped_mjx_rl.training.fitting.teacher_student import TeacherStudentFitter


def get_fitter(model_config: ModelConfig) -> type[Fitter]:
    if isinstance(model_config, TeacherStudentConfig):
        return TeacherStudentFitter
    elif isinstance(model_config, ActorCriticConfig):
        return SimpleFitter
    else:
        raise ValueError(f"Unknown model config type: {type(model_config)}")
