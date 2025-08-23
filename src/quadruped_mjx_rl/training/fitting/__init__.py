from quadruped_mjx_rl.training.fitting.optimization import (
    OptimizerConfig,
    OptimizerState,
    Fitter,
    SimpleFitter,
)
from quadruped_mjx_rl.training.fitting.teacher_student import TeacherStudentFitter
from quadruped_mjx_rl.models import ModelConfig, ActorCriticConfig, TeacherStudentConfig


def get_fitter(model_config: ModelConfig) -> type[Fitter]:
    if isinstance(model_config, ActorCriticConfig):
        return SimpleFitter
    elif isinstance(model_config, TeacherStudentConfig):
        return TeacherStudentFitter
    else:
        raise ValueError(f"Unknown model config type: {type(model_config)}")