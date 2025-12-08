from quadruped_mjx_rl.models import (
    ActorCriticConfig,
    ModelConfig,
    TeacherStudentConfig,
    TeacherStudentRecurrentConfig,
)
from quadruped_mjx_rl.models.architectures import ActorCriticNetworkParams
from quadruped_mjx_rl.training.fitting.optimization import (
    Fitter,
    OptimizerState,
    SimpleFitter,
)
from quadruped_mjx_rl.training.fitting.teacher_student import TeacherStudentFitter
from quadruped_mjx_rl.training.fitting.recurrent_student import RecurrentStudentFitter


def get_fitter(model_config: ModelConfig) -> type[Fitter[ActorCriticNetworkParams]]:
    if isinstance(model_config, TeacherStudentRecurrentConfig):
        return RecurrentStudentFitter
    elif isinstance(model_config, TeacherStudentConfig):
        return TeacherStudentFitter
    elif isinstance(model_config, ActorCriticConfig):
        return SimpleFitter
    else:
        raise ValueError(f"Unknown model config type: {type(model_config)}")
