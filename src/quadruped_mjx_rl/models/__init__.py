from quadruped_mjx_rl.models.architectures import (
    ActorCriticConfig,
    ActorCriticEnrichedConfig,
    ModelConfig,
    TeacherStudentConfig,
    TeacherStudentVisionConfig,
    TeacherStudentRecurrentConfig,
    TeacherStudentMixedModeConfig,
    ActorCriticMixedModeConfig,
)
from quadruped_mjx_rl.models.base_modules import (
    ModuleConfigMLP,
    ModuleConfigCNN,
    ModuleConfigLSTM,
    ModuleConfigMixedModeRNN,
)
from quadruped_mjx_rl.models.factories import get_networks_factory, load_inference_fn
from quadruped_mjx_rl.models.types import AgentParams
