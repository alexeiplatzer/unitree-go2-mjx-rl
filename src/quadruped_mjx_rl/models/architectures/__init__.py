from quadruped_mjx_rl.models.architectures.actor_critic_base import (
    ActorCriticAgentParams,
    ActorCriticConfig,
    ActorCriticNetworkParams,
    ActorCriticNetworks,
)
from quadruped_mjx_rl.models.architectures.configs_base import (
    ModelConfig,
    register_model_config_class,
)
from quadruped_mjx_rl.models.architectures.actor_critic_enriched import (
    ActorCriticEnrichedConfig,
    ActorCriticEnrichedNetworkParams,
    ActorCriticEnrichedAgentParams,
    ActorCriticEnrichedNetworks,
    ActorCriticMixedModeConfig,
)
from quadruped_mjx_rl.models.architectures.teacher_student_base import (
    TeacherStudentAgentParams,
    TeacherStudentConfig,
    TeacherStudentVisionConfig,
    TeacherStudentNetworkParams,
    TeacherStudentNetworks,
    TeacherStudentMixedModeConfig,
)
from quadruped_mjx_rl.models.architectures.teacher_student_recurrent import (
    TeacherStudentRecurrentConfig,
    TeacherStudentRecurrentNetworks,
)
