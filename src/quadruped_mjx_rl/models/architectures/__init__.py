from quadruped_mjx_rl.models.architectures.actor_critic_base import (
    ActorCriticAgentParams,
    ActorCriticConfig,
    ActorCriticNetworkParams,
    ActorCriticAgent,
    ActorCriticNetworks,
)
from quadruped_mjx_rl.models.architectures.configs_base import (
    ModelConfig,
    register_model_config_class,
)
from quadruped_mjx_rl.models.architectures.teacher_student_base import (
    TeacherStudentAgentParams,
    TeacherStudentConfig,
    TeacherStudentNetworkParams,
    TeacherStudentAgent,
    TeacherStudentNetworks,
)
from quadruped_mjx_rl.models.architectures.teacher_student_recurrent import (
    TeacherStudentRecurrentConfig,
    TeacherStudentRecurrentNetworks,
    TeacherStudentRecurrentAgent,
)
from quadruped_mjx_rl.models.architectures.teacher_student_vision import (
    TeacherStudentVisionConfig,
    TeacherStudentVisionNetworks,
    TeacherStudentVisionAgent,
)
