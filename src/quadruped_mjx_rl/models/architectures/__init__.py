from quadruped_mjx_rl.models.architectures.actor_critic_base import (
    ActorCriticAgentParams,
    ActorCriticConfig,
    ActorCriticNetworkParams,
    ActorCriticNetworks,
    make_actor_critic_networks,
)
from quadruped_mjx_rl.models.architectures.configs_base import (
    ModelConfig,
    register_model_config_class,
)
from quadruped_mjx_rl.models.architectures.teacher_student_base import (
    TeacherStudentAgentParams,
    TeacherStudentConfig,
    TeacherStudentNetworkParams,
    TeacherStudentNetworks,
    make_teacher_student_networks,
)
from quadruped_mjx_rl.models.architectures.teacher_student_recurrent import (
    TeacherStudentRecurrentConfig,
    make_teacher_student_recurrent_networks,
)
from quadruped_mjx_rl.models.architectures.teacher_student_vision import (
    TeacherStudentVisionConfig,
    make_teacher_student_vision_networks,
)
