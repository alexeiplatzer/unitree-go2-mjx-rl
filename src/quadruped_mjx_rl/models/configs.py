from dataclasses import dataclass

from typing import Protocol


@dataclass
class ModelConfig:
    @dataclass
    class ModuleConfig:
        name: str
        hidden_layers: list[int]

    name: str
    modules: list[ModuleConfig]


@dataclass
class ActorCriticConfig:
    policy_hidden_layers: list[int]
    value_hidden_layers: list[int]


def actor_critic_model_config(actor_critic_config: ActorCriticConfig) -> ModelConfig:
    return ModelConfig(
        name="actor_critic",
        modules=[
            ModelConfig.ModuleConfig(
                name="policy",
                hidden_layers=actor_critic_config.policy_hidden_layers,
            ),
            ModelConfig.ModuleConfig(
                name="value",
                hidden_layers=actor_critic_config.value_hidden_layers,
            )
        ]
    )


@dataclass
class TeacherStudentConfig:
    actor_critic: ActorCriticConfig
    encoder_hidden_layers: list[int]
    adapter_hidden_layers: list[int]


def teacher_student_model_config(teacher_student_config: TeacherStudentConfig) -> ModelConfig:
    return ModelConfig(
        name="teacher_student",
        modules=[
            ModelConfig.ModuleConfig(
                name="encoder",
                hidden_layers=teacher_student_config.encoder_hidden_layers,
            ),
            ModelConfig.ModuleConfig(
                name="adapter",
                hidden_layers=teacher_student_config.adapter_hidden_layers,
            ),
            ModelConfig.ModuleConfig(
                name="policy",
                hidden_layers=teacher_student_config.actor_critic.policy_hidden_layers,
            ),
            ModelConfig.ModuleConfig(
                name="value",
                hidden_layers=teacher_student_config.actor_critic.value_hidden_layers,
            )
        ]
    )


# --- EXAMPLE VALUES --- #

def ppo_simple_config():
    return actor_critic_model_config(ActorCriticConfig(
        policy_hidden_layers=[8, 8],
        value_hidden_layers=[8, 8],
    ))


def ppo_teacher_student_config():
    return teacher_student_model_config(TeacherStudentConfig(
        actor_critic=ActorCriticConfig(
            policy_hidden_layers=[8, 8],
            value_hidden_layers=[8, 8],
        ),
        encoder_hidden_layers=[8, 8],
        adapter_hidden_layers=[8, 8],
    ))


name_to_model = {
    "actor_critic": ppo_simple_config,
    "teacher_student": ppo_teacher_student_config,
}
