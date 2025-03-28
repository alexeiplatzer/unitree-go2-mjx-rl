from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    name: str
    modules: dict[str, list[int]]


@dataclass
class ActorCriticConfig(ModelConfig):
    @dataclass
    class ActorCriticModulesConfig:
        policy: list[int] = field(default_factory=lambda: [256, 256])
        value: list[int] = field(default_factory=lambda: [256, 256])

    name: str = "actor_critic"
    modules: ActorCriticModulesConfig = field(default_factory=ActorCriticModulesConfig)


@dataclass
class TeacherStudentConfig(ActorCriticConfig):
    @dataclass
    class TeacherStudentModulesConfig(ActorCriticConfig.ActorCriticModulesConfig):
        encoder: list[int] = field(default_factory=lambda: [256, 256])
        adapter: list[int] = field(default_factory=lambda: [256, 256])

    name: str = "teacher_student"
    modules: TeacherStudentModulesConfig = field(default_factory=TeacherStudentModulesConfig)
    latent_size: int = 16


name_to_model = {
    "actor_critic": ActorCriticConfig,
    "teacher_student": TeacherStudentConfig,
}
