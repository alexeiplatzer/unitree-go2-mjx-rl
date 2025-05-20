from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    modules: dict[str, list[int]]
    model_class: str


@dataclass
class ActorCriticConfig(ModelConfig):
    @dataclass
    class ActorCriticModulesConfig:
        policy: list[int] = field(default_factory=lambda: [256, 256])
        value: list[int] = field(default_factory=lambda: [256, 256])

    modules: ActorCriticModulesConfig = field(default_factory=ActorCriticModulesConfig)
    model_class: str = "ActorCritic"


@dataclass
class TeacherStudentConfig(ActorCriticConfig):
    @dataclass
    class TeacherStudentModulesConfig(ActorCriticConfig.ActorCriticModulesConfig):
        encoder: list[int] = field(default_factory=lambda: [256, 256])
        adapter: list[int] = field(default_factory=lambda: [256, 256])

    modules: TeacherStudentModulesConfig = field(default_factory=TeacherStudentModulesConfig)
    latent_size: int = 16
    model_class: str = "TeacherStudent"


@dataclass
class TeacherStudentVisionConfig(TeacherStudentConfig):
    @dataclass
    class TeacherStudentVisionModulesConfig(ActorCriticConfig.ActorCriticModulesConfig):
        encoder_convolutional: list[int] = field(default_factory=lambda: [32, 64, 64])
        encoder_dense: list[int] = field(default_factory=lambda: [256, 256])
        adapter_convolutional: list[int] = field(default_factory=lambda: [32, 64, 64])
        adapter_dense: list[int] = field(default_factory=lambda: [256, 256])

    modules: TeacherStudentVisionModulesConfig = field(
        default_factory=TeacherStudentVisionModulesConfig
    )
    model_class: str = "TeacherStudentVision"


model_config_classes = {
    "ActorCritic": ActorCriticConfig,
    "TeacherStudent": TeacherStudentConfig,
    "TeacherStudentVision": TeacherStudentVisionConfig,
    "default": ActorCriticConfig,
}
