from dataclasses import dataclass, field

from quadruped_mjx_rl.config_utils import Configuration, register_config_base_class


@dataclass
class ModelConfig(Configuration):
    modules: dict[str, list[int]]

    @classmethod
    def config_base_class_key(cls) -> str:
        return "model"

    @classmethod
    def config_class_key(cls) -> str:
        return "custom"

    @classmethod
    def _get_config_class_dict(cls) -> dict[str, type["Configuration"]]:
        return _model_config_classes


register_config_base_class(ModelConfig)


@dataclass
class ActorCriticConfig(ModelConfig):
    @dataclass
    class ModulesConfig:
        policy: list[int] = field(default_factory=lambda: [256, 256])
        value: list[int] = field(default_factory=lambda: [256, 256])

    modules: ModulesConfig = field(default_factory=ModulesConfig)

    @classmethod
    def config_class_key(cls) -> str:
        return "ActorCritic"


@dataclass
class TeacherStudentConfig(ActorCriticConfig):
    @dataclass
    class ModulesConfig(ActorCriticConfig.ModulesConfig):
        encoder: list[int] = field(default_factory=lambda: [256, 256])
        adapter: list[int] = field(default_factory=lambda: [256, 256])

    modules: ModulesConfig = field(default_factory=ModulesConfig)
    latent_size: int = 16

    @classmethod
    def config_class_key(cls) -> str:
        return "TeacherStudent"


@dataclass
class TeacherStudentVisionConfig(TeacherStudentConfig):
    @dataclass
    class ModulesConfig(ActorCriticConfig.ModulesConfig):
        encoder_convolutional: list[int] = field(default_factory=lambda: [32, 64, 64])
        encoder_dense: list[int] = field(default_factory=lambda: [256, 256])
        adapter_convolutional: list[int] = field(default_factory=lambda: [32, 64, 64])
        adapter_dense: list[int] = field(default_factory=lambda: [256, 256])

    modules: ModulesConfig = field(default_factory=ModulesConfig)

    @classmethod
    def config_class_key(cls) -> str:
        return "TeacherStudentVision"


@dataclass
class TeacherStudentRecurrentConfig(TeacherStudentConfig):
    @dataclass
    class ModulesConfig(ActorCriticConfig.ModulesConfig):
        encoder_convolutional: list[int] = field(default_factory=lambda: [16, 16, 16])
        adapter_convolutional: list[int] = field(default_factory=lambda: [32, 32, 32])
        adapter_recurrent_size: int = 16
        adapter_dense: list[int] = field(default_factory=lambda: [16])

    modules: ModulesConfig = field(default_factory=ModulesConfig)

    @classmethod
    def config_class_key(cls) -> str:
        return "TeacherStudentRecurrent"


_model_config_classes = {
    config.config_class_key(): config
    for config in (
        ModelConfig,
        ActorCriticConfig,
        TeacherStudentConfig,
        TeacherStudentVisionConfig,
        TeacherStudentRecurrentConfig,
    )
}
