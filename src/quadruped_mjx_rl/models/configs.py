from dataclasses import dataclass, field

from quadruped_mjx_rl.config_utils import Configuration, register_config_base_class


@dataclass
class ModelConfig(Configuration):
    modules: dict[str, list[int]]

    @classmethod
    def config_base_class_key(cls) -> str:
        return "model"

    @classmethod
    def model_class_key(cls) -> str:
        return "custom"

    @classmethod
    def from_dict(cls, config_dict: dict) -> Configuration:
        model_class_key = config_dict.pop("model_class")
        model_config_class = _model_config_classes[model_class_key]
        return super(Configuration, model_config_class).from_dict(config_dict)

    def to_dict(self) -> dict:
        config_dict = super().to_dict()
        config_dict["model_class"] = type(self).model_class_key()
        return config_dict


register_config_base_class(ModelConfig)


@dataclass
class ActorCriticConfig(ModelConfig):
    @dataclass
    class ModulesConfig:
        policy: list[int] = field(default_factory=lambda: [256, 256])
        value: list[int] = field(default_factory=lambda: [256, 256])

    modules: ModulesConfig = field(default_factory=ModulesConfig)

    @classmethod
    def model_class_key(cls) -> str:
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
    def model_class_key(cls) -> str:
        return "TeacherStudent"


@dataclass
class TeacherStudentVisionConfig(TeacherStudentConfig):
    @dataclass
    class ModulesConfig(ActorCriticConfig.ModulesConfig):
        encoder_convolutional: list[int] = field(default_factory=lambda: [32, 64, 64])
        encoder_dense: list[int] = field(default_factory=lambda: [256, 256])
        adapter_convolutional: list[int] = field(default_factory=lambda: [32, 64, 64])
        adapter_dense: list[int] = field(default_factory=lambda: [256, 256])

    modules: ModulesConfig = field(
        default_factory=ModulesConfig
    )

    @classmethod
    def model_class_key(cls) -> str:
        return "TeacherStudentVision"


_model_config_classes = {
    config.model_class_key(): config for config in (
        ModelConfig, ActorCriticConfig, TeacherStudentConfig, TeacherStudentVisionConfig
    )
}
