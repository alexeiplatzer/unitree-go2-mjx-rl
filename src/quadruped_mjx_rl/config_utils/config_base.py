from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from collections.abc import Callable
from dacite import from_dict


@dataclass
class Configuration(ABC):

    @classmethod
    @abstractmethod
    def config_base_class_key(cls) -> str:
        pass

    @classmethod
    def config_class_key(cls) -> str:
        return "default"

    @classmethod
    def _get_config_class_dict(cls) -> dict[str, type["Configuration"]]:
        return {cls.config_class_key(): cls}

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Configuration":
        config_class_key = config_dict.pop(
            f"{cls.config_base_class_key()}_class", cls.config_class_key()
        )
        config_class = cls._get_config_class_dict()[config_class_key]
        return from_dict(config_class, config_dict)

    def to_dict(self) -> dict:
        config_dict = asdict(self)
        config_dict[f"{self.config_base_class_key()}_class"] = type(self).config_class_key()
        return config_dict

    @classmethod
    def make_register_config_class(cls) -> Callable[[type["Configuration"]], None]:
        def register_config_class(config_class: type["Configuration"]) -> None:
            cls._get_config_class_dict()[config_class.config_class_key()] = config_class

        return register_config_class


_key_to_config_base_class: dict[str, type[Configuration]] = {}


def register_config_base_class(config_base_class: type[Configuration]):
    _key_to_config_base_class[config_base_class.config_base_class_key()] = config_base_class


def configs_from_dicts(config_dicts: dict[str, dict]) -> dict[str, Configuration]:
    return {
        key: _key_to_config_base_class[key].from_dict(config_dict)
        for key, config_dict in config_dicts.items()
    }
