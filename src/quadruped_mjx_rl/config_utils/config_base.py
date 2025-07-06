from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from dacite import from_dict


@dataclass
class Configuration(ABC):

    @classmethod
    @abstractmethod
    def config_base_class_key(cls) -> str:
        pass

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Configuration":
        return from_dict(cls, config_dict)

    def to_dict(self) -> dict:
        return asdict(self)


_key_to_config_base_class: dict[str, type[Configuration]] = {}


def register_config_base_class(config_base_class: type[Configuration]):
    _key_to_config_base_class[config_base_class.config_base_class_key()] = config_base_class


def configs_from_dicts(config_dicts: dict[str, dict]) -> dict[str, Configuration]:
    return {
        key: _key_to_config_base_class[key].from_dict(config_dict)
        for key, config_dict in config_dicts.items()
    }
