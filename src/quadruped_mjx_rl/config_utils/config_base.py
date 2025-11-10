from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from collections.abc import Callable
from dacite import from_dict


@dataclass
class Configuration(ABC):
    """An abstract base class representing a configuration dataclass. Configurations for the
    neural network model and the reinforcement learning environment and other stuff can be kept
    as an instance of an appropriate subclass of this class. This class allows for easy
    storage and retrieval of configurations between dataclasses and YAML files.

    Each type of configuration should define a base class. Specific configurations for different
    use cases can then subclass from it. For example, a base class for an RL environment config
    can be defined, and then each different type of environment requiring different variables
    and data can have its own subclass. The base class allows interfacing with an abstract
    environment config where the specifics are not relevant after configuring. So it can be
    required that a YAML file contains configs for a base class, but the exact class can be
    left to be configured with the necessary data."""

    @classmethod
    @abstractmethod
    def config_base_class_key(cls) -> str:
        """A simple string defining the base class/type of the configuration, for example,
        'environment' or 'model'. When reading from YAML, this key will be searched for to find
        the relevant config."""
        pass

    @classmethod
    def config_class_key(cls) -> str:
        """The specific string describing what exact type/class of configuration an instance is
        supposed to be. This is used for instantiating the correct class when loading from YAML.
        """
        return "default"

    @classmethod
    def _get_config_class_dict(cls) -> dict[str, type["Configuration"]]:
        """Each base config class has its own dictionary of all subclasses indexed by their
        string keys. Helps find the correct subclass when the key is known and instantiate
        the correct type."""
        return {cls.config_class_key(): cls}

    @classmethod
    def get_config_class_from_key(cls, key: str) -> type["Configuration"]:
        """Uses the dictionary of subclasses to find the subclass correcponding to this key.
        See _get_config_class_dict for more details."""
        return cls._get_config_class_dict()[key]

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Configuration":
        """Creates an instance of the correct config class from a yaml-style dictionary of the
        data. Expects to find a string key describing which config class should be used."""
        config_class_key = config_dict.pop(
            f"{cls.config_base_class_key()}_class", cls.config_class_key()
        )
        config_class = cls.get_config_class_from_key(config_class_key)
        return from_dict(config_class, config_dict)

    def to_dict(self) -> dict:
        """Dumpts the config class instance into a dictionary of data. Adds a key describing
        the class of the config."""
        config_dict = asdict(self)
        config_dict[f"{self.config_base_class_key()}_class"] = type(self).config_class_key()
        return config_dict

    @classmethod
    def make_register_config_class(cls) -> Callable[[type["Configuration"]], None]:
        """Utility function for each top-level base config class for a type of config.
        The Configuration base class keeps itself a dictionary of all derived instances to help
        it process them when encountered in YAML files."""

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
