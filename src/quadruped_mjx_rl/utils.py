import yaml
from enum import StrEnum
from dacite import from_dict
from dataclasses import asdict

from etils.epath import PathLike
from collections.abc import Iterator, Mapping, Callable


def conditionally_instantiate[Configuration](
    name_to_factory: Mapping[str, Callable[[], Configuration]],
    passed_config: Configuration | str | None,
    loaded_config: dict | None = None,
) -> Configuration | None:
    """
    Loads a configuration object from given loaded dicts, overwriting with the argument config
    if present.
    """

    if not isinstance(passed_config, str) and passed_config is not None:
        return passed_config
    name = passed_config if isinstance(passed_config, str) else loaded_config.get("name", None)
    if name is None:
        return None
    default_config = name_to_factory[name]()
    final_config = asdict(default_config) | loaded_config
    final_config["name"] = name
    return from_dict(type(default_config), final_config)


def load_configs_from_dicts(keywords: list[str], *loaded_configs: dict):
    keyword_to_config = {keyword: {} for keyword in keywords}
    for loaded_config in loaded_configs:
        for keyword in keywords:
            keyword_to_config[keyword] |= loaded_config.get(keyword, {})
    return keyword_to_config


def load_config_dicts(*args: PathLike, map_dicts: Callable = None) -> Iterator[dict]:
    """
    Loads yaml files into dicts, applies a map function if provided.
    :param args:
    :param map_dicts:
    :return:
    """
    for path in args:
        with open(path) as f:
            loaded_dict = yaml.safe_load(f)
            yield loaded_dict if map_dicts is None else map_dicts(loaded_dict)
