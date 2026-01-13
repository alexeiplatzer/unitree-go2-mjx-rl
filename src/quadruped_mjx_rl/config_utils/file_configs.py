from collections.abc import Callable, Iterator
from typing import TypeVar

import yaml
from etils.epath import PathLike

from quadruped_mjx_rl.config_utils.config_base import Configuration, configs_from_dicts


def prepare_configs(
    *config_paths: PathLike,
    check_configs: list[type[Configuration] | str] | None = None,
    **configs: Configuration,
) -> dict[str, Configuration]:
    """
    Prepares All configs from provided files and config objects.
    :param config_paths: Paths of YAML files to search configs in.
    :param check_configs: A list of configs that must be present, throws an exception if not.
    :param configs: Manually specified configs to override configs from the files if present.
    :return: A dict containing the final config for every Config Key.
    """
    loaded_dicts = load_config_dicts(*config_paths)
    loaded_configs = assemble_configs_from_dicts(*loaded_dicts)
    final_configs = configs_from_dicts(loaded_configs)
    for config_key, configuration in configs.items():
        final_configs[config_key] = configuration
    check_configs = check_configs or []
    for config_key in check_configs:
        if config_key not in final_configs:
            raise RuntimeError(f"Config for {config_key} not provided")
    return final_configs


def assemble_configs_from_dicts(*loaded_configs: dict) -> dict[str, dict]:
    """
    Looks for entries under the provided keywords at the top level in each loaded dict,
    then stores them all in a dict entry under this keyword and returns the resulting dict.
    """
    keyword_to_config = {}
    for loaded_config in loaded_configs:
        for keyword in loaded_config:
            keyword_to_config[keyword] = loaded_config[keyword] | keyword_to_config.get(
                keyword, {}
            )
    return keyword_to_config


MapDictsCodomain = TypeVar("MapDictsCodomain")


def load_config_dicts(
    *args: PathLike, map_dicts: Callable[[dict], MapDictsCodomain] = lambda d: d
) -> Iterator[MapDictsCodomain]:
    """
    Loads YAML files into dicts, applies a map function if provided.
    """
    for path in args:
        with open(path) as f:
            loaded_dict = yaml.safe_load(f)
            yield loaded_dict if map_dicts is None else map_dicts(loaded_dict)


def save_configs(
    save_file_path: PathLike,
    *configs: Configuration,
):
    """
    Saves all configs to a YAML file, each config under an appropriate top level key.
    """
    final_dict = {
        config.config_base_class_key(): config.to_dict()
        for config in configs if config is not None
    }
    with open(save_file_path, "w") as f:
        yaml.dump(final_dict, f)
