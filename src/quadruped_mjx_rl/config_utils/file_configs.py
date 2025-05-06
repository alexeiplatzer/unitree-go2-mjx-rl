from collections.abc import Callable, Iterator
from dataclasses import asdict

import yaml
from dacite import from_dict
from etils.epath import PathLike

from quadruped_mjx_rl.config_utils.config_keys import AnyConfig, ConfigKey
from quadruped_mjx_rl.config_utils.config_keys import config_class_to_key
from quadruped_mjx_rl.config_utils.config_keys import key_to_resolver


def prepare_configs(
    *config_paths: PathLike,
    check_configs: list[type[AnyConfig] | ConfigKey | str] | None = None,
    **configs: AnyConfig | None,
) -> dict[ConfigKey, AnyConfig | None]:
    """
    Prepares All configs from provided files and config objects.
    :param config_paths: Paths of YAML files to search configs in.
    :param check_configs: A list of configs that must be present, throws an exception if not.
    :param configs: Manually specified configs to override configs from the files if present.
    :return: A dict containing the final config for every Config Key.
    """
    loaded_dicts = load_config_dicts(*config_paths)
    loaded_configs = load_configs_from_dicts(list(ConfigKey), *loaded_dicts)
    final_configs = {}
    for config_key in ConfigKey:
        if config_key in configs:
            final_configs[config_key] = configs[config_key]
        elif loaded_configs[config_key] is not None:
            final_configs[config_key] = config_from_dict(config_key, loaded_configs[config_key])
        else:
            final_configs[config_key] = None
        if final_configs[config_key] is None and config_key in check_configs:
            raise RuntimeError(f"Config for {config_key} not provided")
    return final_configs


def config_from_dict(
    config_key: ConfigKey,
    loaded_config: dict,
) -> AnyConfig:
    """
    Creates a config of the correct class from a given dict
    """
    config_class_name = loaded_config.get(f"{config_key.value}_class", "default")
    config_class = key_to_resolver[config_key][config_class_name]
    return from_dict(config_class, loaded_config)


def load_configs_from_dicts(keywords: list[str], *loaded_configs: dict) -> dict[str, dict]:
    """
    Looks for entries under the provided keywords at the top level in each loaded dict,
    then stores them all in a dict entry under this keyword and returns the resulting dict.
    """
    keyword_to_config = {keyword: {} for keyword in keywords}
    for loaded_config in loaded_configs:
        for keyword in keywords:
            keyword_to_config[keyword] |= loaded_config.get(keyword, {})
    return keyword_to_config


def load_config_dicts(
    *args: PathLike, map_dicts: Callable[[dict], ...] = None
) -> Iterator[dict]:
    """
    Loads YAML files into dicts, applies a map function if provided.
    """
    for path in args:
        with open(path) as f:
            loaded_dict = yaml.safe_load(f)
            yield loaded_dict if map_dicts is None else map_dicts(loaded_dict)


def save_configs(
    save_file_path: PathLike,
    *configs: AnyConfig,
):
    """
    Saves all configs to a YAML file, each config under an appropriate top level key.
    """
    final_dict = {
        config_class_to_key(type(config)).value: asdict(config)
        for config in configs
    }
    with open(save_file_path, "w") as f:
        yaml.dump(final_dict, f)
