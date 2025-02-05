import yaml
from ml_collections.config_dict import ConfigDict


def load_config_dicts(*args):
    for path in args:
        with open(path) as f:
            yield ConfigDict(yaml.safe_load(f))
