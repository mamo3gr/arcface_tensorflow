from pathlib import Path
from typing import Union

import yaml


def load_setting(yaml_path: Union[str, Path]):
    with open(yaml_path) as f:
        config = yaml.load(f)
    return config
