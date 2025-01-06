from dataclasses import dataclass, asdict
from typing import Any

from ml_collections import ConfigDict

import jax
import jax.numpy as jnp

from brax import envs
from brax import math
from brax.base import System
from brax.base import State as PipelineState
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf

# from .paths import SCENE_PATH


@dataclass
class Go2TeacherEnvConfig:
    # model_path: str = SCENE_PATH.as_posix()
    dt: float = 0.02
    timestep: float = 0.002


my_config = Go2TeacherEnvConfig(dt=0.02, timestep=0.002)

my_dict = asdict(my_config)

print(my_dict)
