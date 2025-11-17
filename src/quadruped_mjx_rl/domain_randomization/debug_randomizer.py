import jax
from jax import numpy as jnp
import mujoco

from quadruped_mjx_rl.types import PRNGKey
from quadruped_mjx_rl.environments.physics_pipeline import (
    EnvModel,
    PipelineModel,
)


def randomize_minimal(
    pipeline_model: PipelineModel,
    env_model: EnvModel,
    rng_key: PRNGKey,
    num_worlds: int,
):
    in_axes = jax.tree.map(lambda x: None, pipeline_model)
    return pipeline_model, in_axes
