import jax
from jax import numpy as jnp
import mujoco

from quadruped_mjx_rl.environments.physics_pipeline import (
    EnvModel,
    PipelineModel,
)


def randomize(
    pipeline_model: PipelineModel,
    rng: jax.Array,
):
    in_axes = jax.tree.map(lambda x: None, pipeline_model)
    return pipeline_model, in_axes
