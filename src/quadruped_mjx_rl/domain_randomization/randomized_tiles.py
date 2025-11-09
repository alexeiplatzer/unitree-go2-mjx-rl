import jax
from jax import numpy as jnp
import mujoco

from quadruped_mjx_rl.environments.physics_pipeline import (
    EnvModel,
    PipelineModel,
)


def collect_tile_ids(env_model: EnvModel, tile_body_prefix: str = "tile_") -> jax.Array:
    """Collects the ids of geoms that belong to tile bodies.

    Args:
        env_model: Compiled environment model containing the tiled terrain.
        tile_body_prefix: Prefix that identifies bodies representing tiles.

    Returns:
        An array with the ids of all geoms that belong to bodies with the provided prefix.
    """

    geom_ids: list[int] = []
    for body_id in range(env_model.nbody):
        name = mujoco.mj_id2name(env_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if name is None or not name.startswith(tile_body_prefix):
            continue
        first_geom = env_model.body_geomadr[body_id]
        geom_ids.append(first_geom)

    if not geom_ids:
        raise ValueError(
            "Unable to locate any tile geoms. Ensure that the terrain builder "
            f"creates bodies with the prefix '{tile_body_prefix}'."
        )

    return jnp.array(geom_ids)


def randomize_tiles(
    pipeline_model: PipelineModel,
    rng: jax.Array,
    env_model: EnvModel,
    tile_body_prefix: str = "tile_",
):
    """Randomizes the mjx.Model. Assumes the ground is a square grid of square tiles."""
    tile_geom_ids = collect_tile_ids(env_model, tile_body_prefix)
    num_variants = 2
    rgbas = jnp.array([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]])
    friction_min = 0.6
    friction_max = 1.4
    solref_min = 0.002
    solref_max = 0.05

    @jax.vmap
    def rand(rng):
        key_tiles, key_friction, key_solref = jax.random.split(rng, 3)

        colour_friction = jax.random.uniform(
            key_friction,
            shape=(num_variants,),
            minval=friction_min,
            maxval=friction_max,
        )
        colour_solref = jax.random.uniform(
            key_solref,
            shape=(num_variants,),
            minval=solref_min,
            maxval=solref_max,
        )
        tile_colour_indices = jax.random.randint(
            key_tiles,
            shape=(tile_geom_ids.shape[0],),
            minval=0,
            maxval=num_variants,
        )

        chosen_colors = rgbas[tile_colour_indices]
        chosen_frictions = colour_friction[tile_colour_indices]
        chosen_solrefs = colour_solref[tile_colour_indices]

        geom_rgba = pipeline_model.model.geom_rgba.at[tile_geom_ids].set(chosen_colors)
        geom_friction = pipeline_model.model.geom_friction.at[tile_geom_ids, 0].set(
            chosen_frictions
        )
        geom_solref = pipeline_model.model.geom_solref.at[tile_geom_ids, 0].set(chosen_solrefs)

        return geom_rgba, geom_friction, geom_solref

    rgba, friction, solref = rand(rng)

    in_axes = jax.tree.map(lambda x: None, pipeline_model)
    in_axes = in_axes.replace(
        model=in_axes.model.tree_replace(
            {
                "geom_rgba": 0,
                "geom_friction": 0,
                "geom_solref": 0,
            }
        )
    )

    pipeline_model = pipeline_model.replace(
        model=pipeline_model.model.tree_replace(
            {
                "geom_rgba": rgba,
                "geom_friction": friction,
                "geom_solref": solref,
            }
        )
    )

    return pipeline_model, in_axes
