from dataclasses import dataclass

import jax
from jax import numpy as jnp
import mujoco

from quadruped_mjx_rl.types import PRNGKey
from quadruped_mjx_rl.physics_pipeline import (
    EnvModel,
    PipelineModel,
)
from quadruped_mjx_rl.domain_randomization.types import TerrainMapRandomizationConfig


def color_meaning_fn(
    rgba: jax.Array,
    rgba_table: jax.Array,
    friction_table: jax.Array,
    stiffness_table: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    idx = jnp.argmin(jnp.sum(jnp.abs(rgba_table[..., :-1] - rgba), axis=-1))
    return friction_table[idx], stiffness_table[idx]


def collect_tile_ids(env_model: EnvModel, tile_geom_prefix: str = "tile_") -> jax.Array:
    """Collects the ids of geoms that belong to tile bodies.

    Args:
        env_model: Compiled environment model containing the tiled terrain.
        tile_geom_prefix: Prefix that identifies bodies representing tiles.

    Returns:
        An array with the ids of all geoms that belong to bodies with the provided prefix.
    """

    geom_ids: list[int] = []
    for geom_id in range(env_model.ngeom):
        name = mujoco.mj_id2name(env_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if name is None or not name.startswith(tile_geom_prefix):
            continue
        # first_geom = env_model.body_geomadr[body_id]
        geom_ids.append(geom_id)

    if not geom_ids:
        raise ValueError(
            "Unable to locate any tile geoms. Ensure that the terrain builder "
            f"creates bodies with the prefix '{tile_geom_prefix}'."
        )

    return jnp.array(geom_ids)


@dataclass
class ColorMapRandomizationConfig(TerrainMapRandomizationConfig):
    """Randomizes a terrain consisting of multiple surface tiles, giving tiles different
    physical properties, based on their color. The assignment of colors to properties is random
    across environments."""

    num_colors: int = 2
    tile_body_prefix: str = "tile_"
    friction_min = 0.1
    friction_max = 1.6
    stiffness_min = 0.002
    stiffness_max = 0.1

    def domain_randomize(
        self,
        pipeline_model: PipelineModel,
        env_model: EnvModel,
        key: PRNGKey,
        num_worlds: int,
    ) -> tuple[PipelineModel, PipelineModel, tuple[jax.Array, jax.Array, jax.Array]]:
        """Randomizes the mjx.Model. Assumes the ground is a square grid of square tiles."""

        tile_geom_ids = collect_tile_ids(env_model, tile_geom_prefix=self.tile_body_prefix)

        @jax.vmap
        def rand(rng):
            key_tiles, key_friction, key_solref = jax.random.split(rng, 3)

            color_palette = jnp.concatenate(
                [
                    jax.random.uniform(key_tiles, shape=(self.num_colors, 3)),
                    jnp.ones((self.num_colors, 1)),
                ],
                axis=1,
            )
            color_friction = jax.random.uniform(
                key_friction,
                shape=(self.num_colors,),
                minval=self.friction_min,
                maxval=self.friction_max,
            )
            color_solref = jax.random.uniform(
                key_solref,
                shape=(self.num_colors,),
                minval=self.stiffness_min,
                maxval=self.stiffness_max,
            )
            tile_colour_indices = jax.random.randint(
                key_tiles,
                shape=(tile_geom_ids.shape[0],),
                minval=0,
                maxval=self.num_colors,
            )

            chosen_colors = color_palette[tile_colour_indices]
            chosen_frictions = color_friction[tile_colour_indices]
            chosen_solrefs = color_solref[tile_colour_indices]

            geom_rgba = pipeline_model.model.geom_rgba.at[tile_geom_ids].set(chosen_colors)
            geom_friction = pipeline_model.model.geom_friction.at[tile_geom_ids, 0].set(
                chosen_frictions
            )
            geom_solref = pipeline_model.model.geom_solref.at[tile_geom_ids, 0].set(
                chosen_solrefs
            )

            return (
                geom_rgba,
                geom_friction,
                geom_solref,
                color_palette,
                color_friction,
                color_solref,
            )

        key_envs = jax.random.split(key, num_worlds)
        rgba, friction, solref, rgba_table, friction_table, solref_table = rand(key_envs)

        in_axes = jax.tree.map(lambda x: None, pipeline_model)
        in_axes = in_axes.replace(
            model=in_axes.model.tree_replace(
                {
                    "geom_matid": 0,
                    "geom_rgba": 0,
                    "geom_friction": 0,
                    "geom_solref": 0,
                }
            )
        )

        pipeline_model = pipeline_model.replace(
            model=pipeline_model.model.tree_replace(
                {
                    "geom_matid": jnp.repeat(
                        jnp.expand_dims(
                            jnp.repeat(-2, pipeline_model.model.geom_matid.shape[0], 0), 0
                        ),
                        num_worlds,
                        axis=0,
                    ),
                    "geom_rgba": rgba,
                    "geom_friction": friction,
                    "geom_solref": solref,
                }
            )
        )

        return pipeline_model, in_axes, (rgba_table, friction_table, solref_table)
