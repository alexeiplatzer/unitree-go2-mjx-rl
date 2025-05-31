import jax
import jax.numpy as jnp
from mujoco import mjx
import numpy as np
from brax.base import System


# def domain_randomize(
#     sys: System, rng: jax.Array, num_worlds: int
# ) -> tuple[System, System]:
#     """Tile the necessary axes for the Madrona BatchRenderer."""
#
#     in_axes = jax.tree_util.tree_map(lambda x: None, sys)
#     in_axes = in_axes.tree_replace(
#         {
#             'geom_rgba': 0,
#             'geom_matid': 0,
#             'cam_pos': 0,
#             'cam_quat': 0,
#             'light_pos': 0,
#             'light_dir': 0,
#             'light_directional': 0,
#             'light_castshadow': 0,
#         }
#     )
#     rng = jax.random.key(0)
#
#     @jax.vmap
#     def rand(rng: jax.Array, light_position: jax.Array):
#         """Generate randomized model fields."""
#         _, key = jax.random.split(rng, 2)
#
#         #### Apearance ####
#         # Sample a random color for the box
#
#         mat_offset, num_geoms = 5, geom_rgba.shape[0]
#         key_matid, key = jax.random.split(key)
#         geom_matid = (
#             jax.random.randint(key_matid, shape=(num_geoms,), minval=0, maxval=10)
#             + mat_offset
#         )
#         geom_matid = geom_matid.at[box_geom_id].set(
#             -2
#         )  # Use the above randomized colors
#         geom_matid = geom_matid.at[floor_geom_id].set(-2)
#         geom_matid = geom_matid.at[strip_geom_id].set(-2)
#
#         #### Cameras ####
#         key_pos, key_ori, key = jax.random.split(key, 3)
#         cam_offset = jax.random.uniform(key_pos, (3,), minval=-0.05, maxval=0.05)
#         assert (
#             len(mjx_model.cam_pos) == 1
#         ), f'Expected single camera, got {len(mjx_model.cam_pos)}'
#         cam_pos = mjx_model.cam_pos.at[0].set(mjx_model.cam_pos[0] + cam_offset)
#         cam_quat = mjx_model.cam_quat.at[0].set(
#             perturb_orientation(key_ori, mjx_model.cam_quat[0], 10)
#         )
#
#         #### Lighting ####
#         nlight = mjx_model.light_pos.shape[0]
#         assert (
#             nlight == 1
#         ), f'Sim2Real was trained with a single light source, got {nlight}'
#         key_lsha, key_ldir, key = jax.random.split(key, 3)
#
#         # Direction
#         shine_at = jnp.array([0.661, -0.001, 0.179])  # Gripper starting position
#         nom_dir = (shine_at - light_position) / jnp.linalg.norm(
#             shine_at - light_position
#         )
#         light_dir = mjx_model.light_dir.at[0].set(
#             perturb_orientation(key_ldir, nom_dir, 20)
#         )
#
#         # Whether to cast shadows
#         light_castshadow = jax.random.bernoulli(
#             key_lsha, 0.75, shape=(nlight,)
#         ).astype(jnp.float32)
#
#         # No need to randomize into specular lighting
#         light_directional = jnp.ones((nlight,))
#
#         return (
#             geom_rgba,
#             geom_matid,
#             cam_pos,
#             cam_quat,
#             light_dir,
#             light_directional,
#             light_castshadow,
#         )
#
#     (
#         geom_rgba,
#         geom_matid,
#         cam_pos,
#         cam_quat,
#         light_dir,
#         light_directional,
#         light_castshadow,
#     ) = rand(jax.random.split(rng, num_worlds), light_positions)
#
#     mjx_model = mjx_model.tree_replace(
#         {
#             'geom_rgba': geom_rgba,
#             'geom_matid': geom_matid,
#             'cam_pos': cam_pos,
#             'cam_quat': cam_quat,
#             'light_pos': light_positions,
#             'light_dir': light_dir,
#             'light_directional': light_directional,
#             'light_castshadow': light_castshadow,
#         }
#     )
#
#     return mjx_model, in_axes
