"""Physics pipeline for fully articulated dynamics and collision."""

# Supporting
import functools

# Math
import jax
from jax import numpy as jnp
from quadruped_mjx_rl import math

# Sim
from mujoco import mjx
from quadruped_mjx_rl.environments.physics_pipeline.base import Transform, Motion
# from quadruped_mjx_rl.environments.utils import (
#     # Contact,
#     # Motion,
#     # System,
#     # Transform,
#     # State as BaseState,
# )


# class PipelineState(BaseState, mjx.Data):
#     """Dynamic state that changes after every pipeline step."""


# def _reformat_contact(sys: System, data: PipelineState) -> PipelineState:
#     """Reformats the mjx.Contact into a brax.base.Contact."""
#     if data.contact is None:
#         return data
#
#     elasticity = jnp.zeros(data.contact.pos.shape[0])
#     body1 = jnp.array(sys.geom_bodyid)[data.contact.geom1] - 1
#     body2 = jnp.array(sys.geom_bodyid)[data.contact.geom2] - 1
#     link_idx = (body1, body2)
#     data = data.replace(
#         contact=Contact(
#             link_idx=link_idx, elasticity=elasticity, **data.contact.__dict__
#         )
#     )
#     return data


def pipeline_init(
    pipeline_model: mjx.Model,
    qpos: jax.Array,
    qvel: jax.Array,
    act: jax.Array | None = None,
    ctrl: jax.Array | None = None,
) -> mjx.Data:
    """
    Initializes physics data.

    Args:
        pipeline_model: an mjx Model
        qpos: (pipeline_model.nq,) joint angle vector
        qvel: (pipeline_model.nv,) joint velocity vector
        act: actuator activations
        ctrl: actuator controls

    Returns:
        pipeline_state: initial physics data
    """
    pipeline_state = mjx.make_data(pipeline_model)
    if qpos is not None:
        pipeline_state = pipeline_state.replace(qpos=qpos)
    if qvel is not None:
        pipeline_state = pipeline_state.replace(qvel=qvel)
    if act is not None:
        pipeline_state = pipeline_state.replace(act=act)
    if ctrl is not None:
        pipeline_state = pipeline_state.replace(ctrl=ctrl)
    pipeline_state = mjx.forward(pipeline_model, pipeline_state)
    return pipeline_state


def pipeline_step(
    pipeline_model: mjx.Model, pipeline_state: mjx.Data, act: jax.Array
) -> mjx.Data:
    """
    Performs a single physics step using the mjx physics pipeline.

    Args:
        pipeline_model: an mjx Model
        pipeline_state: physics data prior to the step
        act: (pipeline_model.nu,) actuator input vector

    Returns:
        pipeline_state: updated physics data
    """
    pipeline_state = pipeline_state.replace(ctrl=act)
    pipeline_state = mjx.step(pipeline_model, pipeline_state)
    return pipeline_state


def get_world_frame_coordinates(
    pipeline_model: mjx.Model, pipeline_state: mjx.Data
) -> tuple[Transform, Motion]:
    x = Transform(pos=pipeline_state.xpos[1:], rot=pipeline_state.xquat[1:])
    cvel = Motion(vel=pipeline_state.cvel[1:, 3:], ang=pipeline_state.cvel[1:, :3])
    offset = (
        pipeline_state.xpos[1:, :] - pipeline_state.subtree_com[pipeline_model.body_rootid[1:]]
    )
    offset = Transform.create(pos=offset)
    xd = offset.vmap().do(cvel)
    return x, xd

#
# def rotate_to_world_frame(vector: jax.Array, base_xquat: jax.Array) -> jax.Array:
#     return math.rotate(vector, math.quat_inv(base_xquat))
#
#
# def get_base_lin_vel(pipeline_state: mjx.Data) -> jax.Array:
#     return rotate_to_world_frame(pipeline_state.cvel[1, :3], base_xquat=pipeline_state.xquat[1])
#
#
# def get_base_ang_vel(pipeline_state: mjx.Data) -> jax.Array:
#     return rotate_to_world_frame(pipeline_state.cvel[1, 3:], base_xquat=pipeline_state.xquat[1])
