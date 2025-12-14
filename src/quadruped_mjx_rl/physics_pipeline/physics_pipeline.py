"""Physics pipeline for fully articulated dynamics and collision."""

import jax
from mujoco import mjx

from quadruped_mjx_rl.physics_pipeline.base import (
    Motion,
    Transform,
    PipelineState,
    PipelineModel,
)


def pipeline_init(
    pipeline_model: PipelineModel,
    qpos: jax.Array,
    qvel: jax.Array,
    act: jax.Array | None = None,
    ctrl: jax.Array | None = None,
) -> PipelineState:
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
    data = mjx.make_data(pipeline_model.model)
    if qpos is not None:
        data = data.replace(qpos=qpos)
    if qvel is not None:
        data = data.replace(qvel=qvel)
    if act is not None:
        data = data.replace(act=act)
    if ctrl is not None:
        data = data.replace(ctrl=ctrl)
    data = mjx.forward(pipeline_model.model, data)
    q = data.qpos
    qd = data.qvel
    x = Transform(pos=data.xpos[1:], rot=data.xquat[1:])
    cvel = Motion(vel=data.cvel[1:, 3:], ang=data.cvel[1:, :3])
    offset = data.xpos[1:, :] - data.subtree_com[pipeline_model.model.body_rootid[1:]]
    offset = Transform.create(pos=offset)
    xd = offset.vmap().do(cvel)
    return PipelineState(q=q, qd=qd, x=x, xd=xd, data=data)


def pipeline_step(
    pipeline_model: PipelineModel, pipeline_state: PipelineState, act: jax.Array
) -> PipelineState:
    """
    Performs a single physics step using the mjx physics pipeline.

    Args:
        pipeline_model: an mjx Model
        pipeline_state: physics data prior to the step
        act: (pipeline_model.nu,) actuator input vector

    Returns:
        pipeline_state: updated physics data
    """
    data = pipeline_state.data.replace(ctrl=act)
    data = mjx.step(pipeline_model.model, data)
    q, qd = data.qpos, data.qvel
    x = Transform(pos=data.xpos[1:], rot=data.xquat[1:])
    cvel = Motion(vel=data.cvel[1:, 3:], ang=data.cvel[1:, :3])
    offset = data.xpos[1:, :] - data.subtree_com[pipeline_model.model.body_rootid[1:]]
    offset = Transform.create(pos=offset)
    xd = offset.vmap().do(cvel)

    return pipeline_state.replace(q=q, qd=qd, x=x, xd=xd, data=data)
