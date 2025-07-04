"""Physics pipeline for fully articulated dynamics and collision."""

# Math
import jax
from jax import numpy as jnp

# Sim
from mujoco import mjx
from quadruped_mjx_rl.environments.utils import (
    Contact,
    Motion,
    System,
    Transform,
    State as BaseState,
)


class PipelineState(BaseState, mjx.Data):
    """Dynamic state that changes after every pipeline step."""


def _reformat_contact(sys: System, data: PipelineState) -> PipelineState:
    """Reformats the mjx.Contact into a brax.base.Contact."""
    if data.contact is None:
        return data

    elasticity = jnp.zeros(data.contact.pos.shape[0])
    body1 = jnp.array(sys.geom_bodyid)[data.contact.geom1] - 1
    body2 = jnp.array(sys.geom_bodyid)[data.contact.geom2] - 1
    link_idx = (body1, body2)
    data = data.replace(
        contact=Contact(
            link_idx=link_idx, elasticity=elasticity, **data.contact.__dict__
        )
    )
    return data


def init(
    sys: System,
    q: jax.Array,
    qd: jax.Array,
    act: jax.Array | None = None,
    ctrl: jax.Array | None = None,
) -> PipelineState:
    """
    Initializes physics data.

    Args:
        sys: a brax System
        q: (q_size,) joint angle vector
        qd: (qd_size,) joint velocity vector
        act: actuator activations
        ctrl: actuator controls

    Returns:
        data: initial physics data
    """

    data = mjx.make_data(sys)
    data = data.replace(qpos=q, qvel=qd)
    if act is not None:
        data = data.replace(act=act)
    if ctrl is not None:
        data = data.replace(ctrl=ctrl)

    data = mjx.forward(sys, data)

    q, qd = data.qpos, data.qvel
    x = Transform(pos=data.xpos[1:], rot=data.xquat[1:])
    cvel = Motion(vel=data.cvel[1:, 3:], ang=data.cvel[1:, :3])
    offset = data.xpos[1:, :] - data.subtree_com[sys.body_rootid[1:]]
    offset = Transform.create(pos=offset)
    xd = offset.vmap().do(cvel)

    data = _reformat_contact(sys, data)
    return PipelineState(q=q, qd=qd, x=x, xd=xd, **data.__dict__)


def step(
    sys: System, state: PipelineState, act: jax.Array
) -> PipelineState:
    """
    Performs a single physics step using position-based dynamics.
    Resolves actuator forces, joints, and forces at acceleration level, and
    resolves collisions at velocity level with baumgarte stabilization.

    Args:
        sys: a brax System
        state: physics data prior to step
        act: (act_size,) actuator input vector

    Returns:
        x: updated link transform in world frame
        xd: updated link motion in world frame
    """
    data = state.replace(ctrl=act)
    data = mjx.step(sys, data)

    q, qd = data.qpos, data.qvel
    x = Transform(pos=data.xpos[1:], rot=data.xquat[1:])
    cvel = Motion(vel=data.cvel[1:, 3:], ang=data.cvel[1:, :3])
    offset = data.xpos[1:, :] - data.subtree_com[sys.body_rootid[1:]]
    offset = Transform.create(pos=offset)
    xd = offset.vmap().do(cvel)

    if data.ncon > 0:
        data = _reformat_contact(sys, data)
    return data.replace(q=q, qd=qd, x=x, xd=xd)
