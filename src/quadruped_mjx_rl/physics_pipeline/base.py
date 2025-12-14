"""A hand-controlled copy of the most important brax functionality"""

import copy
import functools
from collections.abc import Sequence

import jax
from flax.struct import dataclass as flax_dataclass, field as flax_field
from jax import numpy as jnp
from jax import vmap
from jax.tree_util import tree_map
from mujoco import MjModel, MjSpec
from mujoco import mjx

from quadruped_mjx_rl import math

# f: free, 1: 1-dof, 2: 2-dof, 3: 3-dof
Q_WIDTHS = {"f": 7, "1": 1, "2": 2, "3": 3}
QD_WIDTHS = {"f": 6, "1": 1, "2": 2, "3": 3}

# Abstract over mujoco types
EnvModel = MjModel
EnvSpec = MjSpec
MjxModel = mjx.Model


@flax_dataclass
class Base:
    """
    Base functionality extending all brax types.
    These methods allow for brax types to be operated like arrays/matrices.
    """

    def __add__(self, o):
        return tree_map(lambda x, y: x + y, self, o)

    def __sub__(self, o):
        return tree_map(lambda x, y: x - y, self, o)

    def __mul__(self, o):
        return tree_map(lambda x: x * o, self)

    def __neg__(self):
        return tree_map(lambda x: -x, self)

    def __truediv__(self, o):
        return tree_map(lambda x: x / o, self)

    def reshape(self, shape: Sequence[int]):
        return tree_map(lambda x: x.reshape(shape), self)

    def select(self, o, cond: jax.Array):
        return tree_map(lambda x, y: (x.T * cond + y.T * (1 - cond)).T, self, o)

    def slice(self, beg: int, end: int):
        return tree_map(lambda x: x[beg:end], self)

    def take(self, i, axis=0):
        return tree_map(lambda x: jnp.take(x, i, axis=axis, mode="wrap"), self)

    def concatenate(self, *others, axis: int = 0):
        return tree_map(lambda *x: jnp.concatenate(x, axis=axis), self, *others)

    def index_set(self, idx: jax.Array | Sequence[jax.Array], o):
        return tree_map(lambda x, y: x.at[idx].set(y), self, o)

    def index_sum(self, idx: jax.Array | Sequence[jax.Array], o):
        return tree_map(lambda x, y: x.at[idx].add(y), self, o)

    def vmap(self, in_axes=0, out_axes=0):
        """Returns an object that vmaps each follow-on instance method call."""

        outer_self = self

        class VmapField:
            """Returns instance method calls as vmapped."""

            def __init__(self, in_axes, out_axes):
                self.in_axes = [in_axes]
                self.out_axes = [out_axes]

            def vmap(self, in_axes=0, out_axes=0):
                self.in_axes.append(in_axes)
                self.out_axes.append(out_axes)
                return self

            def __getattr__(self, attr):
                fun = getattr(outer_self.__class__, attr)
                # load the stack from the bottom up
                vmap_order = reversed(list(zip(self.in_axes, self.out_axes)))
                for in_axes, out_axes in vmap_order:
                    fun = vmap(fun, in_axes=in_axes, out_axes=out_axes)
                fun = functools.partial(fun, outer_self)
                return fun

        return VmapField(in_axes, out_axes)

    def tree_replace(self, params: dict[str, jax.typing.ArrayLike | None]) -> "Base":
        """Creates a new object with all parameters set.

        Args:
          params: a dictionary of key value pairs to replace

        Returns:
          data clas with new values

        Example:
          If a system has 3 links, the following code replaces the mass
          of each link in the System:
          >>> sys = sys.tree_replace({'link.inertia.mass', jnp.array([1.0, 1.2, 1.3])})
        """
        new = self
        for k, v in params.items():
            new = _tree_replace(new, k.split("."), v)
        return new

    @property
    def T(self):
        return tree_map(lambda x: x.T, self)


def _tree_replace(
    base: Base,
    attr: Sequence[str],
    val: jax.typing.ArrayLike | None,
) -> Base:
    """Sets attributes in a flax_struct.dataclass with values."""
    if not attr:
        return base

    # special case for List attribute
    if len(attr) > 1 and isinstance(getattr(base, attr[0]), list):
        lst = copy.deepcopy(getattr(base, attr[0]))

        for i, g in enumerate(lst):
            if not hasattr(g, attr[1]):
                continue
            v = val if not hasattr(val, "__iter__") else val[i]
            lst[i] = _tree_replace(g, attr[1:], v)

        return base.replace(**{attr[0]: lst})

    if len(attr) == 1:
        return base.replace(**{attr[0]: val})

    return base.replace(**{attr[0]: _tree_replace(getattr(base, attr[0]), attr[1:], val)})


@flax_dataclass
class Transform(Base):
    """Transforms the position and rotation of a coordinate frame.

    Attributes:
    pos: (3,) position transform of the coordinate frame
    rot: (4,) quaternion rotation the coordinate frame
    """

    pos: jax.Array
    rot: jax.Array

    def do(self, o):
        """Apply the transform."""
        return _transform_do(o, self)

    def inv_do(self, o):
        """Apply the inverse of the transform."""
        return _transform_inv_do(o, self)

    def to_local(self, t: "Transform") -> "Transform":
        """Move transform into basis of t."""
        pos = math.rotate(self.pos - t.pos, math.quat_inv(t.rot))
        rot = math.quat_mul(math.quat_inv(t.rot), self.rot)
        return Transform(pos=pos, rot=rot)

    @classmethod
    def create(cls, pos: jax.Array | None = None, rot: jax.Array | None = None) -> "Transform":
        """Creates a transform with either pos, rot, or both."""
        if pos is None and rot is None:
            raise ValueError("must specify either pos or rot")
        elif pos is None and rot is not None:
            pos = jnp.zeros(rot.shape[:-1] + (3,))
        elif rot is None and pos is not None:
            rot = jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), pos.shape[:-1] + (1,))
        return Transform(pos=pos, rot=rot)

    @classmethod
    def zero(cls, shape=()) -> "Transform":
        """Returns a zero transform with a batch shape."""
        pos = jnp.zeros(shape + (3,))
        rot = jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), shape + (1,))
        return Transform(pos, rot)


@flax_dataclass
class Motion(Base):
    """Spatial motion vector describing linear and angular velocity.

    More on spatial vectors: http://royfeatherstone.org/spatial/v2/index.html

    Attributes:
        ang: (3,) angular velocity about a normal
        vel: (3,) linear velocity in the direction of the normal
    """

    ang: jax.Array
    vel: jax.Array

    # def cross(self, other):
    #     return _motion_cross(other, self)

    def dot(self, m: "Motion") -> jax.Array:
        return jnp.dot(self.vel, m.vel) + jnp.dot(self.ang, m.ang)

    def matrix(self) -> jax.Array:
        return jnp.concatenate([self.ang, self.vel], axis=-1)

    @classmethod
    def create(cls, ang: jax.Array | None = None, vel: jax.Array | None = None) -> "Motion":
        if ang is None and vel is None:
            raise ValueError("must specify either ang or vel")
        ang = jnp.zeros_like(vel) if ang is None else ang
        vel = jnp.zeros_like(ang) if vel is None else vel

        return Motion(ang=ang, vel=vel)

    @classmethod
    def zero(cls, shape=()) -> "Motion":
        ang = jnp.zeros(shape + (3,))
        vel = jnp.zeros(shape + (3,))
        return Motion(ang, vel)


# Abstract over MJX types
@flax_dataclass
class PipelineState:
    q: jax.Array
    qd: jax.Array
    x: Transform
    xd: Motion
    data: mjx.Data


@flax_dataclass
class PipelineModel:
    model: MjxModel


def make_pipeline_model(env_model: EnvModel) -> PipelineModel:
    return PipelineModel(model=mjx.put_model(env_model))


# below are some operation dispatch derivations


@functools.singledispatch
def _transform_do(other, self: Transform):
    del other, self
    return NotImplemented


@functools.singledispatch
def _transform_inv_do(other, self: Transform):
    del other, self
    return NotImplemented


@_transform_do.register(Transform)
def _(t: Transform, self: Transform) -> Transform:
    pos = self.pos + math.rotate(t.pos, self.rot)
    rot = math.quat_mul(self.rot, t.rot)
    return Transform(pos, rot)


@_transform_do.register(Motion)
def _(m: Motion, self: Transform) -> Motion:
    rot_t = math.quat_inv(self.rot)
    ang = math.rotate(m.ang, rot_t)
    vel = math.rotate(m.vel - jnp.cross(self.pos, m.ang), rot_t)
    return Motion(ang, vel)


@_transform_inv_do.register(Motion)
def _(m: Motion, self: Transform) -> Motion:
    rot_t = self.rot
    ang = math.rotate(m.ang, rot_t)
    vel = math.rotate(m.vel, rot_t) + jnp.cross(self.pos, ang)
    return Motion(ang, vel)
