"""Function to load MuJoCo mjcf format to Brax model."""

# Supporting
import itertools
from xml.etree import ElementTree
from etils.epath import PathLike, Path

# Math
import jax
from jax import numpy as jnp
from quadruped_mjx_rl import math
import numpy as np

# Sim
import mujoco
from mujoco import mjx
# from quadruped_mjx_rl.environments.utils import (
#     Actuator,
#     DoF,
#     Inertia,
#     Link,
#     Motion,
#     System,
#     Transform,
# )


# def _transform_do(
#     parent_pos: np.ndarray,
#     parent_quat: np.ndarray,
#     pos: np.ndarray,
#     quat: np.ndarray,
# ) -> tuple[np.ndarray, np.ndarray]:
#     pos = parent_pos + math.rotate_np(pos, parent_quat)
#     rot = math.quat_mul_np(parent_quat, quat)
#     return pos, rot
#
#
# def _offset(
#     elem: ElementTree.Element, parent_pos: np.ndarray, parent_quat: np.ndarray
# ):
#     """Offsets an element."""
#     pos = elem.attrib.get('pos', '0 0 0')
#     quat = elem.attrib.get('quat', '1 0 0 0')
#     pos = np.fromstring(pos, sep=' ')
#     quat = np.fromstring(quat, sep=' ')
#     fromto = elem.attrib.get('fromto', None)
#     if fromto:
#         # fromto attributes are not compatible with pos/quat attributes
#         from_pos = np.fromstring(' '.join(fromto.split(' ')[0:3]), sep=' ')
#         to_pos = np.fromstring(' '.join(fromto.split(' ')[3:6]), sep=' ')
#         from_pos, _ = _transform_do(parent_pos, parent_quat, from_pos, quat)
#         to_pos, _ = _transform_do(parent_pos, parent_quat, to_pos, quat)
#         fromto = ' '.join('%f' % i for i in np.concatenate([from_pos, to_pos]))
#         elem.attrib['fromto'] = fromto
#         return
#     pos, quat = _transform_do(parent_pos, parent_quat, pos, quat)
#     pos = ' '.join('%f' % i for i in pos)
#     quat = ' '.join('%f' % i for i in quat)
#     elem.attrib['pos'] = pos
#     elem.attrib['quat'] = quat
#
#
# def _fuse_bodies(elem: ElementTree.Element):
#     """Fuses together parent child bodies that have no joint."""
#
#     for child in list(elem):  # we will modify elem children, so make a copy
#         _fuse_bodies(child)
#         # this only applies to bodies with no joints
#         if child.tag != 'body':
#             continue
#         if child.find('joint') is not None or child.find('freejoint') is not None:
#             continue
#         cpos = child.attrib.get('pos', '0 0 0')
#         cpos = np.fromstring(cpos, sep=' ')
#         cquat = child.attrib.get('quat', '1 0 0 0')
#         cquat = np.fromstring(cquat, sep=' ')
#         for grandchild in child:
#             if (
#                 grandchild.tag in ('body', 'geom', 'site', 'camera')
#                 and (cpos != 0).any()
#             ):
#                 _offset(grandchild, cpos, cquat)
#             elem.append(grandchild)
#         elem.remove(child)


def _get_meshdir(elem: ElementTree.Element) -> str | None:
    """Gets the mesh directory specified by the mujoco compiler tag."""
    elems = list(elem.iter('compiler'))
    return elems[0].get('meshdir') if elems else None


def _find_assets(
    elem: ElementTree.Element,
    path: Path,
    meshdir: str | None,
) -> dict[str, bytes]:
    """Loads assets from an xml given a base path."""
    assets = {}
    path = path if path.is_dir() else path.parent
    fname = elem.attrib.get('file') or elem.attrib.get('filename')
    if fname and fname.endswith('.xml'):
        # an asset can be another xml!  if so, we must traverse it, too
        asset = (path / fname).read_text()
        asset_xml = ElementTree.fromstring(asset)
        # _fuse_bodies(asset_xml)
        asset_meshdir = _get_meshdir(asset_xml)
        assets[fname] = ElementTree.tostring(asset_xml)
        assets.update(_find_assets(asset_xml, path, asset_meshdir))
    elif fname:
        # mesh, png, etc
        path = path / meshdir if meshdir else path
        assets[fname] = (path / fname).read_bytes()

    for child in list(elem):
        assets.update(_find_assets(child, path, meshdir))

    return assets


def _get_name(mj: mujoco.MjModel, i: int) -> str:
    names = mj.names[i:].decode('utf-8')
    return names[: names.find('\x00')]


# def validate_model(mj: mujoco.MjModel) -> None:
#     """Checks if a MuJoCo model is compatible with brax physics pipelines."""
#     if mj.opt.integrator != 0:
#         raise NotImplementedError('Only euler integration is supported.')
#     if mj.opt.cone != 0:
#         raise NotImplementedError('Only pyramidal cone friction is supported.')
#     if (mj.geom_fluid != 0).any():
#         raise NotImplementedError('Ellipsoid fluid model not implemented.')
#     if mj.opt.wind.any():
#         raise NotImplementedError('option.wind is not implemented.')
#     if mj.opt.impratio != 1:
#         raise NotImplementedError('Only impratio=1 is supported.')
#
#     # actuators
#     if any(i not in [0, 1] for i in mj.actuator_biastype):
#         raise NotImplementedError('Only actuator_biastype in [0, 1] are supported.')
#     if any(i != 0 for i in mj.actuator_gaintype):
#         raise NotImplementedError('Only actuator_gaintype in [0] is supported.')
#     if not (mj.actuator_trntype == 0).all():
#         raise NotImplementedError(
#             'Only joint transmission types are supported for actuators.'
#         )
#
#     # solver parameters
#     if (mj.geom_solmix[0] != mj.geom_solmix).any():
#         raise NotImplementedError('geom_solmix parameter not supported.')
#     if (mj.geom_priority[0] != mj.geom_priority).any():
#         raise NotImplementedError('geom_priority parameter not supported.')
#
#     # check joints
#     q_width = {0: 7, 1: 4, 2: 1, 3: 1}
#     non_free = np.concatenate([[j != 0] * q_width[j] for j in mj.jnt_type])
#     if mj.qpos0[non_free].any():
#         raise NotImplementedError(
#             'The `ref` attribute on joint types is not supported.'
#         )
#
#     for _, group in itertools.groupby(
#         zip(mj.jnt_bodyid, mj.jnt_pos), key=lambda x: x[0]
#     ):
#         position = np.array([p for _, p in group])
#         if not (position == position[0]).all():
#             raise RuntimeError('invalid joint stack: only one joint position allowed')
#
#     # check dofs
#     jnt_range = mj.jnt_range.copy()
#     jnt_range[~(mj.jnt_limited == 1), :] = np.array([-np.inf, np.inf])
#     for typ, limit, stiffness in zip(mj.jnt_type, jnt_range, mj.jnt_stiffness):
#         if typ == 0:
#             if stiffness > 0:
#                 raise RuntimeError('brax does not support stiffness for free joints')
#         elif typ == 1:
#             if np.any(~np.isinf(limit)):
#                 raise RuntimeError('brax does not support joint ranges for ball joints')
#         elif typ in (2, 3):
#             continue
#         else:
#             raise RuntimeError(f'invalid joint type: {typ}')
#
#     for _, group in itertools.groupby(
#         zip(mj.jnt_bodyid, mj.jnt_type), key=lambda x: x[0]
#     ):
#         typs = [t for _, t in group]
#         if len(typs) == 1 and typs[0] == 0:
#             continue  # free
#         elif 0 in typs:
#             raise RuntimeError('invalid joint stack: cannot stack free joints')
#         elif 1 in typs:
#             raise NotImplementedError('ball joints not supported')
#
#     # check collision geometries
#     for i, typ in enumerate(mj.geom_type):
#         mask = mj.geom_contype[i] | mj.geom_conaffinity[i] << 32
#         if typ == 5:  # Cylinder
#             _, halflength = mj.geom_size[i, 0:2]
#             if halflength > 0.001 and mask > 0:
#                 raise NotImplementedError(
#                     'Cylinders of half-length>0.001 are not supported for collision.'
#                 )


# def load_model(mj: mujoco.MjModel) -> System:
#     """Creates a brax system from a MuJoCo model."""
#
#     # create links
#     joint_positions = [np.array([0.0, 0.0, 0.0])]
#     for _, group in itertools.groupby(
#         zip(mj.jnt_bodyid, mj.jnt_pos), key=lambda x: x[0]
#     ):
#         position = np.array([p for _, p in group])
#         joint_positions.append(position[0])
#     joint_position = np.array(joint_positions)
#     identity = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (mj.nbody, 1))
#     link = Link(
#         transform=Transform(pos=mj.body_pos, rot=mj.body_quat),
#         inertia=Inertia(
#             transform=Transform(pos=mj.body_ipos, rot=mj.body_iquat),
#             i=np.array([np.diag(i) for i in mj.body_inertia]),
#             mass=mj.body_mass,
#         ),
#         invweight=mj.body_invweight0[:, 0],
#         joint=Transform(pos=joint_position, rot=identity),
#     )
#     # skip link 0 which is the world body in mujoco
#     # copy to avoid writing to mj model
#     link = jax.tree.map(lambda x: x[1:].copy(), link)
#
#     # create dofs
#     mj.jnt_range[~(mj.jnt_limited == 1), :] = np.array([-np.inf, np.inf])
#     motions, limits, stiffnesses = [], [], []
#     for typ, axis, limit, stiffness in zip(
#         mj.jnt_type, mj.jnt_axis, mj.jnt_range, mj.jnt_stiffness
#     ):
#         if typ == 0:
#             motion = Motion(ang=np.eye(6, 3, -3), vel=np.eye(6, 3))
#             limit = np.array([-np.inf] * 6), np.array([np.inf] * 6)
#             stiffness = np.zeros(6)
#         elif typ == 1:
#             motion = Motion(ang=np.eye(3), vel=np.zeros((3, 3)))
#             limit = np.array([-np.inf] * 3), np.array([np.inf] * 3)
#             stiffness = np.zeros(3)
#         elif typ == 2:
#             motion = Motion(ang=np.zeros((1, 3)), vel=axis.reshape((1, 3)))
#             limit = limit[0:1], limit[1:2]
#             stiffness = np.array([stiffness])
#         elif typ == 3:
#             motion = Motion(ang=axis.reshape((1, 3)), vel=np.zeros((1, 3)))
#             limit = limit[0:1], limit[1:2]
#             stiffness = np.array([stiffness])
#         else:
#             # invalid joint type
#             continue
#         motions.append(motion)
#         limits.append(limit)
#         stiffnesses.append(stiffness)
#     motion = jax.tree.map(lambda *x: np.concatenate(x), *motions)
#
#     limit = None
#     if np.any(mj.jnt_limited):
#         limit = jax.tree.map(lambda *x: np.concatenate(x), *limits)
#     stiffness = np.concatenate(stiffnesses)
#     solver_params_jnt = np.concatenate((mj.jnt_solref, mj.jnt_solimp), axis=1)
#     solver_params_dof = solver_params_jnt[mj.dof_jntid]
#
#     dof = DoF(
#         motion=motion,
#         armature=mj.dof_armature,
#         stiffness=stiffness,
#         damping=mj.dof_damping,
#         limit=limit,
#         invweight=mj.dof_invweight0,
#         solver_params=solver_params_dof,
#     )
#
#     # create actuators
#     ctrl_range = mj.actuator_ctrlrange
#     ctrl_range[~(mj.actuator_ctrllimited == 1), :] = np.array([-np.inf, np.inf])
#     force_range = mj.actuator_forcerange
#     force_range[~(mj.actuator_forcelimited == 1), :] = np.array([-np.inf, np.inf])
#     bias_q = mj.actuator_biasprm[:, 1] * (mj.actuator_biastype != 0)
#     bias_qd = mj.actuator_biasprm[:, 2] * (mj.actuator_biastype != 0)
#     # mask actuators since brax only supports joint transmission types
#     act_mask = mj.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT
#     trnid = mj.actuator_trnid[act_mask, 0].astype(np.uint32)
#     q_id = mj.jnt_qposadr[trnid]
#     qd_id = mj.jnt_dofadr[trnid]
#     act_kwargs = {
#         'gain': mj.actuator_gainprm[:, 0],
#         'gear': mj.actuator_gear[:, 0],
#         'ctrl_range': ctrl_range,
#         'force_range': force_range,
#         'bias_q': bias_q,
#         'bias_qd': bias_qd,
#     }
#     act_kwargs = jax.tree.map(lambda x: x[act_mask], act_kwargs)
#
#     actuator = Actuator(
#         q_id=q_id, qd_id=qd_id, **act_kwargs
#     )
#
#     # create non-pytree params.  these do not live on device directly, and they
#     # cannot be differentiated, but they do change the emitted control flow
#     link_names = [_get_name(mj, i) for i in mj.name_bodyadr[1:]]
#     # convert stacked joints to 1, 2, or 3
#     link_types = ''
#     for _, group in itertools.groupby(
#         zip(mj.jnt_bodyid, mj.jnt_type), key=lambda x: x[0]
#     ):
#         typs = [t for _, t in group]
#         if len(typs) == 1 and typs[0] == 0:  # free
#             typ = 'f'
#         elif 0 in typs or 1 in typs:
#             # invalid joint stack
#             continue
#         else:
#             typ = str(len(typs))
#         link_types += typ
#     link_parents = tuple(mj.body_parentid - 1)[1:]
#
#     # mujoco stores free q in world frame, so clear link transform for free links
#     if 'f' in link_types:
#         free_idx = np.array([i for i, typ in enumerate(link_types) if typ == 'f'])
#         link.transform.pos[free_idx] = np.zeros(3)
#         link.transform.rot[free_idx] = np.array([1.0, 0.0, 0.0, 0.0])
#
#     mjx_model = mjx.put_model(mj)
#
#     sys = System(
#         gravity=mj.opt.gravity,
#         viscosity=mj.opt.viscosity,
#         density=mj.opt.density,
#         link=link,
#         dof=dof,
#         actuator=actuator,
#         init_q=mj.qpos0,
#         enable_fluid=(mj.opt.viscosity > 0) | (mj.opt.density > 0),
#         link_names=link_names,
#         link_types=link_types,
#         link_parents=link_parents,
#         mj_model=mj,
#         **mjx_model.__dict__,
#     )
#
#     sys = jax.tree.map(jnp.array, sys)
#
#     return sys


# def fuse_bodies(xml: str):
#     """Fuses together parent child bodies that have no joint."""
#     xml = ElementTree.fromstring(xml)
#     _fuse_bodies(xml)
#     return ElementTree.tostring(xml, encoding='unicode')


def model_loads(xml: str, asset_path: PathLike | None = None) -> mujoco.MjModel:
    """Loads a brax system from a MuJoCo mjcf xml string."""
    elem = ElementTree.fromstring(xml)
    # _fuse_bodies(elem)
    assets = {}
    if asset_path is not None:
        meshdir = _get_meshdir(elem)
        asset_path = Path(asset_path)
        assets = _find_assets(elem, asset_path, meshdir)
    xml = ElementTree.tostring(elem, encoding='unicode')
    mj = mujoco.MjModel.from_xml_string(xml, assets=assets)
    return mj


def model_load(path: PathLike) -> mujoco.MjModel:
    """Loads an mj model from a MuJoCo mjcf file path."""
    elem = ElementTree.fromstring(Path(path).read_text())
    #_fuse_bodies(elem)
    meshdir = _get_meshdir(elem)
    assets = _find_assets(elem, Path(path), meshdir)
    xml = ElementTree.tostring(elem, encoding='unicode')
    mj = mujoco.MjModel.from_xml_string(xml, assets=assets)
    return mj


def load_to_spec(path: PathLike) -> mujoco.MjSpec:
    """Loads the mujoco spec from a given mjcf file path."""
    elem = ElementTree.fromstring(Path(path).read_text())
    meshdir = _get_meshdir(elem)
    assets = _find_assets(elem, Path(path), meshdir)
    xml = ElementTree.tostring(elem, encoding='unicode')
    spec = mujoco.MjSpec.from_string(xml, assets=assets)
    return spec


def spec_to_model(spec: mujoco.MjSpec) -> mujoco.MjModel:
    """Converts a mujoco spec to a mujoco model."""
    return spec.compile()


def make_pipeline_model(env_model: mujoco.MjModel) -> mjx.Model:
    return mjx.put_model(env_model)
