"""Function to load MuJoCo mjcf format to Brax model."""

from xml.etree import ElementTree
from pathlib import Path

import mujoco
from etils.epath import PathLike
from mujoco import mjx


def _get_meshdir(elem: ElementTree.Element) -> str | None:
    """Gets the mesh directory specified by the mujoco compiler tag."""
    elems = list(elem.iter("compiler"))
    return elems[0].get("meshdir") if elems else None


def _find_assets(
    elem: ElementTree.Element,
    path: Path,
    meshdir: str | None,
) -> dict[str, bytes]:
    """Loads assets from an xml given a base path."""
    assets = {}
    path = path if path.is_dir() else path.parent
    fname = elem.attrib.get("file") or elem.attrib.get("filename")
    if fname and fname.endswith(".xml"):
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
    names = mj.names[i:].decode("utf-8")
    return names[: names.find("\x00")]


def string_to_model(xml: str, asset_path: PathLike | None = None) -> mujoco.MjModel:
    """Loads a brax system from a MuJoCo mjcf xml string."""
    elem = ElementTree.fromstring(xml)
    assets = {}
    if asset_path is not None:
        meshdir = _get_meshdir(elem)
        asset_path = Path(asset_path)
        assets = _find_assets(elem, asset_path, meshdir)
    xml = ElementTree.tostring(elem, encoding="unicode")
    mj = mujoco.MjModel.from_xml_string(xml, assets=assets)
    return mj


def load_to_model(path: PathLike) -> mujoco.MjModel:
    """Loads an mj model from a MuJoCo mjcf file path."""
    elem = ElementTree.fromstring(Path(path).read_text())
    meshdir = _get_meshdir(elem)
    assets = _find_assets(elem, Path(path), meshdir)
    xml = ElementTree.tostring(elem, encoding="unicode")
    mj = mujoco.MjModel.from_xml_string(xml, assets=assets)
    return mj


def load_to_spec(path: PathLike) -> mujoco.MjSpec:
    """Loads the mujoco spec from a given mjcf file path."""
    elem = ElementTree.fromstring(Path(path).read_text())
    meshdir = _get_meshdir(elem)
    assets = _find_assets(elem, Path(path), meshdir)
    spec = mujoco.MjSpec.from_file(Path(path).as_posix(), assets=assets)
    return spec


def spec_to_model(spec: mujoco.MjSpec) -> mujoco.MjModel:
    """Converts a mujoco spec to a mujoco model."""
    return spec.compile()
