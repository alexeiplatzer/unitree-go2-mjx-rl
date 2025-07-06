"""Utility functions to help visualize mujoco environments."""
from etils.epath import PathLike, Path
import mediapy as media
import mujoco


def render_scene(
    scene_path: PathLike,
    initial_keyframe: str,
    camera: str | None = None,
    save_path: PathLike | None = None,
):
    """Renders the initial scene from a given xml file"""
    xml_path = Path(scene_path).as_posix()
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    renderer = mujoco.Renderer(mj_model)
    init_q = mj_model.keyframe(initial_keyframe).qpos
    mj_data = mujoco.MjData(mj_model)
    mj_data.qpos = init_q
    mujoco.mj_forward(mj_model, mj_data)
    renderer.update_scene(mj_data, camera=camera)
    image = renderer.render()
    if save_path is not None:
        media.write_image(save_path, image)
    else:
        media.show_image(image)
