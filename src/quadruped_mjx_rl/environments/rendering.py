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


def render_model(
    env_model: mujoco.MjModel,
    initial_keyframe: str | None = None,
    camera: mujoco.MjvCamera | str | int = -1,
):
    """Renders the mujoco model starting under the position given under the given keyframe."""
    init_q = env_model.keyframe(initial_keyframe).qpos if initial_keyframe else None
    mj_data = mujoco.MjData(env_model)
    mj_data.qpos = init_q if init_q is not None else mj_data.qpos
    with mujoco.Renderer(env_model) as renderer:
        mujoco.mj_forward(env_model, mj_data)
        renderer = mujoco.Renderer(env_model)
        renderer.update_scene(mj_data, camera=camera)
        return renderer.render()


def large_overview_camera() -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat = [-2, 0, -2]
    camera.distance = 18
    camera.elevation = -30
    return camera


def save_image(image, save_path: PathLike):
    """Saves the image to the given path."""
    media.write_image(save_path, image)


def show_image(image):
    """Shows the image in a window."""
    media.show_image(image)
