
# Typing
from collections.abc import Sequence

# Supporting
from etils.epath import PathLike, Path
import mediapy as media

# Math
import numpy as np

# Sim
import mujoco
from quadruped_mjx_rl.environments.pipeline_utils import System, PipelineState


def render_array(
    sys: System,
    trajectory: list[PipelineState] | PipelineState,
    height: int = 240,
    width: int = 320,
    camera: str | None = None,
) -> Sequence[np.ndarray] | np.ndarray:
    """Returns a sequence of np.ndarray images using the MuJoCo renderer."""
    renderer = mujoco.Renderer(sys.mj_model, height=height, width=width)
    camera = camera or -1

    def get_image(state: PipelineState):
        d = mujoco.MjData(sys.mj_model)
        d.qpos, d.qvel = state.q, state.qd
        if hasattr(state, 'mocap_pos') and hasattr(state, 'mocap_quat'):
            d.mocap_pos, d.mocap_quat = state.mocap_pos, state.mocap_quat
        mujoco.mj_forward(sys.mj_model, d)
        renderer.update_scene(d, camera=camera)
        return renderer.render()

    if isinstance(trajectory, list):
        return [get_image(s) for s in trajectory]

    return get_image(trajectory)


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
