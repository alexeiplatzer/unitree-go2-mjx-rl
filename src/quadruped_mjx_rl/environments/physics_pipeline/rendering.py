# Typing
from collections.abc import Sequence

# Math
import numpy as np

# Sim
import mujoco
from mujoco import mjx


def render_array(
    env_model: mujoco.MjModel,
    trajectory: list[mjx.Data] | mjx.Data,
    height: int = 240,
    width: int = 320,
    camera: str | None = None,
) -> Sequence[np.ndarray] | np.ndarray:
    """Returns a sequence of np.ndarray images using the MuJoCo renderer."""
    renderer = mujoco.Renderer(env_model, height=height, width=width)
    camera = camera or -1

    def get_image(state: mjx.Data) -> np.ndarray:
        d = mujoco.MjData(env_model)
        d.qpos, d.qvel = state.qpos, state.qvel
        if hasattr(state, "mocap_pos") and hasattr(state, "mocap_quat"):
            d.mocap_pos, d.mocap_quat = state.mocap_pos, state.mocap_quat
        mujoco.mj_forward(env_model, d)
        renderer.update_scene(d, camera=camera)
        return renderer.render()

    if isinstance(trajectory, list):
        return [get_image(s) for s in trajectory]

    return get_image(trajectory)
