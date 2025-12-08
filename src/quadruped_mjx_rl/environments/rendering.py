"""Utility functions to help visualize mujoco environments."""

import mediapy as media
import mujoco
import jax
import jax.numpy as jnp
import numpy as np
from etils.epath import Path, PathLike
import functools

from quadruped_mjx_rl.domain_randomization.types import DomainRandomizationFn
from quadruped_mjx_rl.environments.wrappers import wrap_for_training
from quadruped_mjx_rl.environments.physics_pipeline import Env


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
) -> np.ndarray:
    """Renders the mujoco model starting under the position given under the given keyframe."""
    init_q = env_model.keyframe(initial_keyframe).qpos if initial_keyframe else None
    mj_data = mujoco.MjData(env_model)
    mj_data.qpos = init_q if init_q is not None else mj_data.qpos
    with mujoco.Renderer(env_model) as renderer:
        mujoco.mj_forward(env_model, mj_data)
        renderer = mujoco.Renderer(env_model)
        renderer.update_scene(mj_data, camera=camera)
        return renderer.render()


def tile_images(img: np.ndarray, d: int) -> np.ndarray:
    assert img.shape[0] == d * d
    img = img.reshape((d, d) + img.shape[1:])
    return np.concat(np.concat(img, axis=1), axis=1)  # replace with 2 for multi-camera tensors!


def render_vision_observations(
    env: Env, seed: int, domain_rand_fn: DomainRandomizationFn, num_worlds: int
) -> list[np.ndarray]:
    rng_key = jax.random.PRNGKey(seed)
    domain_rand_key, reset_key = jax.random.split(rng_key, 2)
    wrapped_env = wrap_for_training(
        env=env,
        vision=True,
        num_vision_envs=num_worlds,
        randomization_fn=functools.partial(
            domain_rand_fn,
            env_model=env.env_model,
            rng_key=domain_rand_key,
            num_worlds=num_worlds,
        ),
    )
    jit_reset = jax.jit(wrapped_env.reset)
    jit_step = jax.jit(wrapped_env.step)

    # Execute one step
    state = jit_reset(jax.random.split(reset_key, num_worlds))
    state = jit_step(state, jnp.zeros((num_worlds, env.action_size)))

    n_images, n_rows = (
        (16, 4)
        if num_worlds >= 16
        else (9, 3) if num_worlds >= 9 else (4, 2) if num_worlds >= 4 else (1, 1)
    )
    images = []
    for key in state.obs:
        if key.startswith("pixels/"):
            view_tensor = state.obs[key]
            view_image = tile_images(view_tensor[:n_images], n_rows)
            images.append(view_image)
    return images


def large_overview_camera(lookat=None) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    if lookat is None:
        lookat = [0, 0, 0]
    camera.lookat = lookat
    camera.distance = 18
    camera.elevation = -30
    return camera


def save_image(image: np.ndarray, save_path: PathLike):
    """Saves the image to the given path."""
    media.write_image(save_path, image)


def show_image(image: np.ndarray):
    """Shows the image in a window."""
    media.show_image(image)
