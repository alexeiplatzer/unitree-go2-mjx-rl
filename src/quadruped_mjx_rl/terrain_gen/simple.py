"""This module adds simple geometric bodies directly to the environment spec, bypassing the
tiled terrain approach."""

import mujoco as mj
import numpy as np

from quadruped_mjx_rl.environments.physics_pipeline import EnvSpec


def add_lights(
    spec: EnvSpec,
):
    # Add lights
    for x in [-1, 1]:
        for y in [-1, 1]:
            spec.worldbody.add_light(pos=[x, y, 40], dir=[-x, -y, -15])


def add_cylinders(
    spec: EnvSpec,
    locations: list[list[float]] | None = None,
    sizes: list[float] | float = 0.2,
):
    """Adds a bunch of orange cylinders to the environment spec."""

    if locations is None:
        locations = [[4.5, 0.0, 0.5], [5.5, 0.0, 0.5]]

    if isinstance(sizes, float):
        sizes = [sizes] * len(locations)

    for idx, loc in enumerate(locations):
        body = spec.worldbody.add_body(pos=loc, name=f"cylinder_{idx}")
        body.add_geom(
            type=mj.mjtGeom.mjGEOM_CYLINDER,
            size=[sizes[idx], loc[2], 0],
            rgba=[1, 0.3, 0, 1],
        )
        body.add_freejoint()
        for frame_key in spec.keys:
            frame_key.qpos = np.concatenate([frame_key.qpos, body.pos, body.quat])


def add_goal_sphere(
    spec: EnvSpec,
    location: list[float] | None = None,
    size: float = 0.5,
):
    """Adds a green sphere to the environment spec."""

    if location is None:
        location = [10, 0, size]

    body = spec.worldbody.add_body(pos=location, name="goal_sphere")
    body.add_geom(
        type=mj.mjtGeom.mjGEOM_SPHERE,
        size=[size, 0, 0],
        rgba=[1, 0, 0, 1],
    )
