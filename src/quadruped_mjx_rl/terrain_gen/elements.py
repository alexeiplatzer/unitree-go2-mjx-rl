"""This module adds simple geometric bodies directly to the environment spec, bypassing the
tiled terrain approach."""

import mujoco as mj
import numpy as np

from quadruped_mjx_rl.environments.physics_pipeline import EnvSpec
from quadruped_mjx_rl.robots import RobotConfig


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


def add_world_camera(
    spec: EnvSpec,
    name: str,
    location: list[float] | None = None,
    xyaxes: list[float] | None = None,
):
    """Adds a fixed camera attached to the world body to the environment spec."""
    spec.worldbody.add_camera(
        name=name,
        mode=mj.mjtCamLight.mjCAMLIGHT_FIXED,
        pos=location or [0, 0, 0],
        xyaxes=xyaxes or [0, 0, 1, 0, -1, 0],
    )


def add_robot_camera(
    spec: EnvSpec,
    name: str,
    robot_config: RobotConfig,
    mode: str = "fixed",  # or "track"
    location: list[float] | None = None,
    xyaxes: list[float] | None = None,
    orthographic: bool = False,
    fovy: float = 45.0,
):
    """Adds a camera to the robot's torso/main body."""
    modes = {
        "fixed": mj.mjtCamLight.mjCAMLIGHT_FIXED,
        "track": mj.mjtCamLight.mjCAMLIGHT_TRACK,
    }
    spec.body(robot_config.main_body_name).add_camera(
        name=name,
        mode=modes[mode],
        pos=location or [0, 0, 0],
        xyaxes=xyaxes or [0, 0, 1, 0, -1, 0],
        orthographic=orthographic,
        fovy=fovy,
    )
