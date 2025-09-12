import mujoco as mj

from quadruped_mjx_rl.environments.physics_pipeline import (
    EnvSpec
)


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
        spec.worldbody.add_geom(
            type=mj.mjtGeom.mjGEOM_CYLINDER,
            pos=loc,
            size=[sizes[idx], loc[2], 0],
            rgba=[1, 0.3, 0, 1],
            name=f"cylinder_{idx}"
        )


def add_goal_sphere(
    spec: EnvSpec,
    location: list[float] | None = None,
    size: float = 0.5,
):
    """Adds a green sphere to the environment spec."""

    if location is None:
        location = [10, 0, size]

    spec.worldbody.add_geom(
        type=mj.mjtGeom.mjGEOM_SPHERE,
        pos=location,
        size=[size, 0, 0],
        rgba=[1, 0, 0, 1],
        name="goal_sphere"
    )
