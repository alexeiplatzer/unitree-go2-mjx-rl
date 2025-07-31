# This should have a function similar to obstacles.py, but it should only be vertical
# stripes in the ground, slight protrusions up and down
import numpy as np

def generate_vertical_stripes(width, height, stripe_width=10, elevation_amplitude=2, elevation_steepness=1):
    """
    Generate a 2D array representing ground with vertical stripes and slight elevation changes.

    Parameters:
        width (int): Width of the ground.
        height (int): Height of the ground.
        stripe_width (int): Width of each vertical stripe.
        elevation_amplitude (float): Maximum elevation change.
        elevation_steepness (float): Controls how steep the elevation changes are.

    Returns:
        np.ndarray: 2D array of ground elevations.
    """
    ground = np.zeros((height, width))
    for x in range(width):
        stripe_idx = x // stripe_width
        elevation = elevation_amplitude * np.sin(stripe_idx * elevation_steepness)
        ground[:, x] += elevation
    return ground


def generate_striped_ground(spec=None, grid_loc=[0, 0], width=20, height=20, stripe_width=4, elevation_amplitude=0.1, elevation_steepness=1, name='striped_ground'):
    """
    Adds vertical stripes with slight elevation changes to the ground in the MuJoCo spec.

    Parameters:
        spec: MuJoCo spec object.
        grid_loc: Starting location of the ground.
        width: Number of stripes horizontally.
        height: Number of stripes vertically.
        stripe_width: Width of each vertical stripe.
        elevation_amplitude: Maximum elevation change for stripes.
        elevation_steepness: Controls steepness of elevation changes.
        name: Name of the ground body.
    """
    if spec is None:
        spec = mj.MjSpec()

    main = spec.default
    main.geom.type = mj.mjtGeom.mjGEOM_BOX

    body = spec.worldbody.add_body(pos=grid_loc + [0], name=name)
    BROWN = [0.460, 0.362, 0.216, 1.0]

    for x in range(0, width, stripe_width):
        elevation = elevation_amplitude * np.sin((x // stripe_width) * elevation_steepness)
        pos = [x + stripe_width / 2 - width / 2, 0, elevation]
        size = [stripe_width / 2, height / 2, 0.01]
    body.add_geom(pos=pos, size=size, rgba=BROWN)