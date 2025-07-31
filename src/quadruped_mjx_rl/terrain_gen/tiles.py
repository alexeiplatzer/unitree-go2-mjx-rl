
import mujoco as mj
import numpy as np
import mediapy as media
from scipy.signal import convolve2d


def render_tile(tile_func, direction=None, cam_distance=6, cam_elevation=-30):
    arena_xml = """
    <mujoco>
      <visual>
        <headlight diffuse=".5 .5 .5" specular="1 1 1"/>
        <global elevation="-10" offwidth="2048" offheight="1536"/>
        <quality shadowsize="8192"/>
      </visual>

      <asset>
        <texture type="skybox" builtin="gradient" rgb1=".5 .5 .5" rgb2="0 0 0" width="10" height="10"/>
        <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="1 1 1" rgb2="1 1 1" markrgb="0 0 0" width="300" height="300"/>
        <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.3"/>
      </asset>

      <worldbody>
        <light pos="0 0 5" diffuse="1 1 1" specular="1 1 1"/>
      </worldbody>
    </mujoco>
    """

    spec = mj.MjSpec.from_string(arena_xml)
    main = spec.default
    main.geom.type = mj.mjtGeom.mjGEOM_BOX

    name = 'base_tile'

    spec.worldbody.add_body(pos=[-3, 0, 0], name=name)
    if direction:
        tile_func(spec, direction=direction)
    else:
        tile_func(spec)

    model = spec.compile()
    data = mj.MjData(model)

    cam = mj.MjvCamera()
    mj.mjv_defaultCamera(cam)
    cam.lookat = [0, 0, 0]
    cam.distance = cam_distance
    cam.elevation = cam_elevation

    height = 300

    with mj.Renderer(model, 480, 640) as renderer:
        mj.mj_forward(model, data)
        renderer.update_scene(data, cam)
        media.show_image(renderer.render(), height=height)


def interpolant(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def perlin(shape, res, tileable=(False, False), interpolant=interpolant):
    """Generate a 2D numpy array of perlin noise.

    Args:
        shape: The shape of the generated array (tuple of two ints).
            This must be a multiple of res.
        res: The number of periods of noise to generate along each
            axis (tuple of two ints). Note shape must be a multiple of
            res.
        tileable: If the noise should be tileable along each axis
            (tuple of two bools). Defaults to (False, False).
        interpolant: The interpolation function, defaults to
            t*t*t*(t*(t*6 - 15) + 10).

    Returns:
        A numpy array of shape shape with the generated noise.

    Raises:
        ValueError: If shape is not a multiple of res.
    """
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    if tileable[0]:
        gradients[-1, :] = gradients[0, :]
    if tileable[1]:
        gradients[:, -1] = gradients[:, 0]
    gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)
    g00 = gradients[:-d[0], :-d[1]]
    g10 = gradients[d[0]:, :-d[1]]
    g01 = gradients[:-d[0], d[1]:]
    g11 = gradients[d[0]:, d[1]:]
    # Ramps
    n00 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1])) * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
    # Interpolation
    t = interpolant(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)


def edge_slope(size, border_width=5, blur_iterations=20):
    """Creates a grayscale image with a white center and fading black edges using convolution."""
    img = np.ones((size, size), dtype=np.float32)
    img[:border_width, :] = 0
    img[-border_width:, :] = 0
    img[:, :border_width] = 0
    img[:, -border_width:] = 0

    kernel = np.array(
        [[1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]]
    ) / 9.0

    for _ in range(blur_iterations):
        img = convolve2d(img, kernel, mode='same', boundary='symm')

    return img
