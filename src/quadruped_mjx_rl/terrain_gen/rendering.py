"""This module helps visualize terrain generation intermediate results."""

import mediapy as media
import mujoco as mj

from quadruped_mjx_rl.terrain_gen.obstacles import TerrainTileConfig


def render_tile(tile_config: TerrainTileConfig, cam_distance=6, cam_elevation=-30):
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
    name = "base_tile"
    tile_config.create_tile(spec, name=name, grid_loc=[0, 0])

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
