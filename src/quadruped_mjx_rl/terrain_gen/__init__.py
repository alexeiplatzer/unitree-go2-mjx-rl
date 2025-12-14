"""This module allows modifying the terrain of the environment, such as the floor surfaces,
lighting and obstacles. This is achieved by modifying the environment specification, which can
be then compiled into the environment model with which the simulator will work."""

from quadruped_mjx_rl.terrain_gen.configs import (
    TerrainConfig,
    FlatTerrainConfig,
    FlatTiledTerrainConfig,
    StripeTilesTerrainConfig,
    SimpleObstacleTerrainConfig,
    ColorMapTerrainConfig,
    make_terrain,
)
from quadruped_mjx_rl.terrain_gen.elements import add_cylinders, add_goal_sphere, add_lights
