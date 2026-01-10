"""This module uses the obstacles module to tile the terrain with a square grid of tiles.
It has several functions with example tilings."""

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pathlib import Path

import mujoco as mj

from quadruped_mjx_rl.domain_randomization import (
    DomainRandomizationConfig,
    TerrainMapRandomizationConfig,
    PhysicsDomainRandomizationConfig,
    ObstaclePositionRandomizationConfig,
    ColorMapRandomizationConfig,
)
from quadruped_mjx_rl.config_utils import Configuration, register_config_base_class
from quadruped_mjx_rl.terrain_gen.obstacles import FlatTile, TerrainTileConfig, StripesTile
from quadruped_mjx_rl.terrain_gen.elements import (
    add_camera_to_body,
    add_cylinders,
    add_goal_sphere,
    add_lights,
    add_robot_camera,
    add_world_camera,
    CameraConfig,
    predefined_camera_configs,
)
from quadruped_mjx_rl.robots import RobotConfig
from quadruped_mjx_rl.physics_pipeline import load_to_spec, spec_to_model, EnvModel


@dataclass
class TerrainConfig(Configuration, ABC):
    """Configuration class describing the terrain of the environment."""

    base_scene_file: str = "scene_mjx_vision.xml"
    randomization_config: DomainRandomizationConfig | TerrainMapRandomizationConfig | None = (
        None
    )
    egocentric_camera: CameraConfig = field(
        default_factory=lambda: predefined_camera_configs["frontal_ego"]
    )
    add_goal: bool = False
    goal_location: list[float] = field(default_factory=lambda: [20, 0, 2])
    goal_size: float = 2

    @property
    def visualization_cameras(self) -> dict[str, CameraConfig]:
        visualization_cameras = ["high_above", "large_overview", "follow_behind"]
        return {name: predefined_camera_configs[name] for name in visualization_cameras}

    @classmethod
    def config_base_class_key(cls) -> str:
        return "terrain"

    @classmethod
    @abstractmethod
    def config_class_key(cls) -> str:
        pass

    @classmethod
    def _get_config_class_dict(cls) -> dict[str, type["Configuration"]]:
        return _terrain_config_classes

    @abstractmethod
    def create_in_spec(self, spec: mj.MjSpec, robot_config: RobotConfig) -> None:
        """Creates the terrain in the MuJoCo spec."""
        add_robot_camera(spec, robot_config=robot_config, camera_config=self.egocentric_camera)
        for camera_config in self.visualization_cameras.values():
            add_world_camera(spec, camera_config=camera_config)
        if self.add_goal:
            add_goal_sphere(spec, location=self.goal_location, size=self.goal_size)


register_config_base_class(TerrainConfig)

_terrain_config_classes = {}

register_terrain_config_class = TerrainConfig.make_register_config_class()


def make_terrain(
    resources_directory: Path, terrain_config: TerrainConfig, robot_config: RobotConfig
) -> EnvModel:
    """Creates a model with the given terrain configuration."""
    spec = load_to_spec(
        resources_directory / robot_config.robot_name / terrain_config.base_scene_file
    )
    terrain_config.create_in_spec(spec, robot_config)
    return spec_to_model(spec)


@dataclass
class FlatTerrainConfig(TerrainConfig):
    """A simple flat terrain with no modifications, and a simple domain randomization function,
    that randomizes global physical parameters."""

    randomization_config: PhysicsDomainRandomizationConfig = field(
        default_factory=PhysicsDomainRandomizationConfig
    )

    @classmethod
    def config_class_key(cls) -> str:
        return "Flat"

    def create_in_spec(self, spec: mj.MjSpec, robot_config: RobotConfig) -> None:
        super().create_in_spec(spec, robot_config)


def check_tiles(tiles: list[list[TerrainTileConfig]]) -> bool:
    """Checks whether the tiles form a rectangular grid."""
    square_side = tiles[0][0].square_side
    row_length = len(tiles[0])
    for row in tiles:
        if len(row) != row_length:
            return False
        for tile in row:
            if tile.square_side != square_side:
                return False
    return True


def make_arena(
    empty_arena_spec: mj.MjSpec, tiles: list[list[TerrainTileConfig]], column_offset: int = 0
) -> None:
    """Adds some lights and the listed tiles in a grid pattern to the provided spec"""
    spec = empty_arena_spec

    # TODO: maybe make lights addition a configurable option
    # # Add lights
    # for x in [-1, 1]:
    #     for y in [-1, 1]:
    #         spec.worldbody.add_light(pos=[x, y, 40], dir=[-x, -y, -15])

    # Sanity check
    if not check_tiles(tiles):
        raise ValueError("Tiles do not form a rectangular grid!")

    square_side = tiles[0][0].square_side
    row_offset = len(tiles) // 2
    for i in range(len(tiles)):
        for j in range(len(tiles[0])):
            tiles[i][j].create_tile(
                spec=spec,
                grid_loc=[
                    (j - column_offset) * 2 * square_side,
                    (i - row_offset) * 2 * square_side,
                ],
                name=f"tile_{i}_{j}",
            )

    return spec


def get_tile_center_qpos(
    tiles: list[list[TerrainTileConfig]], row: int, col: int
) -> tuple[float, float, float]:
    """Returns the x-y coordinates and the z offset of the center of the tile at row, col."""
    square_side = tiles[0][0].square_side
    x = col * 2 * square_side
    y = row * 2 * square_side
    z_offset = tiles[row][col].floor_thickness
    return x, y, z_offset


@dataclass
class FlatTiledTerrainConfig(TerrainConfig):
    """A terrain with a rectangular grid of featureless monochromatic flat tiles.
    Useful for domain-randomizing their properties"""

    base_scene_file: str = "scene_mjx_vision.xml"
    n_rows: int = 20
    n_columns: int = 20
    square_size: float = 1.0
    floor_thickness: float = 0.05
    column_offset: int = 0

    @classmethod
    def config_class_key(cls) -> str:
        return "FlatTiled"

    def create_in_spec(self, spec: mj.MjSpec, robot_config: RobotConfig) -> None:
        super().create_in_spec(spec, robot_config)
        tiles = [
            [
                FlatTile(square_side=self.square_size, floor_thickness=self.floor_thickness)
                for _ in range(self.n_columns)
            ]
            for _ in range(self.n_rows)
        ]
        make_arena(spec, tiles, self.column_offset)


@dataclass
class ColorMapTerrainConfig(FlatTiledTerrainConfig):
    randomization_config: ColorMapRandomizationConfig = field(
        default_factory=ColorMapRandomizationConfig
    )
    n_rows: int = 5
    n_columns: int = 20
    terrain_map_camera: CameraConfig = field(
        default_factory=lambda: predefined_camera_configs["terrain_map"]
    )

    @classmethod
    def config_class_key(cls) -> str:
        return "ColorMap"

    def create_in_spec(self, spec: mj.MjSpec, robot_config: RobotConfig) -> None:
        add_robot_camera(
            spec=spec, robot_config=robot_config, camera_config=self.terrain_map_camera
        )
        super().create_in_spec(spec, robot_config)


@dataclass
class StripeTilesTerrainConfig(TerrainConfig):
    """A terrain with a line of tiles, starting out flat and then with stripes of
    elevated and depressed ground level."""

    base_scene_file: str = "scene_mjx_empty_arena.xml"
    n_flat_tiles: int = 4
    n_stripe_tiles: int = 4
    square_size: float = 1.0

    @classmethod
    def config_class_key(cls) -> str:
        return "StripeTiles"

    def create_in_spec(self, spec: mj.MjSpec, robot_config: RobotConfig) -> None:
        super().create_in_spec(spec, robot_config)
        flat_tiles = [FlatTile(square_side=self.square_size) for _ in range(self.n_flat_tiles)]
        stripe_tiles = [
            StripesTile(square_side=self.square_size) for _ in range(self.n_stripe_tiles)
        ]
        tiles = [flat_tiles + stripe_tiles]
        make_arena(spec, tiles)


@dataclass
class SimpleObstacleTerrainConfig(TerrainConfig):
    """A terrain with two cylinder bodies obstacles and a goal sphere."""

    # TODO: make the cylinder addition dynamical
    base_scene_file: str = "scene_mjx_cylinders.xml"
    randomization_config: PhysicsDomainRandomizationConfig = field(
        default_factory=PhysicsDomainRandomizationConfig
    )

    @classmethod
    def config_class_key(cls) -> str:
        return "SimpleObstacle"

    def create_in_spec(self, spec: mj.MjSpec, robot_config: RobotConfig) -> None:
        super().create_in_spec(spec, robot_config)


register_terrain_config_class(FlatTerrainConfig)
register_terrain_config_class(ColorMapTerrainConfig)
register_terrain_config_class(StripeTilesTerrainConfig)
register_terrain_config_class(FlatTiledTerrainConfig)
register_terrain_config_class(SimpleObstacleTerrainConfig)
