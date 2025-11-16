"""This module uses the obstacles module to tile the terrain with a square grid of tiles.
It has several functions with example tilings."""

import random
from dataclasses import dataclass

import mujoco as mj

from quadruped_mjx_rl.config_utils import Configuration, register_config_base_class
from quadruped_mjx_rl.terrain_gen.obstacles import FlatTile, TerrainTileConfig


@dataclass
class TerrainConfig(Configuration):
    """Utility dataclass for generating a square grid of square terrain tiles."""

    tiles: list[list[TerrainTileConfig]]

    @classmethod
    def config_base_class_key(cls) -> str:
        return "terrain"

    def check_tiles(self):
        square_side = self.tiles[0][0].square_side
        row_length = len(self.tiles[0])
        for row in self.tiles:
            if len(row) != row_length:
                return False
            for tile in row:
                if tile.square_side != square_side:
                    return False
        return True

    def make_arena(self, empty_arena_spec: mj.MjSpec):
        spec = empty_arena_spec

        # Add lights
        for x in [-1, 1]:
            for y in [-1, 1]:
                spec.worldbody.add_light(pos=[x, y, 40], dir=[-x, -y, -15])

        # Sanity check
        if not self.check_tiles():
            raise ValueError("Tiles do not form a rectangular grid!")

        square_side = self.tiles[0][0].square_side
        for i in range(len(self.tiles)):
            for j in range(len(self.tiles[0])):
                self.tiles[i][j].create_tile(
                    spec=spec,
                    grid_loc=[j * 2 * square_side, i * 2 * square_side],
                    name=f"tile_{i}_{j}",
                )

        return spec

    def get_tile_center_qpos(self, row, col):
        square_side = self.tiles[0][0].square_side
        x = col * 2 * square_side
        y = row * 2 * square_side
        z_offset = self.tiles[row][col].floor_thickness
        return x, y, z_offset


register_config_base_class(TerrainConfig)


def get_simple_tiled_terrain(
    n_rows: int = 4,
    n_columns: int = 4,
    square_size: float = 1.0,
):
    """This terrain is just a square grid of featureless flat tiles."""
    return TerrainConfig(
        tiles=[
            [FlatTile(square_side=square_size) for _ in range(n_columns)] for _ in range(n_rows)
        ]
    )


def get_randomized_terrain(
    n_rows: int = 4,
    n_columns: int = 4,
    obstacles_weights: list[tuple[TerrainTileConfig, int]] | None = None,
):
    """This terrain is a square grid of randomly chosen terrain tile types according to the
    probabilities specified in the weights."""
    if obstacles_weights is None:
        obstacles_weights = [(FlatTile, 1)]
    obstacles, weights = zip(*obstacles_weights)
    return TerrainConfig(
        tiles=[random.choices(obstacles, weights=weights, k=n_columns) for _ in range(n_rows)]
    )
