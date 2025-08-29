import random
from dataclasses import dataclass

import mujoco as mj

from quadruped_mjx_rl.config_utils import Configuration
from quadruped_mjx_rl.terrain_gen.obstacles import FlatTile, TerrainTileConfig


@dataclass
class TerrainConfig(Configuration):
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
                    name=f"obstacle_{i}_{j}",
                )

        return spec

    def get_tile_center_qpos(self, row, col):
        square_side = self.tiles[0][0].square_side
        x = col * 2 * square_side
        y = row * 2 * square_side
        z_offset = self.tiles[row][col].floor_thickness
        return x, y, z_offset


def get_randomized_terrain(
    n_rows: int = 4,
    n_columns: int = 4,
    obstacles_weights: list[tuple[TerrainTileConfig, int]] | None = None,
):
    if obstacles_weights is None:
        obstacles_weights = [(FlatTile, 1)]
    obstacles, weights = zip(*obstacles_weights)
    return TerrainConfig(
        tiles=[random.choices(obstacles, weights=weights, k=n_columns) for _ in range(n_rows)]
    )

    # for i in range(0, 8):
    #     stripe_w = 0.05 * (i + 1)
    #     harder_level = [(functools.partial(obstacles.stripes, stripe_w=stripe_w), 1)]
    #     add_tile(
    #         spec=spec,
    #         grid_loc=[i * 2 * SQUARE_LENGTH, 0],
    #         distribution=harder_level,
    #         name=f"obstacle_{i}",
    #     )
