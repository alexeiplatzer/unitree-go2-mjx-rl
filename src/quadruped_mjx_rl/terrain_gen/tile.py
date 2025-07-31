
import random
import mujoco as mj

from quadruped_mjx_rl.terrain_gen import obstacles


def add_tile(
    spec: mj.MjSpec | None = None,
    grid_loc: list[int] | None = None,
    distribution: list[tuple[obstacles.ObstacleMaker, int]] | None = None
):
    if spec is None:
        spec = mj.MjSpec()

    if grid_loc is None:
        grid_loc = [0, 0]

    if distribution is None:
        distribution = [(obstacles.stripes, 1)]

    obstacle_makers, weights = zip(*distribution)

    chosen_obstacle_maker = random.choices(obstacle_makers, weights=weights, k=1)[0]
    chosen_obstacle_maker(spec, grid_loc)
    return spec

    # tile_type = random.randint(0, 9)
    #
    # if tile_type == 0:
    #     debris_with_simple_geoms(spec, grid_loc, name=f"plane_{grid_loc[0]}_{grid_loc[1]}")
    # elif tile_type == 1:
    #     stairs(spec, grid_loc, name=f"stairs_up_{grid_loc[0]}_{grid_loc[1]}", direction=1)
    # elif tile_type == 2:
    #     stairs(spec, grid_loc, name=f"stairs_down_{grid_loc[0]}_{grid_loc[1]}", direction=-1)
    # elif tile_type == 3:
    #     debris(spec, grid_loc, name=f"debris_{grid_loc[0]}_{grid_loc[1]}")
    # elif tile_type == 4:
    #     box_extrusions(spec, grid_loc, name=f"box_extrusions_{grid_loc[0]}_{grid_loc[1]}")
    # elif tile_type == 5:
    #     boxy_terrain(spec, grid_loc, name=f"boxy_terrain_{grid_loc[0]}_{grid_loc[1]}")
    # elif tile_type == 6:
    #     h_field(spec, grid_loc, name=f"h_field_{grid_loc[0]}_{grid_loc[1]}")
    # elif tile_type == 7:
    #     simple_suspended_stair(spec, grid_loc, name=f"sss_{grid_loc[0]}_{grid_loc[1]}")
    # elif tile_type == 8:
    #     sin_suspended_stair(spec, grid_loc, name=f"sinss_{grid_loc[0]}_{grid_loc[1]}")
    # elif tile_type == 9:
    #     stripes(spec, grid_loc, name=f"stripes_{grid_loc[0]}_{grid_loc[1]}")
    # return spec


def make_arena(
    empty_arena_spec: mj.MjSpec,
):
    spec = empty_arena_spec

    # Add lights
    for x in [-1, 1]:
        for y in [-1, 1]:
            spec.worldbody.add_light(pos=[x, y, 40], dir=[-x, -y, -15])

    SQUARE_LENGTH = 2
    #     add_tile(spec=spec, grid_loc=[i * 2 * SQUARE_LENGTH, j * 2 * SQUARE_LENGTH])

    easy_level = [(obstacles.flat, 1)]

    for i in range(-8, 0):
        add_tile(spec=spec, grid_loc=[i * 2 * SQUARE_LENGTH, 0], distribution=easy_level)

    # TODO: continue with partial functions increasing difficulty





