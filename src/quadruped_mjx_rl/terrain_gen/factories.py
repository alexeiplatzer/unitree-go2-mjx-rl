from pathlib import Path

from quadruped_mjx_rl.environments import EnvModel
from quadruped_mjx_rl.environments.physics_pipeline import load_to_spec, spec_to_model
from quadruped_mjx_rl.terrain_gen import (
    add_cylinders,
    add_goal_sphere,
    get_simple_tiled_terrain,
)


def make_simple_obstacle_terrain(empty_scene_path: Path) -> EnvModel:
    spec = load_to_spec(empty_scene_path)
    # add_lights(spec)
    add_cylinders(spec)
    add_goal_sphere(spec)
    return spec_to_model(spec)


def make_empty_terrain(empty_scene_path: Path) -> EnvModel:
    """This is an example of a very simple terrain modification, which just loads the provided
    scene file into an environment model without any additional modifications."""
    spec = load_to_spec(empty_scene_path)
    return spec_to_model(spec)


def make_plain_tiled_terrain(
    empty_scene_path: Path, n_rows: int = 4, n_columns: int = 4, square_size: float = 1.0
) -> EnvModel:
    """Loads a terrain consisting of a square grid of square flat monochromatic tiles. Useful for
    domain randomization of tile properties."""
    spec = load_to_spec(empty_scene_path)
    terrain_config = get_simple_tiled_terrain(n_rows, n_columns, square_size)
    terrain_config.make_arena(spec)
    return spec_to_model(spec)
