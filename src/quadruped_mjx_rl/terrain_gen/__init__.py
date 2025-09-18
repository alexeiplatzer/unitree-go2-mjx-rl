from pathlib import Path
from quadruped_mjx_rl.terrain_gen.tile import TerrainConfig
from quadruped_mjx_rl.terrain_gen.simple import add_cylinders, add_goal_sphere, add_lights
from quadruped_mjx_rl.environments.physics_pipeline import EnvModel, load_to_spec, spec_to_model


def make_simple_obstacle_terrain(empty_scene_path: Path) -> EnvModel:
    spec = load_to_spec(empty_scene_path)
    # add_lights(spec)
    add_cylinders(spec)
    add_goal_sphere(spec)
    return spec_to_model(spec)


def make_empty_terrain(empty_scene_path: Path) -> EnvModel:
    spec = load_to_spec(empty_scene_path)
    return spec_to_model(spec)
