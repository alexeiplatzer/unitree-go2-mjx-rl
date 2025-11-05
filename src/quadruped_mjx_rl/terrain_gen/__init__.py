"""This module allows modifying the terrain of the environment, such as the floor surfaces,
lighting and obstacles. This is achieved by modifying the environment specification which can be
then compiled into the environment model with which the simulator will work."""
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
    """This is an example of a very simple terrain modification, which just loads the provided
    scene file into an environment model without any additional modifications. """
    spec = load_to_spec(empty_scene_path)
    return spec_to_model(spec)
