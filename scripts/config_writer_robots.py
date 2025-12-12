import paths
from quadruped_mjx_rl import config_utils as cfg
from quadruped_mjx_rl import robots


if __name__ == "__main__":
    for robot_name, robot_config_factory in robots.predefined_robot_configs.items():
        cfg.save_configs(
            paths.ROBOT_CONFIGS_DIRECTORY / f"{robot_name}.yaml", robot_config_factory()
        )
