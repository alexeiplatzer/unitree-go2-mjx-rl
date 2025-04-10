from dataclasses import dataclass
from collections.abc import Callable


@dataclass
class RobotConfig:
    robot_name: str
    scene_file: str
    initial_keyframe: str
    torso_name: str
    upper_leg_name: str
    lower_leg_name: str
    foot_name: str

    @dataclass
    class LimbsConfig:
        front_left: str
        rear_left: str
        front_right: str
        rear_right: str

    lower_leg_bodies: LimbsConfig
    feet_sites: LimbsConfig
    joints_lower_limits: list[float]
    joints_upper_limits: list[float]
    foot_radius: float

    robot_class: str = "Quadruped"


def _names_and_config_classes() -> dict[str, type[RobotConfig]]:
    return {
        "Quadruped": RobotConfig,
    }


def list_robot_classes() -> list[str]:
    return list(_names_and_config_classes().keys())


def get_robot_config_class(robot_class: str) -> type[RobotConfig]:
    if robot_class in _names_and_config_classes():
        return _names_and_config_classes()[robot_class]
    else:
        return RobotConfig


def unitree_go2_config() -> RobotConfig:
    return RobotConfig(
        robot_name="unitree_go2",
        scene_file="scene_mjx.xml",
        initial_keyframe="home",
        torso_name="base",
        upper_leg_name="thigh",
        lower_leg_name="calf",
        foot_name="foot",
        lower_leg_bodies=RobotConfig.LimbsConfig(
            front_left="FL_calf",
            rear_left="RL_calf",
            front_right="FR_calf",
            rear_right="RR_calf",
        ),
        feet_sites=RobotConfig.LimbsConfig(
            front_left="FL_foot",
            rear_left="RL_foot",
            front_right="FR_foot",
            rear_right="RR_foot",
        ),
        joints_lower_limits=[-0.5, 0.4, -2.3],
        joints_upper_limits=[0.5, 1.4, -0.85],
        foot_radius=0.0175,
    )


def google_barkour_vb_config() -> RobotConfig:
    return RobotConfig(
        robot_name="google_barkour_vb",
        scene_file="scene_mjx.xml",
        initial_keyframe="home",
        torso_name="torso",
        upper_leg_name="upper_leg",
        lower_leg_name="lower_leg",
        foot_name="foot",
        lower_leg_bodies=RobotConfig.LimbsConfig(
            front_left="lower_leg_front_left",
            rear_left="lower_leg_hind_left",
            front_right="lower_leg_front_right",
            rear_right="lower_leg_hind_right",
        ),
        feet_sites=RobotConfig.LimbsConfig(
            front_left="foot_front_left",
            rear_left="foot_hind_left",
            front_right="foot_front_right",
            rear_right="foot_hind_right",
        ),
        joints_lower_limits=[-0.7, -1.0, 0.05],
        joints_upper_limits=[0.52, 2.1, 2.1],
        foot_radius=0.0175,
    )


def _names_and_configs() -> dict[str, Callable[[], RobotConfig]]:
    return {
        "unitree_go2": unitree_go2_config,
        "google_barkour_vb": google_barkour_vb_config,
    }


def list_predefined_robots():
    """Lists all robots names for which a robot config is predefined and can be obtained with
    the `get_robot_config` function."""
    return list(_names_and_configs().keys())


def get_robot_config(robot_name: str) -> RobotConfig:
    """Returns the predefined robot config for the robot."""
    if robot_name in _names_and_configs():
        return _names_and_configs()[robot_name]()
    else:
        raise ValueError(
            f"Robot configs not predefined: {robot_name} -- Please define manually."
        )
