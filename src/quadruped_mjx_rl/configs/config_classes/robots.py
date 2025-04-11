from dataclasses import dataclass


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


robot_config_classes = {
    "Quadruped": RobotConfig,
}
