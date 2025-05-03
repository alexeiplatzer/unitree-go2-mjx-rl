from quadruped_mjx_rl.configs.config_classes import RobotConfig


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


predefined_robot_configs = {
    "unitree_go2": unitree_go2_config,
    "google_barkour_vb": google_barkour_vb_config,
}
