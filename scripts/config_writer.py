from quadruped_mjx_rl import config_utils as cfg
from quadruped_mjx_rl import robots, environments, models, policy_rendering
from quadruped_mjx_rl.environments.vision import robotic_vision
from quadruped_mjx_rl.training.configs import TrainingConfig, TrainingWithVisionConfig

import paths


if __name__ == "__main__":

    # --- Robot configs ---
    unitree_go2_robot_config = robots.unitree_go2_config()
    cfg.save_configs(
        paths.CONFIGS_DIRECTORY / "unitree_go2.yaml",
        unitree_go2_robot_config,
    )

    google_barkour_vb_robot_config = robots.google_barkour_vb_config()
    cfg.save_configs(
        paths.CONFIGS_DIRECTORY / "google_barkour_vb.yaml",
        google_barkour_vb_robot_config,
    )

    # --- Reinforcement learning configs ---
    raw_ppo_env_config = environments.JoystickBaseEnvConfig()
    raw_ppo_model_config = models.ActorCriticConfig()

    guided_ppo_env_config = environments.TeacherStudentEnvironmentConfig()
    guided_ppo_model_config = models.TeacherStudentConfig()

    ppo_training_config = TrainingConfig()

    cfg.save_configs(
        paths.CONFIGS_DIRECTORY / "raw_ppo_example.yaml",
        raw_ppo_env_config,
        raw_ppo_model_config,
        ppo_training_config,
    )

    cfg.save_configs(
        paths.CONFIGS_DIRECTORY / "guided_ppo_example.yaml",
        guided_ppo_env_config,
        guided_ppo_model_config,
        ppo_training_config,
    )

    vision_ppo_env_config = environments.QuadrupedJoystickVisionEnvConfig()
    vision_ppo_model_config = models.TeacherStudentVisionConfig()
    vision_ppo_training_config = TrainingWithVisionConfig()
    vision_config = robotic_vision.VisionConfig()

    cfg.save_configs(
        paths.CONFIGS_DIRECTORY / "vision_ppo_example.yaml",
        vision_ppo_env_config,
        vision_ppo_model_config,
        vision_ppo_training_config,
        vision_config,
    )

    # --- Example rendering config ---
    cfg.save_configs(
        paths.CONFIGS_DIRECTORY / "render_basic_example.yaml",
        policy_rendering.RenderConfig(),
    )
