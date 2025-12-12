import paths
from quadruped_mjx_rl import config_utils as cfg
from quadruped_mjx_rl import environments, models, policy_rendering, robots
from quadruped_mjx_rl.training.configs import TrainingConfig, TrainingWithVisionConfig


if __name__ == "__main__":

    # --- Robot configs ---
    for robot_name, robot_config_factory in robots.predefined_robot_configs.items():
        cfg.save_configs(
            paths.ROBOT_CONFIGS_DIRECTORY / f"{robot_name}.yaml", robot_config_factory()
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

    cfg.save_configs(
        paths.CONFIGS_DIRECTORY / "vision_ppo_example.yaml",
        vision_ppo_env_config,
        vision_ppo_model_config,
        vision_ppo_training_config,
    )

    # --- Example rendering config ---
    cfg.save_configs(
        paths.CONFIGS_DIRECTORY / "render_basic_example.yaml",
        policy_rendering.RenderConfig(),
    )
