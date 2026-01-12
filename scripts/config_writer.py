import paths
from quadruped_mjx_rl import config_utils as cfg
from quadruped_mjx_rl import environments, models, policy_rendering, training, terrain_gen
from quadruped_mjx_rl.domain_randomization import (
    ColorMapRandomizationConfig,
    SurfaceDomainRandomizationConfig,
)
from quadruped_mjx_rl.environments import JoystickBaseEnvConfig
from quadruped_mjx_rl.terrain_gen import ColorMapTerrainConfig

# TODO: update this and the default values for models
if __name__ == "__main__":
    # PLAIN TERRAIN JOYSTICK CONFIGS
    # Joystick basic DEBUG SURFACE
    terrain_config = terrain_gen.FlatTerrainConfig(
        randomization_config=SurfaceDomainRandomizationConfig.default_easier()
    )
    env_config = environments.JoystickBaseEnvConfig()
    model_config = models.ActorCriticConfig.default()
    training_config = training.TrainingConfig()
    cfg.save_configs(
        paths.CONFIGS_DIRECTORY / "joystick_basic.yaml",
        terrain_config,
        env_config,
        model_config,
        training_config,
    )

    # Joystick difficult DEBUG SURFACE
    terrain_config = terrain_gen.FlatTerrainConfig(
        randomization_config=SurfaceDomainRandomizationConfig()
    )
    env_config = environments.JoystickBaseEnvConfig()
    model_config = models.ActorCriticConfig.default()
    training_config = training.TrainingConfig()
    cfg.save_configs(
        paths.CONFIGS_DIRECTORY / "joystick_difficult.yaml",
        terrain_config,
        env_config,
        model_config,
        training_config,
    )

    # Joystick with proprioceptive privileged observations
    terrain_config = terrain_gen.FlatTerrainConfig()
    env_config = environments.JoystickBaseEnvConfig()
    env_config.observation_noise.extended_history_length = 45
    env_config.observation_noise.add_privileged_obs = True
    model_config = models.TeacherStudentConfig.default()
    training_config = training.TrainingConfig()
    training_config.optimizer = training.TeacherStudentOptimizerConfig.default()
    cfg.save_configs(
        paths.CONFIGS_DIRECTORY / "joystick_teacher_student.yaml",
        terrain_config,
        env_config,
        model_config,
        training_config,
    )

    # PLAIN TERRAIN TARGET REACHING
    terrain_config = terrain_gen.FlatTerrainConfig(add_goal=True)
    env_config = environments.QuadrupedVisionTargetEnvConfig()

    # Target-reaching with no observations pointing to the ball
    model_config = models.ActorCriticConfig.default()
    training_config = training.TrainingConfig()
    cfg.save_configs(
        paths.CONFIGS_DIRECTORY / "target_reaching_clueless.yaml",
        terrain_config,
        env_config,
        model_config,
        training_config,
    )

    # Target-reaching basic, given direction vector
    model_config = models.ActorCriticEnrichedConfig.default()
    training_config = training.TrainingConfig()
    cfg.save_configs(
        paths.CONFIGS_DIRECTORY / "target_reaching_basic.yaml",
        terrain_config,
        env_config,
        model_config,
        training_config,
    )

    # Target-reaching with naive rgb vision input
    model_config = models.ActorCriticEnrichedConfig.default_vision()
    training_config = training.TrainingConfig()
    training_config.vision_config = training.VisionConfig()
    vision_wrapper_config = environments.VisionWrapperConfig()
    cfg.save_configs(
        paths.CONFIGS_DIRECTORY / "target_reaching_vision.yaml",
        terrain_config,
        env_config,
        model_config,
        training_config,
        vision_wrapper_config,
    )

    # direction vector teacher with rgb vision student
    model_config = models.TeacherStudentConfig.default()
    model_config.encoder = models.ModuleConfigMLP(layer_sizes=[256], obs_key="goal_direction")
    model_config.student = models.ModuleConfigCNN(
        filter_sizes=[16, 24, 32],
        obs_key="pixels/frontal_ego/rgb_adjusted",
        dense=models.ModuleConfigMLP(layer_sizes=[256])
    )
    training_config = training.TrainingConfig()
    training_config.vision_config = training.VisionConfig(vision_obs_period=1)
    training_config.optimizer = training.TeacherStudentOptimizerConfig.default()
    cfg.save_configs(
        paths.CONFIGS_DIRECTORY / "target_reaching_vision_student.yaml",
        terrain_config,
        env_config,
        model_config,
        training_config,
        vision_wrapper_config,
    )

    # COLORED TERRAIN MAPS
    # Joystick
    terrain_config_joystick = terrain_gen.ColorMapTerrainConfig.default_joystick()
    env_config_joystick = environments.JoystickBaseEnvConfig()
    env_config_joystick.domain_rand.apply_kicks = False
    env_config_joystick.command.ranges = JoystickBaseEnvConfig.CommandConfig.RangesConfig(
        lin_vel_x_max=1.5,
        lin_vel_x_min=0.1,
        lin_vel_y_max=0.0,
        lin_vel_y_min=0.0,
        ang_vel_yaw_max=0.0,
        ang_vel_yaw_min=-0.0,
    )

    # Target-reaching
    terrain_config_target = terrain_gen.ColorMapTerrainConfig(add_goal=True)
    env_config_target = environments.QuadrupedVisionTargetEnvConfig()
    env_config_target.domain_rand.apply_kicks = False

    # Colored terrain map, blind joystick
    model_config = models.ActorCriticConfig.default()
    training_config = training.TrainingConfig()
    cfg.save_configs(
        paths.CONFIGS_DIRECTORY / "color_map_blind_joystick.yaml",
        terrain_config_joystick,
        env_config_joystick,
        model_config,
        training_config,
    )

    # Colored terrain map, blind joystick, no surface values randomized baseline
    # This serves to compare, whether the blind joystick defined above experiences any
    # difficulty at all from our terrain randomization.
    terrain_config_joystick_baseline = terrain_gen.ColorMapTerrainConfig.default_joystick()
    terrain_config_joystick_baseline.randomization_config = ColorMapRandomizationConfig(
        friction_min=1.00, friction_max=1.00, stiffness_min=0.02, stiffness_max=0.02
    )
    model_config = models.ActorCriticConfig.default()
    training_config = training.TrainingConfig()
    cfg.save_configs(
        paths.CONFIGS_DIRECTORY / "color_map_blind_joystick_baseline.yaml",
        terrain_config_joystick_baseline,
        env_config_joystick,
        model_config,
        training_config,
    )

    # Direction vector guided target reaching
    model_config = models.ActorCriticEnrichedConfig.default()
    training_config = training.TrainingConfig()
    cfg.save_configs(
        paths.CONFIGS_DIRECTORY / "color_map_blind_target.yaml",
        terrain_config_target,
        env_config_target,
        model_config,
        training_config,
    )

    # Plain rgb vision target-reaching
    model_config = models.ActorCriticEnrichedConfig.default_vision()
    training_config = training.TrainingConfig()
    training_config.vision_config = training.VisionConfig()
    vision_wrapper_config = environments.VisionWrapperConfig()
    cfg.save_configs(
        paths.CONFIGS_DIRECTORY / "color_map_plain_rgb_target.yaml",
        terrain_config_target,
        env_config_target,
        model_config,
        training_config,
        vision_wrapper_config,
    )

    # Privileged terrain map, teacher only joystick
    model_config = models.ActorCriticEnrichedConfig.default_privileged_map()
    training_config = training.TrainingConfig()
    training_config.vision_config = training.VisionConfig()
    vision_wrapper_config = environments.ColorGuidedEnvConfig()
    cfg.save_configs(
        paths.CONFIGS_DIRECTORY / "color_map_privileged_joystick.yaml",
        terrain_config_joystick,
        env_config_joystick,
        model_config,
        training_config,
        vision_wrapper_config,
    )

    # Privileged terrain map, teacher only target-reaching
    model_config = models.ActorCriticEnrichedConfig.default_mixed()
    training_config = training.TrainingConfig()
    training_config.vision_config = training.VisionConfig()
    vision_wrapper_config = environments.ColorGuidedEnvConfig()
    cfg.save_configs(
        paths.CONFIGS_DIRECTORY / "color_map_privileged_target.yaml",
        terrain_config_target,
        env_config_target,
        model_config,
        training_config,
        vision_wrapper_config,
    )

    # Mixed-mode Teacher with RGB Student
    model_config = models.TeacherStudentConfig.default_mixed()
    training_config = training.TrainingConfig()
    training_config.vision_config = training.VisionConfig(vision_obs_period=4)
    training_config.optimizer = training.TeacherStudentOptimizerConfig.default()
    vision_wrapper_config = environments.ColorGuidedEnvConfig()
    cfg.save_configs(
        paths.CONFIGS_DIRECTORY / "color_map_vision_student.yaml",
        terrain_config_target,
        env_config_target,
        model_config,
        training_config,
        vision_wrapper_config,
    )

    # Recurrent student joystick
    env_config_joystick.observation_noise.history_length = 1
    env_config_joystick.observation_noise.extended_history_length = 15
    model_config = models.TeacherStudentRecurrentConfig.default()
    model_config.encoder = model_config.encoder.vision_preprocessing  # no extra goal vector
    training_config = training.TrainingConfig()
    training_config.vision_config = training.VisionConfig()
    training_config.optimizer = training.TeacherStudentOptimizerConfig.default()
    vision_wrapper_config = environments.ColorGuidedEnvConfig()
    cfg.save_configs(
        paths.CONFIGS_DIRECTORY / "color_guided_joystick.yaml",
        terrain_config_joystick,
        env_config_joystick,
        model_config,
        training_config,
        vision_wrapper_config,
    )

    # Recurrent Student Target
    env_config_target.observation_noise.history_length = 1
    env_config_target.observation_noise.extended_history_length = 15
    model_config = models.TeacherStudentRecurrentConfig.default()
    training_config = training.TrainingConfig()
    training_config.vision_config = training.VisionConfig()
    training_config.optimizer = training.TeacherStudentOptimizerConfig.default()
    vision_wrapper_config = environments.ColorGuidedEnvConfig()
    cfg.save_configs(
        paths.CONFIGS_DIRECTORY / "color_guided_target.yaml",
        terrain_config_target,
        env_config_target,
        model_config,
        training_config,
        vision_wrapper_config,
    )

    # OBSTACLE AVOIDING: speculative work
    # Target-reaching and obstacle-avoiding
    terrain_config = terrain_gen.SimpleObstacleTerrainConfig()
    terrain_config.add_goal = True
    env_config = environments.QuadrupedObstacleAvoidingEnvConfig()
    env_config.domain_rand.apply_kicks = False
    model_config = models.TeacherStudentConfig.default_mixed()
    training_config = training.TrainingConfig()
    training_config.vision_config = training.VisionConfig()
    training_config.optimizer = training.TeacherStudentOptimizerConfig.default()
    vision_wrapper_config = environments.VisionWrapperConfig()
    vision_wrapper_config.camera_inputs[0].use_depth = True
    cfg.save_configs(
        paths.CONFIGS_DIRECTORY / "obstacle_avoiding.yaml",
        terrain_config,
        env_config,
        model_config,
        training_config,
        vision_wrapper_config,
    )

    # Example rendering config
    cfg.save_configs(
        paths.CONFIGS_DIRECTORY / "render_basic_example.yaml",
        policy_rendering.RenderConfig(),
    )
