import paths
from quadruped_mjx_rl import config_utils as cfg
from quadruped_mjx_rl import environments, models, policy_rendering, training, terrain_gen


# TODO: update this and the default values for models
if __name__ == "__main__":
    # Joystick basic
    terrain_config = terrain_gen.FlatTerrainConfig()
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

    # Joystick with proprioceptive privileged observations
    terrain_config = terrain_gen.FlatTerrainConfig()
    env_config = environments.JoystickBaseEnvConfig()
    env_config.observation_noise.extended_history_length = 45
    env_config.observation_noise.add_privileged_obs = True
    model_config = models.TeacherStudentConfig.default()
    training_config = training.TrainingConfig()
    training_config.optimizer = training.TeacherStudentOptimizerConfig()
    cfg.save_configs(
        paths.CONFIGS_DIRECTORY / "joystick_teacher_student.yaml",
        terrain_config,
        env_config,
        model_config,
        training_config,
    )

    # Target-reaching basic
    terrain_config = terrain_gen.FlatTerrainConfig()
    terrain_config.add_goal = True
    env_config = environments.QuadrupedVisionTargetEnvConfig()
    model_config = models.ActorCriticEnrichedConfig.default()
    training_config = training.TrainingConfig()
    cfg.save_configs(
        paths.CONFIGS_DIRECTORY / "target_reaching_basic.yaml",
        terrain_config,
        env_config,
        model_config,
        training_config,
    )

    # Basic Target-reaching with vision
    terrain_config = terrain_gen.FlatTerrainConfig()
    terrain_config.add_goal = True
    env_config = environments.QuadrupedVisionTargetEnvConfig()
    model_config = models.ActorCriticEnrichedConfig.default_vision()
    training_config = training.TrainingConfig.default_vision()
    vision_wrapper_config = environments.VisionWrapperConfig()
    cfg.save_configs(
        paths.CONFIGS_DIRECTORY / "target_reaching_vision.yaml",
        terrain_config,
        env_config,
        model_config,
        training_config,
        vision_wrapper_config,
    )

    # Target-reaching and obstacle-avoiding
    terrain_config = terrain_gen.SimpleObstacleTerrainConfig()
    terrain_config.add_goal = True
    env_config = environments.QuadrupedObstacleAvoidingEnvConfig()
    env_config.domain_rand.apply_kicks = False
    model_config = models.TeacherStudentConfig.default_mixed()
    training_config = training.TrainingConfig.default_vision()
    training_config.optimizer = training.TeacherStudentOptimizerConfig()
    training_config.optimizer.max_grad_norm = 1.0
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

    # COLORED TERRAIN MAPS
    terrain_config_joystick = terrain_gen.ColorMapTerrainConfig(add_goal=False)
    terrain_config_joystick.column_offset = 5  # for occasional backwards movements
    terrain_config_target = terrain_gen.ColorMapTerrainConfig(add_goal=True)

    # Colored terrain map, blind joystick
    env_config = environments.JoystickBaseEnvConfig()
    env_config.domain_rand.apply_kicks = False
    model_config = models.ActorCriticConfig.default()
    training_config = training.TrainingConfig()
    cfg.save_configs(
        paths.CONFIGS_DIRECTORY / "color_map_blind_joystick.yaml",
        terrain_config_joystick,
        env_config,
        model_config,
        training_config,
    )

    # Plain rgb vision target-reaching
    env_config = environments.QuadrupedVisionTargetEnvConfig()
    env_config.domain_rand.apply_kicks = False
    model_config = models.ActorCriticEnrichedConfig.default_vision()
    training_config = training.TrainingConfig.default_vision()
    vision_wrapper_config = environments.VisionWrapperConfig()
    cfg.save_configs(
        paths.CONFIGS_DIRECTORY / "color_map_plain_rgb_target.yaml",
        terrain_config_target,
        env_config,
        model_config,
        training_config,
        vision_wrapper_config,
    )

    # # Privileged terrain map, teacher only
    #
    # # Colored terrain map with target goal
    # terrain_config = terrain_gen.ColorMapTerrainConfig()
    # env_config = environments.QuadrupedVisionTargetEnvConfig()
    # env_config.domain_rand.apply_kicks = False
    # env_config.observation_noise.history_length = 1
    # env_config.observation_noise.extended_history_length = 15
    # vision_wrapper_config = environments.ColorGuidedEnvConfig()
    # cfg.save_configs(
    #     paths.CONFIGS_DIRECTORY / "color_map_guided.yaml",
    #     terrain_config,
    #     env_config,
    #     vision_wrapper_config,
    # )
    #
    # # Colored terrain map, but joystick
    # terrain_config = terrain_gen.ColorMapTerrainConfig()
    # terrain_config.add_goal = False
    # terrain_config.column_offset = 5  # for occasional backwards movements
    # env_config = environments.JoystickBaseEnvConfig()
    # env_config.domain_rand.apply_kicks = False
    # env_config.observation_noise.history_length = 1
    # env_config.observation_noise.extended_history_length = 15
    # vision_wrapper_config = environments.ColorGuidedEnvConfig()
    # cfg.save_configs(
    #     paths.ENVIRONMENT_CONFIGS_DIRECTORY / "color_map_joystick.yaml",
    #     terrain_config,
    #     env_config,
    #     vision_wrapper_config,
    # )
    #
    #
    # # Mixed-mode Teacher with RGB Student
    # model_config = models.TeacherStudentMixedModeConfig.default()
    # training_config = training.TrainingWithVisionConfig()
    # cfg.save_configs(
    #     paths.MODEL_CONFIGS_DIRECTORY / "teacher_student_mixed.yaml",
    #     model_config,
    #     training_config,
    # )
    #
    # # Teacher with recurrent student
    # model_config = models.TeacherStudentRecurrentConfig.default()
    # training_config = training.TrainingWithRecurrentStudentConfig()
    # cfg.save_configs(
    #     paths.MODEL_CONFIGS_DIRECTORY / "color_guided_recurrent.yaml",
    #     model_config,
    #     training_config,
    # )
    #
    # # Randomized terrain tiles with privileged terrain map encoder, without a student
    # model_config = models.ActorCriticMixedModeConfig.default()
    # training_config = training.TrainingWithVisionConfig()
    # cfg.save_configs(
    #     paths.MODEL_CONFIGS_DIRECTORY / "color_guided_vision.yaml",
    #     model_config,
    #     training_config,
    # )

    # Example rendering config
    cfg.save_configs(
        paths.CONFIGS_DIRECTORY / "render_basic_example.yaml",
        policy_rendering.RenderConfig(),
    )
