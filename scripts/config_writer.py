import paths
from quadruped_mjx_rl import config_utils as cfg
from quadruped_mjx_rl import environments, models, policy_rendering, training, terrain_gen


def make_model_hyperparams_lighter(model_cfg: models.ActorCriticConfig) -> None:
    model_cfg.policy.layer_sizes = [
        model_cfg.policy.layer_sizes[0],
        model_cfg.policy.layer_sizes[-1],
    ]
    model_cfg.value.layer_sizes = [
        model_cfg.value.layer_sizes[0],
        model_cfg.value.layer_sizes[-1],
    ]


def make_training_hyperparams_lighter(training_cfg: training.TrainingConfig) -> None:
    training_cfg.num_timesteps = 100_000
    training_cfg.num_envs = 4
    training_cfg.num_eval_envs = 4
    training_cfg.batch_size = 4
    training_cfg.num_minibatches = 4
    training_cfg.num_evals = 5


if __name__ == "__main__":
    # --- ENVIRONMENTS ---
    # Joystick basic
    terrain_config = terrain_gen.FlatTerrainConfig()
    env_config = environments.JoystickBaseEnvConfig()
    vision_wrapper_config = environments.VisionWrapperConfig()
    cfg.save_configs(
        paths.ENVIRONMENT_CONFIGS_DIRECTORY / "joystick_basic.yaml",
        terrain_config,
        env_config,
        vision_wrapper_config,
    )

    # Joystick with proprioceptive privileged observations
    terrain_config = terrain_gen.FlatTerrainConfig()
    env_config = environments.JoystickBaseEnvConfig()
    env_config.observation_noise.extended_history_length = 45
    env_config.add_privileged_obs = True
    cfg.save_configs(
        paths.ENVIRONMENT_CONFIGS_DIRECTORY / "joystick_teacher_student.yaml",
        terrain_config,
        env_config,
        vision_wrapper_config,
    )

    # Forward moving joystick with rough terrain
    terrain_config = terrain_gen.StripeTilesTerrainConfig()
    env_config = environments.JoystickBaseEnvConfig()
    env_config.domain_rand.apply_kicks = False
    env_config.command.ranges.lin_vel_y_max = 0.0
    env_config.command.ranges.lin_vel_y_min = 0.0
    env_config.command.ranges.ang_vel_yaw_max = 0.0
    env_config.command.ranges.ang_vel_yaw_min = 0.0
    env_config.command.ranges.lin_vel_x_min = 0.0
    cfg.save_configs(
        paths.ENVIRONMENT_CONFIGS_DIRECTORY / "joystick_rough_tiles.yaml",
        terrain_config,
        env_config,
        vision_wrapper_config,
    )

    # Target-reaching and obstacle-avoiding
    terrain_config = terrain_gen.SimpleObstacleTerrainConfig()
    env_config = environments.QuadrupedObstacleAvoidingEnvConfig()
    env_config.domain_rand.apply_kicks = False
    cfg.save_configs(
        paths.ENVIRONMENT_CONFIGS_DIRECTORY / "obstacle_avoiding.yaml",
        terrain_config,
        env_config,
        vision_wrapper_config,
    )

    # Colored terrain map with target goal
    terrain_config = terrain_gen.ColorMapTerrainConfig()
    env_config = environments.QuadrupedVisionTargetEnvConfig()
    env_config.domain_rand.apply_kicks = False
    env_config.observation_noise.history_length = 1
    env_config.observation_noise.extended_history_length = 15
    vision_wrapper_config = environments.ColorGuidedEnvConfig()
    cfg.save_configs(
        paths.ENVIRONMENT_CONFIGS_DIRECTORY / "color_map_guided.yaml",
        terrain_config,
        env_config,
        vision_wrapper_config,
    )

    # Colored terrain map, but joystick
    terrain_config = terrain_gen.ColorMapTerrainConfig()
    terrain_config.add_goal = False
    terrain_config.column_offset = 5  # for occasional backwards movements
    env_config = environments.JoystickBaseEnvConfig()
    env_config.domain_rand.apply_kicks = False
    env_config.observation_noise.history_length = 1
    env_config.observation_noise.extended_history_length = 15
    vision_wrapper_config = environments.ColorGuidedEnvConfig()
    cfg.save_configs(
        paths.ENVIRONMENT_CONFIGS_DIRECTORY / "color_guided_joystick.yaml",
        terrain_config,
        env_config,
        vision_wrapper_config,
    )

    # --- MODELS ---
    # Actor-Critic proprioceptive
    model_config = models.ActorCriticConfig.default()
    training_config = training.TrainingConfig()
    cfg.save_configs(
        paths.MODEL_CONFIGS_DIRECTORY / "basic.yaml",
        model_config,
        training_config,
    )

    training_config.num_envs = 1024
    training_config.num_eval_envs = 1024
    training_config.batch_size = 64
    training_config.num_minibatches = 16
    cfg.save_configs(
        paths.MODEL_CONFIGS_DIRECTORY / "basic_lighter.yaml",
        model_config,
        training_config,
    )

    make_model_hyperparams_lighter(model_config)
    make_training_hyperparams_lighter(training_config)
    cfg.save_configs(
        paths.MODEL_CONFIGS_DIRECTORY / "basic_light.yaml",
        model_config,
        training_config,
    )

    # Teacher-Student proprioceptive
    model_config = models.TeacherStudentConfig.default()
    training_config = training.TrainingConfig()
    cfg.save_configs(
        paths.MODEL_CONFIGS_DIRECTORY / "joystick_teacher_student.yaml",
        model_config,
        training_config,
    )

    make_model_hyperparams_lighter(model_config)
    make_training_hyperparams_lighter(training_config)
    cfg.save_configs(
        paths.MODEL_CONFIGS_DIRECTORY / "joystick_teacher_student_light.yaml",
        model_config,
        training_config,
    )

    # Depth-vision Teacher with RGB-vision Student
    model_config = models.TeacherStudentVisionConfig.default()
    training_config = training.TrainingWithVisionConfig()
    cfg.save_configs(
        paths.MODEL_CONFIGS_DIRECTORY / "teacher_student_vision.yaml",
        model_config,
        training_config,
    )

    make_model_hyperparams_lighter(model_config)
    make_training_hyperparams_lighter(training_config)
    cfg.save_configs(
        paths.MODEL_CONFIGS_DIRECTORY / "teacher_student_vision_light.yaml",
        model_config,
        training_config,
    )

    # Mixed-mode Teacher with RGB Student
    model_config = models.TeacherStudentMixedModeConfig.default()
    training_config = training.TrainingWithVisionConfig()
    cfg.save_configs(
        paths.MODEL_CONFIGS_DIRECTORY / "teacher_student_mixed.yaml",
        model_config,
        training_config,
    )

    make_model_hyperparams_lighter(model_config)
    make_training_hyperparams_lighter(training_config)
    cfg.save_configs(
        paths.MODEL_CONFIGS_DIRECTORY / "teacher_student_mixed_light.yaml",
        model_config,
        training_config,
    )

    # Teacher with recurrent student
    model_config = models.TeacherStudentRecurrentConfig.default()
    training_config = training.TrainingWithRecurrentStudentConfig()
    cfg.save_configs(
        paths.MODEL_CONFIGS_DIRECTORY / "color_guided_recurrent.yaml",
        model_config,
        training_config,
    )

    make_model_hyperparams_lighter(model_config)
    make_training_hyperparams_lighter(training_config)
    cfg.save_configs(
        paths.MODEL_CONFIGS_DIRECTORY / "color_guided_recurrent_light.yaml",
        model_config,
        training_config,
    )

    # Randomized terrain tiles with privileged terrain map encoder, without a student
    model_config = models.ActorCriticMixedModeConfig.default()
    training_config = training.TrainingWithVisionConfig()
    cfg.save_configs(
        paths.MODEL_CONFIGS_DIRECTORY / "color_guided_vision.yaml",
        model_config,
        training_config,
    )

    # Example rendering config
    cfg.save_configs(
        paths.CONFIGS_DIRECTORY / "render_basic_example.yaml",
        policy_rendering.RenderConfig(),
    )
