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

    # Joystick Basic
    terrain_config = terrain_gen.FlatTerrainConfig()
    env_config = environments.JoystickBaseEnvConfig()
    model_config = models.ActorCriticConfig()
    training_config = training.TrainingConfig()
    cfg.save_configs(
        paths.CONFIGS_DIRECTORY / "joystick_basic_ppo.yaml",
        terrain_config,
        env_config,
        model_config,
        training_config,
    )

    make_model_hyperparams_lighter(model_config)
    make_training_hyperparams_lighter(training_config)
    cfg.save_configs(
        paths.CONFIGS_DIRECTORY / "joystick_basic_ppo_light.yaml",
        terrain_config,
        env_config,
        model_config,
        training_config,
    )

    # Joystick Teacher-Student Proprioceptive
    terrain_config = terrain_gen.FlatTerrainConfig()
    env_config = environments.TeacherStudentEnvironmentConfig()
    model_config = models.TeacherStudentConfig()
    training_config = training.TrainingConfig()
    cfg.save_configs(
        paths.CONFIGS_DIRECTORY / "joystick_teacher_student.yaml",
        terrain_config,
        env_config,
        model_config,
        training_config,
    )

    make_model_hyperparams_lighter(model_config)
    make_training_hyperparams_lighter(training_config)
    cfg.save_configs(
        paths.CONFIGS_DIRECTORY / "joystick_teacher_student_light.yaml",
        terrain_config,
        env_config,
        model_config,
        training_config,
    )

    # Joystick Depth-vision Teacher with RGB-vision Student
    terrain_config = terrain_gen.SimpleObstacleTerrainConfig()
    env_config = environments.QuadrupedJoystickVisionEnvConfig()
    model_config = models.TeacherStudentVisionConfig()
    training_config = training.TrainingWithVisionConfig()
    cfg.save_configs(
        paths.CONFIGS_DIRECTORY / "teacher_student_vision.yaml",
        terrain_config,
        env_config,
        model_config,
        training_config,
    )

    make_model_hyperparams_lighter(model_config)
    make_training_hyperparams_lighter(training_config)
    cfg.save_configs(
        paths.CONFIGS_DIRECTORY / "teacher_student_vision_light.yaml",
        terrain_config,
        env_config,
        model_config,
        training_config,
    )

    # Obstacle-avoiding Depth-teacher RGB-student
    terrain_config = terrain_gen.SimpleObstacleTerrainConfig()
    env_config = environments.QuadrupedObstacleAvoidingEnvConfig()
    model_config = models.TeacherStudentVisionConfig()
    training_config = training.TrainingWithVisionConfig()
    cfg.save_configs(
        paths.CONFIGS_DIRECTORY / "obstacle_avoiding_vision.yaml",
        terrain_config,
        env_config,
        model_config,
        training_config,
    )

    make_model_hyperparams_lighter(model_config)
    make_training_hyperparams_lighter(training_config)
    cfg.save_configs(
        paths.CONFIGS_DIRECTORY / "obstacle_avoiding_vision_light.yaml",
        terrain_config,
        env_config,
        model_config,
        training_config,
    )

    # Randomized terrain tiles Depth-teacher Recurrent Student
    terrain_config = terrain_gen.ColorMapTerrainConfig()
    env_config = environments.QuadrupedColorGuidedEnvConfig()
    model_config = models.TeacherStudentRecurrentConfig()
    training_config = training.TrainingWithRecurrentStudentConfig()
    cfg.save_configs(
        paths.CONFIGS_DIRECTORY / "color_guided_recurrent.yaml",
        terrain_config,
        env_config,
        model_config,
        training_config,
    )

    make_model_hyperparams_lighter(model_config)
    make_training_hyperparams_lighter(training_config)
    cfg.save_configs(
        paths.CONFIGS_DIRECTORY / "color_guided_recurrent_light.yaml",
        terrain_config,
        env_config,
        model_config,
        training_config,
    )

    # Example rendering config
    cfg.save_configs(
        paths.CONFIGS_DIRECTORY / "render_basic_example.yaml",
        policy_rendering.RenderConfig(),
    )
