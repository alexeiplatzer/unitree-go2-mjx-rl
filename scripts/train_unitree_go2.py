from quadruped_mjx_rl.configs import prepare_configs, save_configs, ConfigKey
from quadruped_mjx_rl.configs import predefined_robot_configs
from quadruped_mjx_rl import configs
from quadruped_mjx_rl.training import train

import paths

if __name__ == "__main__":
    configs = prepare_configs(
        training=configs.TrainingConfig(),
        robot=predefined_robot_configs["unitree_go2"](),
        environment=configs.EnvironmentConfig()
    )
    configs[ConfigKey("training")].learning_rate = 1e-2
    save_configs("here.yaml", *configs.values())
    # train(
    #     rl_config_path=paths.ppo_enhanced_config,
    #     robot_config_path=paths.unitree_go2_config,
    #     init_scene_path=paths.unitree_go2_init_scene,
    #     model_save_path=paths.TRAINED_POLICIES_DIRECTORY / "teacher_ppo_unitree_go2",
    #     student_model_save_path=paths.TRAINED_POLICIES_DIRECTORY / "student_ppo_unitree_go2",
    #     checkpoints_save_path=paths.ckpt_path,
    # )
