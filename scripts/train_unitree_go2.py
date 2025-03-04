from quadruped_mjx_rl.training.ppo_enhanced import train as train_enhanced

import paths

if __name__ == "__main__":
    train_enhanced(
        rl_config_path=paths.ppo_enhanced_config,
        robot_config_path=paths.unitree_go2_config,
        init_scene_path=paths.unitree_go2_init_scene,
        model_save_path=paths.TRAINED_POLICIES_DIRECTORY / "teacher_ppo_unitree_go2",
        student_model_save_path=paths.TRAINED_POLICIES_DIRECTORY / "student_ppo_unitree_go2",
        checkpoints_save_path=paths.ckpt_path,
    )
