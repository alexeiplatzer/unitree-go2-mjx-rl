from quadruped_mjx_rl.training.ppo_simple import train

import paths

if __name__ == '__main__':
    train(
        rl_config_path=paths.ppo_simple_config,
        robot_config_path=paths.unitree_go2_config,
        init_scene_path=paths.unitree_go2_init_scene,
        model_save_path=paths.TRAINED_POLICIES_DIRECTORY / "simple_ppo_unitree_go2",
        checkpoints_save_path=paths.ckpt_path,
    )
