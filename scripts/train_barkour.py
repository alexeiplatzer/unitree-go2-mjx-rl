from quadruped_mjx_rl.training.ppo_simple import train

import paths

if __name__ == "__main__":
    train(
        rl_config_path=paths.ppo_simple_config,
        robot_config_path=paths.barkour_config,
        init_scene_path=paths.barkour_init_scene,
        model_save_path=paths.TRAINED_POLICIES_DIRECTORY / "try_simple_ppo_barkour",
        checkpoints_save_path=paths.ckpt_path,
    )
