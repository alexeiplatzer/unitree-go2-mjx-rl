import paths

from quadruped_mjx_rl.render_ppo_simple import render


if __name__ == "__main__":
    render(
        trained_model_path=paths.TRAINED_POLICIES_DIRECTORY / "try_simple_ppo_unitree_go2",
        render_config_path=paths.CONFIGS_DIRECTORY / "render_ppo_simple.yaml",
        rl_config_path=paths.ppo_simple_config,
        robot_config_path=paths.unitree_go2_config,
        init_scene_path=paths.unitree_go2_init_scene,
        save_animation=True,
        animation_save_path=paths.ANIMATIONS_DIRECTORY / "try_simple_ppo_unitree_go2.gif",
    )
