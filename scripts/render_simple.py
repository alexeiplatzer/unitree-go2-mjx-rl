import paths

from quadruped_mjx_rl import render


if __name__ == "__main__":
    render(
        paths.ppo_simple_config,
        paths.unitree_go2_config,
        paths.render_config,
        init_scene_path=paths.unitree_go2_init_scene,
        trained_model_path=paths.TRAINED_POLICIES_DIRECTORY / "try_simple_ppo_go2",
        animation_save_path=paths.ANIMATIONS_DIRECTORY / "try_teacher_student_unitree_go2.gif",
    )
