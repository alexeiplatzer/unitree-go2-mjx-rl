import paths

from brax.io import model

from quadruped_mjx_rl import render


if __name__ == "__main__":
    render(
        paths.ppo_enhanced_config,
        paths.unitree_go2_config,
        paths.render_config,
        init_scene_path=paths.unitree_go2_init_scene,
        trained_model_path=paths.TRAINED_POLICIES_DIRECTORY / "ppo_teacher_student_policy",
        animation_save_path={
            "teacher": paths.ANIMATIONS_DIRECTORY / "unitree_go2_teacher_try2.gif",
            "student": paths.ANIMATIONS_DIRECTORY / "unitree_go2_student_try2.gif",
        },
    )
