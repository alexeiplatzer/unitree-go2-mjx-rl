from quadruped_mjx_rl.config_utils import prepare_configs, ConfigKey
from quadruped_mjx_rl.environments import get_env_factory
from quadruped_mjx_rl.training import train

import paths

if __name__ == "__main__":
    configs = prepare_configs(
        paths.CONFIGS_DIRECTORY / "google_barkour_vb.yaml",
        paths.CONFIGS_DIRECTORY / "raw_ppo.yaml",
    )
    env_factory = get_env_factory(
        robot_config=configs[ConfigKey.ROBOT],
        init_scene_path=paths.unitree_go2_init_scene,
        env_config=configs[ConfigKey.ENVIRONMENT],
    )
    train(
        env_factory=env_factory,
        model_config=configs[ConfigKey.MODEL],
        model_save_path=paths.TRAINED_POLICIES_DIRECTORY / "raw_ppo_barkour_vb_trial",
        training_config=configs[ConfigKey.TRAINING],
    )
