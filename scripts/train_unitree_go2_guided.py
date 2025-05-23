from quadruped_mjx_rl.config_utils import prepare_configs, ConfigKey
from quadruped_mjx_rl.environments import get_env_factory
from quadruped_mjx_rl.training import train

import jax

# from jax import config
# config.update("jax_debug_nans", True)
# config.update("jax_enable_x64", True)
# config.update('jax_default_matmul_precision', "high")

import paths


if __name__ == "__main__":
    configs = prepare_configs(
        paths.CONFIGS_DIRECTORY / "unitree_go2.yaml",
        paths.CONFIGS_DIRECTORY / "vision_ppo.yaml",
    )
    env_factory, uses_vision = get_env_factory(
        robot_config=configs[ConfigKey.ROBOT],
        init_scene_path=paths.unitree_go2_init_scene,
        env_config=configs[ConfigKey.ENVIRONMENT],
    )
    train(
        env_factory=env_factory,
        model_config=configs[ConfigKey.MODEL],
        model_save_path=paths.TRAINED_POLICIES_DIRECTORY / "guided_ppo_unitree_go2_trial",
        training_config=configs[ConfigKey.TRAINING],
        vision=uses_vision,
        vision_config=configs.get(ConfigKey.VISION),
    )
