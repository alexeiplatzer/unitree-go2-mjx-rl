from quadruped_mjx_rl.config_utils import prepare_configs
from quadruped_mjx_rl.environments import EnvironmentConfig, get_env_factory
from quadruped_mjx_rl.models import ModelConfig
from quadruped_mjx_rl.robots import RobotConfig
from quadruped_mjx_rl.training.training import train, TrainingConfig

# from jax import config
# config.update("jax_debug_nans", True)
# config.update("jax_enable_x64", True)
# config.update('jax_default_matmul_precision', "high")

import paths


if __name__ == "__main__":
    configs = prepare_configs(
        paths.CONFIGS_DIRECTORY / "unitree_go2.yaml",
        paths.CONFIGS_DIRECTORY / "raw_ppo.yaml",
    )
    robot_config = configs.get("robot")
    assert isinstance(robot_config, RobotConfig)
    env_config = configs.get("environment")
    assert isinstance(env_config, EnvironmentConfig)
    model_config = configs.get("model")
    assert isinstance(model_config, ModelConfig)
    training_config = configs.get("training")
    assert isinstance(training_config, TrainingConfig)
    training_config.timesteps = 1000
    env_factory, _ = get_env_factory(
        robot_config=robot_config,
        init_scene_path=paths.unitree_go2_init_scene,
        env_config=env_config,
    )
    train(
        env_factory=env_factory,
        model_config=model_config,
        model_save_path=paths.TRAINED_POLICIES_DIRECTORY / "raw_ppo_unitree_go2_trial",
        training_config=training_config,
    )
