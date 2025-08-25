import numpy as np
import paths
from quadruped_mjx_rl.robots import predefined_robot_configs
from quadruped_mjx_rl.environments import JoystickBaseEnvConfig
from quadruped_mjx_rl.models import ActorCriticConfig
from quadruped_mjx_rl.training.configs import TrainingConfig
from quadruped_mjx_rl.environments.physics_pipeline import load_to_spec, spec_to_model
from quadruped_mjx_rl.environments import resolve_env_class, get_env_factory
from quadruped_mjx_rl.training.train_interface import train
from quadruped_mjx_rl.domain_randomization.randomized_physics import domain_randomize


if __name__ == "__main__":
    # More legible printing from numpy.
    np.set_printoptions(precision=3, suppress=True, linewidth=100)

    # Prepare configs
    robot_config = predefined_robot_configs["unitree_go2"]()

    env_config = JoystickBaseEnvConfig()

    model_config = ActorCriticConfig(
        modules=ActorCriticConfig.ModulesConfig(
            policy=[128, 128, 128, 128, 128],
            value=[256, 256, 256, 256, 256],
        ),
    )

    training_config = TrainingConfig(num_timesteps=1_00_000)

    init_scene_path = paths.unitree_go2_init_scene

    env_model = spec_to_model(load_to_spec(init_scene_path))

    env_class = resolve_env_class(env_config)
    env_model = env_class.customize_model(env_model, env_config)
    env_factory = get_env_factory(
        robot_config=robot_config,
        environment_config=env_config,
        env_class=env_class,
        env_model=env_model,
    )

    policy_factories, trained_params, evaluation_metrics = train(
        training_config=training_config,
        model_config=model_config,
        training_env=env_factory(),
        evaluation_env=env_factory(),
        randomization_fn=domain_randomize,
    )