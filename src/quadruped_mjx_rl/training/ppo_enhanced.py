# Supporting
import functools
from datetime import datetime
import matplotlib.pyplot as plt

# Math
from flax.training import orbax_utils
from orbax import checkpoint as ocp

# Sim

# Brax
from brax import envs
from brax.io import model

# Algorithm
from ..brax_alt.training.agents.teacher import train as teacher_train
from ..brax_alt.training.agents.teacher import networks as teacher_networks

from .domain_randomization import domain_randomize
from ..training_environments.go2_teacher import Go2TeacherEnv
from ..utils import load_config_dicts


def train(
    rl_config_path,
    robot_config_path,
    init_scene_path,
    model_save_path,
    checkpoints_save_path,
):

    def policy_params_fn(current_step, make_policy, parameters):
        # save checkpoints
        orbax_checkpointer = ocp.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(parameters)
        path = checkpoints_save_path / f"{current_step}"
        orbax_checkpointer.save(path, parameters, force=True, save_args=save_args)

    def progress(num_steps, metrics):
        times.append(datetime.now())
        x_data.append(num_steps)
        y_data.append(metrics["eval/episode_reward"])
        ydataerr.append(metrics["eval/episode_reward_std"])

        plt.xlim([0, train_fn.keywords["num_timesteps"] * 1.25])
        plt.ylim([min_y, max_y])

        plt.xlabel("# environment steps")
        plt.ylabel("reward per episode")
        plt.title(f"y={y_data[-1]:.3f}")

        plt.errorbar(x_data, y_data, yerr=ydataerr)
        plt.show()

    # Load configs
    rl_config, robot_config = load_config_dicts(rl_config_path, robot_config_path)

    env_name = "joystick_go2"
    envs.register_environment(env_name, Go2TeacherEnv)
    env_config = rl_config.environment
    env = envs.get_environment(
        env_name,
        environment_config=env_config,
        robot_config=robot_config,
        init_scene_path=init_scene_path,
    )

    make_networks_factory = functools.partial(
        teacher_networks.make_teacher_networks,
        latent_representation_size=rl_config.model.latent_size,
        policy_hidden_layer_sizes=tuple(rl_config.model.policy_hidden_sizes),
        value_hidden_layer_sizes=tuple(rl_config.model.value_hidden_sizes),
        encoder_hidden_layer_sizes=tuple(rl_config.model.encoder_hidden_sizes),
    )
    training_config = rl_config.training
    train_fn = functools.partial(
        teacher_train.train,
        network_factory=make_networks_factory,
        randomization_fn=domain_randomize,
        policy_params_fn=policy_params_fn,
        seed=0,
        **training_config,
    )

    x_data = []
    y_data = []
    ydataerr = []
    times = [datetime.now()]
    max_y, min_y = 40, 0

    # Reset environments since internals may be overwritten by tracers from the
    # domain randomization function.
    env = envs.get_environment(
        env_name,
        environment_config=env_config,
        robot_config=robot_config,
        init_scene_path=init_scene_path,
    )
    eval_env = envs.get_environment(
        env_name,
        environment_config=env_config,
        robot_config=robot_config,
        init_scene_path=init_scene_path,
    )
    make_inference_fn, params, _ = train_fn(
        environment=env, progress_fn=progress, eval_env=eval_env
    )

    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")

    # Save and reload params.
    model.save_params(model_save_path, params)
    params = model.load_params(model_save_path)
