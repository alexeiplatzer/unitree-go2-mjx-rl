# Import all necessary packages

# Supporting
import functools

# Math
import jax
import jax.numpy as jp
import mediapy as media

# Brax
from brax import envs
from brax.io import model
from brax.training.acme import running_statistics

# Algorithms
from brax.training.agents.ppo import networks as ppo_networks

from .training_environments.ppo_simple import JoystickEnv
from .utils import load_config_dicts


def render(
    trained_model_path,
    render_config_path,
    rl_config_path,
    robot_config_path,
    init_scene_path,
    save_animation=False,
    animation_save_path=None,
):
    rl_config, robot_config, render_config = load_config_dicts(
        rl_config_path,
        robot_config_path,
        render_config_path,
    )

    env_name = "joystick_go2"
    envs.register_environment(env_name, JoystickEnv)
    env_config = rl_config.environment
    env = envs.get_environment(
        env_name,
        environment_config=env_config,
        robot_config=robot_config,
        init_scene_path=init_scene_path,
    )

    make_networks_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=tuple(rl_config.model.hidden_sizes),
    )

    nets = make_networks_factory(
        # Observation_size argument doesn't matter since it's only used for param init.
        observation_size=1,
        action_size=12,
        preprocess_observations_fn=running_statistics.normalize,
    )

    make_inference_fn = ppo_networks.make_inference_fn(nets)

    params = model.load_params(trained_model_path)

    # Inference function
    ppo_inference_fn = make_inference_fn(params)

    # Commands
    x_vel = render_config.command.x_vel
    y_vel = render_config.command.y_vel
    ang_vel = render_config.command.ang_vel

    movement_command = jp.array([x_vel, y_vel, ang_vel])

    demo_env = envs.training.EpisodeWrapper(
        env,
        episode_length=render_config.episode_length,
        action_repeat=1,
    )

    render_rollout(
        jax.jit(demo_env.reset),
        jax.jit(demo_env.step),
        jax.jit(ppo_inference_fn),
        demo_env,
        render_config=render_config,
        save_animation=save_animation,
        save_path=animation_save_path,
        the_command=movement_command,
        camera="track"
    )


def render_rollout(
    reset_fn,
    step_fn,
    inference_fn,
    env,
    render_config,
    save_animation=False,
    save_path=None,
    camera=None,
    the_command=None,
):
    rng = jax.random.key(render_config.seed)
    render_every = render_config.render_every
    state = reset_fn(rng)
    if the_command is not None:
      state.info['command'] = the_command
    rollout = [state.pipeline_state]

    for i in range(render_config.n_steps):
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = inference_fn(state.obs, act_rng)
        state = step_fn(state, ctrl)
        if i % render_every == 0:
            rollout.append(state.pipeline_state)

    rendering = env.render(rollout, camera=camera)
    if save_animation:
        media.write_video(
            save_path,
            env.render(rollout, camera=camera),
            fps=1.0 / (env.dt * render_every),
            codec='gif',
        )
    else:
        media.show_video(
            env.render(rollout, camera=camera),
            fps=1.0 / (env.dt * render_every),
            codec='gif',
        )

