import functools
import logging
import time
from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from quadruped_mjx_rl import running_statistics
from quadruped_mjx_rl.environments.vision import VisionWrapper
from quadruped_mjx_rl.environments.wrappers import wrap_for_training
from quadruped_mjx_rl.physics_pipeline import Env
from quadruped_mjx_rl.domain_randomization import (
    DomainRandomizationConfig,
    TerrainMapRandomizationConfig,
)
from quadruped_mjx_rl.models import get_networks_factory
from quadruped_mjx_rl.training import logger as metric_logger, training_utils as _utils
from quadruped_mjx_rl.models.architectures import (
    ActorCriticConfig,
    TeacherStudentRecurrentNetworks,
)
from quadruped_mjx_rl.models.types import AgentParams
from quadruped_mjx_rl.training.algorithms.ppo import compute_ppo_loss
from quadruped_mjx_rl.training.configs import (
    TrainingConfig,
)
from quadruped_mjx_rl.training.fitting import get_fitter
from quadruped_mjx_rl.training.train_backend import train as train_backend, TrainingState


def train(
    training_config: TrainingConfig,
    model_config: ActorCriticConfig,
    training_env: Env,
    evaluation_env: Env | None = None,
    max_devices_per_host: int | None = None,
    # environment wrapper
    wrap_env: bool = True,
    randomization_config: (
        DomainRandomizationConfig | TerrainMapRandomizationConfig | None
    ) = None,
    # checkpointing
    policy_params_fn: Callable[..., None] = lambda *args: None,
    restore_params: AgentParams | None = None,
    # progress plotting
    show_outputs: bool = True,
    run_in_cell: bool = True,
    save_plots_path: Path | None = None,
) -> AgentParams:
    """

    Args:
        training_config: Hyperparameters for the training loop and RL algorithm
        model_config: Hyperparameters for the neural network architecture
        training_env: Environment used for training
        evaluation_env: Environment used for evaluation. If None, the training env is used.
        max_devices_per_host: Limit of the number of devices to be used per process.
        wrap_env: Whether to wrap the environment for training and evaluation, or use it as is.
        randomization_config: Domain randomization config, randomizes each parallel environment.
        policy_params_fn: Callback function called with the intermediate agent parameters after
            each training epoch.
        restore_params: Pre-trained agent parameters which will be used as initial.
        show_outputs: Whether to show progress plots dynamically
        run_in_cell: Whether an IPython context is present (like a Jupyter notebook cell). In
            this case, the plots are updated dynamically, replacing the older ones each step.
        save_plots_path: A path to a directory where the plot images should be saved.

    Returns:
        The final trained parameters for the agent.

    """
    # Unpack hyperparams
    num_envs = training_config.num_envs
    num_eval_envs = training_config.num_eval_envs
    num_evals = training_config.num_evals
    num_timesteps = training_config.num_timesteps
    # num_updates_per_batch = training_config.num_updates_per_batch
    num_minibatches = training_config.num_minibatches
    batch_size = training_config.batch_size
    unroll_length = training_config.unroll_length
    action_repeat = training_config.action_repeat

    vision = model_config.vision

    # Check arguments
    training_config.check_validity()
    if vision and evaluation_env is not None:
        raise ValueError(
            "Evaluation env is not None. The Madrona-MJX vision backend doesn't support "
            "multiple env instances, one env must be used for both training and evaluation."
        )

    xt = time.time()

    # Gather devices and processes info
    process_count = jax.process_count()
    process_id = jax.process_index()
    local_device_count = jax.local_device_count()
    local_devices_to_use = local_device_count
    if max_devices_per_host is not None:
        if max_devices_per_host <= 0:
            raise ValueError("max_devices_per_host must be > 0 when provided.")
        local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
    logging.info(
        "Device count: %d, process count: %d (id %d), local device count: %d, "
        "devices to be used count: %d",
        jax.device_count(),
        process_count,
        process_id,
        local_device_count,
        local_devices_to_use,
    )
    device_count = local_devices_to_use * process_count

    # The number of environment steps executed for every training step.
    env_step_per_training_step = batch_size * unroll_length * num_minibatches * action_repeat
    num_evals_after_init = max(num_evals - 1, 1)

    # The number of training_step calls per training_epoch call.
    num_training_steps_per_epoch = np.ceil(
        num_timesteps
        / (
            num_evals_after_init
            * env_step_per_training_step
            * max(training_config.num_resets_per_eval, 1)
        )
    ).astype(int)
    logging.info(
        f"Important numbers:\n"
        f"Num evals after init: {num_evals_after_init}\n"
        f"Num resets (epochs) per eval: {max(training_config.num_resets_per_eval, 1)}\n"
        f"Num training steps per epoch: {num_training_steps_per_epoch}\n"
        f"Env steps per training step: {env_step_per_training_step}"
    )

    key = jax.random.PRNGKey(training_config.seed)
    # key_networks should be global, so that networks are initialized the same
    # way for different processes.
    global_key, local_key = jax.random.split(key)
    del key
    local_key = jax.random.fold_in(local_key, process_id)
    local_key, batch_key, wrapping_key, eval_key = jax.random.split(local_key, 4)

    if num_envs % device_count != 0:
        raise ValueError(
            f"Number of environments ({num_envs}) must be divisible by device count "
            f"({device_count})."
        )

    logging.info("Wrapping the training environment...")
    env = (
        wrap_for_training(
            env=training_env,
            num_envs=num_envs,
            device_count=device_count,
            episode_length=training_config.episode_length,
            action_repeat=action_repeat,
            randomization_config=randomization_config,
            rng_key=wrapping_key,
            vision=vision,
        )
        if wrap_env
        else training_env
    )

    batch_key = jax.random.split(batch_key, num_envs // process_count)
    batch_key = jnp.reshape(batch_key, (local_devices_to_use, -1) + batch_key.shape[1:])
    keyandrs = jax.vmap(jax.vmap(jax.random.split))(batch_key)
    key_envs, key_agent_states = keyandrs[:, :, 0], keyandrs[:, :, 1]

    logging.info("Instantiating environment states...")
    reset_fn = jax.jit(jax.vmap(env.reset))
    # TODO deal with the reusage of these key envs in backend
    env_state = reset_fn(key_envs)

    # Shapes of different observation tensors
    # Discard the batch axes over devices and envs.
    obs = env_state.obs
    if vision:
        vision_obs = jax.jit(jax.vmap(env.get_vision_obs))(
            env_state.pipeline_state, env_state.info
        )
        obs = obs | vision_obs
    obs_shape = jax.tree_util.tree_map(lambda x: x.shape[2:], obs)

    preprocess_fn = (
        running_statistics.normalize
        if training_config.normalize_observations
        else lambda x, y: x
    )

    network_factory = get_networks_factory(model_config)

    ppo_networks = network_factory(
        observation_size=obs_shape,
        action_size=env.action_size,
        vision_obs_period=training_config.vision_config.vision_obs_period,
        preprocess_observations_fn=preprocess_fn,
    )

    make_fitter = get_fitter(model_config)
    fitter = make_fitter(
        optimizer_config=training_config.optimizer,
        network=ppo_networks,
        main_loss_fn=compute_ppo_loss,
        algorithm_hyperparams=training_config.rl_hyperparams,
    )

    agent_state = None
    recurrent = model_config.recurrent

    if recurrent:
        logging.info("Initializing recurrent agent state...")
        assert isinstance(ppo_networks, TeacherStudentRecurrentNetworks)
        ppo_networks.set_recurrent_period(unroll_length)
        agent_state = jax.vmap(jax.vmap(ppo_networks.init_agent_state))(key_agent_states)

    logging.info("Initializing model params and training state...")
    network_init_params = ppo_networks.initialize(global_key)

    obs_shape = jax.tree_util.tree_map(
        lambda x: running_statistics.Array(x.shape[-1:], jnp.dtype("float32")), env_state.obs
    )
    training_state = TrainingState(
        optimizer_state=fitter.optimizer_init(network_init_params),
        agent_params=ppo_networks.agent_params_class()(
            network_params=network_init_params,
            preprocessor_params=running_statistics.init_state(_utils.remove_pixels(obs_shape)),
        ),
        env_steps=jnp.array(0, dtype=jnp.int32),
    )

    if restore_params is not None:
        logging.info("Restoring TrainingState from `restore_params`.")
        training_state = training_state.replace(
            agent_params=training_state.agent_params.restore_params(
                restore_params, restore_value=True
            )
        )
    logging.info(f"Agent params type: {type(training_state.agent_params)}")

    if num_timesteps == 0:
        return training_state.agent_params

    logging.info("Distributing training state to devices...")
    training_state = jax.device_put_replicated(
        training_state, jax.local_devices()[:local_devices_to_use]
    )

    logging.info("Wrapping the evaluation environment...")
    eval_env = (
        env
        if vision or not wrap_env
        else wrap_for_training(
            env=evaluation_env or training_env,
            num_envs=num_eval_envs,
            device_count=1,  # eval on the host only
            episode_length=training_config.episode_length,
            action_repeat=action_repeat,
            randomization_config=randomization_config,
            rng_key=eval_key,
            vision=vision,
        )
    )

    logging.info("Setting up evaluation functions...")
    run_evaluations, eval_times = fitter.make_evaluation_fn(
        eval_env=eval_env,
        eval_key=eval_key,
        training_config=training_config,
        show_outputs=show_outputs,
        run_in_cell=run_in_cell,
        save_plots_path=save_plots_path,
    )

    metrics_aggregator = metric_logger.EpisodeMetricsLogger(
        steps_between_logging=training_config.training_metrics_steps
        or env_step_per_training_step,
        progress_fn=None,  # TODO: think how to pass it
    )

    logging.info("Setup took %s", time.time() - xt)

    final_params = train_backend(
        training_config=training_config,
        env=env,
        fitter=fitter,
        policy_params_fn=policy_params_fn,
        run_evaluations=run_evaluations,
        env_step_per_training_step=env_step_per_training_step,
        num_training_steps_per_epoch=num_training_steps_per_epoch,
        local_devices_to_use=local_devices_to_use,
        num_evals_after_init=num_evals_after_init,
        process_id=process_id,
        training_state=training_state,
        reset_fn=reset_fn,
        local_key=local_key,
        key_envs=key_envs,
        env_state=env_state,
        metrics_aggregator=metrics_aggregator,
        recurrent_agent_state=agent_state,
        recurrent=recurrent,
        generate_unroll_factory=functools.partial(
            ppo_networks.make_acting_unroll_fn, deterministic=False
        ),
    )

    logging.info("Time to jit: %s", eval_times[1] - eval_times[0])
    logging.info("Time to train: %s", eval_times[-1] - eval_times[1])

    return final_params
