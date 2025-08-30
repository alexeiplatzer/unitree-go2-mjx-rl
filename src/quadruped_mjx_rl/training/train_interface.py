import functools
import logging
import time
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from quadruped_mjx_rl import running_statistics, types
from quadruped_mjx_rl.environments.physics_pipeline import Env, PipelineModel
from quadruped_mjx_rl.models.configs import ModelConfig
from quadruped_mjx_rl.models.factories import get_networks_factory
from quadruped_mjx_rl.models.networks import AgentParams
from quadruped_mjx_rl.training import (
    acting,
    logger as metric_logger,
    training_utils as _utils,
)
from quadruped_mjx_rl.training.algorithms.ppo import (
    compute_ppo_loss,
)
from quadruped_mjx_rl.training.configs import TrainingConfig
from quadruped_mjx_rl.training.evaluation import make_progress_fn
from quadruped_mjx_rl.training.fitting import get_fitter
from quadruped_mjx_rl.training.train_backend import train as backed_train, TrainingState


def validate_args_for_vision(
    vision: bool,
    num_envs: int,
    num_eval_envs: int,
    action_repeat: int,
    eval_env: Env | None = None,
):
    """Validates arguments for Madrona-MJX."""
    if vision:
        if eval_env:
            raise ValueError(
                "Evaluation env is not None. The Madrona-MJX vision backend doesn't support "
                "multiple env instances, one env must be used for both training and evaluation."
            )
        if num_eval_envs != num_envs:
            raise ValueError(
                "Number of eval envs != number of training envs. The Madrona-MJX vision backend"
                " requires a fixed batch size, the number of environments must be consistent."
            )
        if action_repeat != 1:
            raise ValueError(
                "Implement action_repeat using PipelineEnv's _n_frames to avoid"
                " unnecessary rendering!"
            )


def train(
    training_config: TrainingConfig,
    model_config: ModelConfig,
    training_env: Env,
    evaluation_env: Env | None = None,
    max_devices_per_host: int | None = None,
    # environment wrapper
    wrap_env: bool = True,
    randomization_fn: (
        Callable[[PipelineModel, jnp.ndarray], tuple[PipelineModel, PipelineModel]] | None
    ) = None,
    # checkpointing
    policy_params_fn: Callable[..., None] = lambda *args: None,
    restore_params_fn: Callable | None = None,
    run_in_cell: bool = True,
) -> tuple[tuple, AgentParams, list[dict]]:
    # Unpack hyperparams
    num_envs = training_config.num_envs
    num_eval_envs = training_config.num_eval_envs
    num_evals = training_config.num_evals
    num_timesteps = training_config.num_timesteps
    num_updates_per_batch = training_config.num_updates_per_batch
    num_minibatches = training_config.num_minibatches
    batch_size = training_config.batch_size
    unroll_length = training_config.unroll_length
    action_repeat = training_config.action_repeat

    # Check arguments
    if batch_size * num_minibatches % num_envs != 0:
        raise ValueError(
            f"Batch size ({batch_size}) times number of minibatches ({num_minibatches}) "
            f"must be divisible by number of environments ({num_envs})."
        )
    validate_args_for_vision(
        training_config.use_vision, num_envs, num_eval_envs, action_repeat, evaluation_env
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
    # Equals to ceil(
    #   num_timesteps
    #   / (
    #       num_evals
    #       * env_step_per_training_step
    #       * num_resets_per_eval
    #   )
    # )
    num_training_steps_per_epoch = np.ceil(
        num_timesteps
        / (
            num_evals_after_init
            * env_step_per_training_step
            * max(training_config.num_resets_per_eval, 1)
        )
    ).astype(int)

    key = jax.random.PRNGKey(training_config.seed)
    global_key, local_key = jax.random.split(key)
    del key
    local_key = jax.random.fold_in(local_key, process_id)
    local_key, key_env, eval_key = jax.random.split(local_key, 3)
    # key_networks should be global, so that networks are initialized the same
    # way for different processes.

    if num_envs % device_count != 0:
        raise ValueError(
            f"Number of environments ({num_envs}) must be divisible by device count "
            f"({device_count})."
        )

    env = _utils.maybe_wrap_env(
        training_env,
        wrap_env,
        num_envs,
        training_config.episode_length,
        action_repeat,
        device_count,
        key_env,
        None,
        randomization_fn,
        vision=training_config.use_vision,
    )
    reset_fn = jax.jit(jax.vmap(env.reset))
    key_envs = jax.random.split(key_env, num_envs // process_count)
    key_envs = jnp.reshape(key_envs, (local_devices_to_use, -1) + key_envs.shape[1:])
    env_state = reset_fn(key_envs)

    # Shapes of different observation tensors
    # Discard the batch axes over devices and envs.
    obs_shape = jax.tree_util.tree_map(lambda x: x.shape[2:], env_state.obs)

    preprocess_fn = (
        running_statistics.normalize
        if training_config.normalize_observations
        else lambda x, y: x
    )

    network_factory = get_networks_factory(model_config)

    ppo_networks = network_factory(
        observation_size=obs_shape,
        action_size=env.action_size,
        preprocess_observations_fn=preprocess_fn,
    )
    policy_factories = ppo_networks.policy_metafactory()

    make_fitter = get_fitter(model_config)
    fitter = make_fitter(
        optimizer_config=training_config.optimizer,
        network=ppo_networks,
        main_loss_fn=compute_ppo_loss,
        algorithm_hyperparams=training_config.rl_hyperparams,
    )

    progress_fn, eval_times = make_progress_fn(
        num_timesteps=training_config.num_timesteps, run_in_cell=run_in_cell
    )

    # Ensure positive logging cadence; fallback to env_step_per_training_step if unset or invalid.
    steps_between_logging = (
        training_config.training_metrics_steps
        if training_config.training_metrics_steps and training_config.training_metrics_steps > 0
        else env_step_per_training_step
    )
    metrics_aggregator = metric_logger.EpisodeMetricsLogger(
        steps_between_logging=steps_between_logging,
        progress_fn=progress_fn,
    )

    # Initialize model params and training state.
    network_init_params = ppo_networks.initialize(global_key)

    obs_shape = jax.tree_util.tree_map(
        lambda x: running_statistics.Array(x.shape[-1:], jnp.dtype("float32")), env_state.obs
    )
    training_state = TrainingState(
        optimizer_state=fitter.optimizer_init(network_init_params),
        agent_params=AgentParams(
            network_params=network_init_params,
            preprocessor_params=running_statistics.init_state(_utils.remove_pixels(obs_shape)),
        ),
        env_steps=jnp.array(0, dtype=jnp.int32),
    )

    if restore_params_fn is not None:
        logging.info("Restoring TrainingState from `restore_params`.")
        training_state = training_state.replace(
            agent_params=restore_params_fn(training_state.agent_params)
        )

    if num_timesteps == 0:
        return (
            policy_factories,
            training_state.agent_params,
            [{}],
        )

    training_state = jax.device_put_replicated(
        training_state, jax.local_devices()[:local_devices_to_use]
    )

    eval_env = (
        env
        if training_config.use_vision
        else _utils.maybe_wrap_env(
            evaluation_env or training_env,
            wrap_env,
            num_eval_envs,
            training_config.episode_length,
            action_repeat,
            device_count=1,  # eval on the host only
            key_env=eval_key,
            wrap_env_fn=None,
            randomization_fn=randomization_fn,
            vision=training_config.use_vision,
        )
    )
    eval_keys = jax.random.split(eval_key, len(policy_factories))
    evaluators = [
        acting.Evaluator(
            eval_env,
            functools.partial(policy_factory, deterministic=training_config.deterministic_eval),
            num_eval_envs=num_eval_envs,
            episode_length=training_config.episode_length,
            action_repeat=action_repeat,
            key=policy_eval_key,
        )
        for policy_factory, policy_eval_key in zip(policy_factories, eval_keys)
    ]
    evaluators_metrics = [{} for evaluator in evaluators]

    def run_evaluations(
        current_step,
        params,
        training_metrics: types.Metrics,
    ):
        for idx in range(len(evaluators)):
            evaluator_metrics = evaluators[idx].run_evaluation(
                params, training_metrics=training_metrics
            )
            logging.info("eval/episode_reward: %s" % evaluator_metrics["eval/episode_reward"])
            logging.info(
                "eval/episode_reward_std: %s" % evaluator_metrics["eval/episode_reward_std"]
            )
            logging.info("current_step: %s" % current_step)
            progress_fn(current_step, evaluator_metrics)
            evaluators_metrics[idx] = evaluator_metrics

    logging.info("Setup took %s", time.time() - xt)

    final_params = backed_train(
        training_config=training_config,
        env=env,
        fitter=fitter,
        policy_factories=policy_factories,
        metrics_aggregator=metrics_aggregator,
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
    )

    logging.info("Time to jit: %s", eval_times[1] - eval_times[0])
    logging.info("Time to train: %s", eval_times[-1] - eval_times[1])

    return policy_factories, final_params, evaluators_metrics
