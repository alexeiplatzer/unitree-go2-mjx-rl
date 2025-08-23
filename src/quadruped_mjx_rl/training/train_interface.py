
import functools
import logging
import time
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from flax.struct import dataclass as flax_dataclass

from quadruped_mjx_rl import running_statistics, types
from quadruped_mjx_rl.environments import Env, PipelineModel, State
from quadruped_mjx_rl.models.architectures import raw_actor_critic
from quadruped_mjx_rl.models.networks import AgentParams, NetworkFactory
from quadruped_mjx_rl.training import (
    acting, logger as metric_logger, pmap,
    training_utils as _utils,
)
from quadruped_mjx_rl.training.algorithms.ppo import (
    compute_ppo_loss,
)
from quadruped_mjx_rl.training.fitting import Fitter, OptimizerState, SimpleFitter, get_fitter
from quadruped_mjx_rl.training.training import TrainingConfig
from quadruped_mjx_rl.types import PRNGKey
from quadruped_mjx_rl.training.train_main import train as ppo_train
from quadruped_mjx_rl.models.configs import ModelConfig
from quadruped_mjx_rl.models.factories import get_networks_factory


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
            raise ValueError("Madrona-MJX doesn't support multiple env instances")
        if num_eval_envs != num_envs:
            raise ValueError("Madrona-MJX requires a fixed batch size")
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
    # callbacks
    progress_fn: Callable[[int, ...], None] = lambda *args: None,
    policy_params_fn: Callable[..., None] = lambda *args: None,
    # checkpointing
    restore_params_fn: Callable | None = None,
    # restore_params: AgentParams | None = None,
    # restore_value_fn: bool = True,
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
    if max_devices_per_host:
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

    metrics_aggregator = metric_logger.EpisodeMetricsLogger(
        steps_between_logging=training_config.training_metrics_steps or env_step_per_training_step,
        progress_fn=progress_fn,
    )

    def sgd_step(
        carry,
        unused_t,
        data: types.Transition,
        normalizer_params: running_statistics.RunningStatisticsState,
    ):
        opt_state, network_params, key = carry
        key, key_perm, key_grad = jax.random.split(key, 3)

        if training_config.augment_pixels:
            key, key_rt = jax.random.split(key)
            r_translate = functools.partial(_utils.random_translate_pixels, key=key_rt)
            data = types.Transition(
                observation=r_translate(data.observation),
                action=data.action,
                reward=data.reward,
                discount=data.discount,
                next_observation=r_translate(data.next_observation),
                extras=data.extras,
            )

        def convert_data(x: jnp.ndarray):
            x = jax.random.permutation(key_perm, x)
            x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
            return x

        shuffled_data = jax.tree_util.tree_map(convert_data, data)
        (opt_state, network_params, ignore_key), metrics = jax.lax.scan(
            functools.partial(fitter.minibatch_step, normalizer_params=normalizer_params),
            (opt_state, network_params, key_grad),
            shuffled_data,
            length=num_minibatches,
        )
        return (opt_state, network_params, key), metrics

    def training_step(
        carry: tuple[TrainingState, State, PRNGKey], unused_t
    ) -> tuple[tuple[TrainingState, State, PRNGKey], types.Metrics]:
        training_state, state, key = carry
        key_sgd, key_generate_unroll, new_key = jax.random.split(key, 3)

        acting_policy = policy_factories[0](training_state.agent_params)
        # policies = (policies,) if not isinstance(policies, tuple) else policies

        def roll(carry, unused_t):
            current_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            next_state, data = acting.generate_unroll(
                env,
                current_state,
                acting_policy,
                current_key,
                unroll_length,
                extra_fields=("truncation", "episode_metrics", "episode_done"),
            )
            return (next_state, next_key), data

        (state, ignore_key), data = jax.lax.scan(
            roll,
            (state, key_generate_unroll),
            (),
            length=batch_size * num_minibatches // num_envs,
        )
        # Have leading dimensions (batch_size * num_minibatches, unroll_length)
        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
        data = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data)
        assert data.discount.shape[1:] == (unroll_length,)

        if training_config.log_training_metrics:  # log unroll metrics
            jax.debug.callback(
                metrics_aggregator.update_episode_metrics,
                data.extras["state_extras"]["episode_metrics"],
                data.extras["state_extras"]["episode_done"],
            )

        # Update normalization params and normalize observations.
        normalizer_params = running_statistics.update(
            training_state.agent_params.preprocessor_params,
            _utils.remove_pixels(data.observation),
            pmap_axis_name=_utils.PMAP_AXIS_NAME,
        )

        (opt_state, ts_params, ignore_key), metrics = jax.lax.scan(
            functools.partial(sgd_step, data=data, normalizer_params=normalizer_params),
            (
                training_state.optimizer_state,
                training_state.agent_params.network_params,
                key_sgd,
            ),
            (),
            length=num_updates_per_batch,
        )

        new_training_state = TrainingState(
            optimizer_state=opt_state,
            agent_params=AgentParams(
                network_params=ts_params,
                preprocessor_params=normalizer_params,
            ),
            env_steps=training_state.env_steps + env_step_per_training_step,
        )
        return (new_training_state, state, new_key), metrics

    def training_epoch(
        training_state: TrainingState, state: State, key: PRNGKey
    ) -> tuple[TrainingState, State, types.Metrics]:
        (training_state, state, ignored_key), loss_metrics = jax.lax.scan(
            training_step,
            (training_state, state, key),
            (),
            length=num_training_steps_per_epoch,
        )
        loss_metrics = jax.tree_util.tree_map(jnp.mean, loss_metrics)
        return training_state, state, loss_metrics

    training_epoch = jax.pmap(training_epoch, axis_name=_utils.PMAP_AXIS_NAME)

    # Note that this is NOT a pure jittable method.
    def training_epoch_with_timing(
        training_state: TrainingState, env_state: State, key: PRNGKey
    ) -> tuple[TrainingState, State, types.Metrics]:
        nonlocal training_walltime
        t = time.time()
        training_state, env_state = _utils.strip_weak_type((training_state, env_state))
        result = training_epoch(training_state, env_state, key)
        training_state, env_state, metrics = _utils.strip_weak_type(result)

        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

        epoch_training_time = time.time() - t
        training_walltime += epoch_training_time
        sps = (
                  num_training_steps_per_epoch
                  * env_step_per_training_step
                  * max(training_config.num_resets_per_eval, 1)
              ) / epoch_training_time
        metrics = {
            "training/sps": sps,
            "training/walltime": training_walltime,
            **{f"training/{name}": value for name, value in metrics.items()},
        }
        return training_state, env_state, metrics

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
            eval_env or environment,
            wrap_env,
            num_eval_envs,
            training_config.episode_length,
            action_repeat,
            device_count=1,  # eval on the host only
            key_env=eval_key,
            wrap_env_fn=wrap_env_fn,
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
        ) for policy_factory, policy_eval_key in zip(policy_factories, eval_keys)
    ]
    evaluators_metrics = [{} for evaluator in evaluators]

    def run_evaluations(
        params: AgentParams, training_metrics: types.Metrics,
    ):
        for idx in range(len(evaluators)):
            evaluator_metrics = evaluators[idx].run_evaluation(
                _utils.unpmap(params),
                training_metrics=training_metrics
            )
            logging.info(evaluator_metrics)
            progress_fn(0, evaluator_metrics)
            evaluators_metrics[idx] = evaluator_metrics


    # Run initial eval
    if process_id == 0 and num_evals > 1:
        run_evaluations(training_state.agent_params, training_metrics={})

    training_metrics = {}
    training_walltime = 0
    current_step = 0
    for it in range(num_evals_after_init):
        logging.info("starting iteration %s %s", it, time.time() - xt)

        for _ in range(max(training_config.num_resets_per_eval, 1)):
            # optimization
            epoch_key, local_key = jax.random.split(local_key)
            epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
            training_state, env_state, training_metrics = training_epoch_with_timing(
                training_state, env_state, epoch_keys
            )
            current_step = int(_utils.unpmap(training_state.env_steps))

            key_envs = jax.vmap(lambda x, s: jax.random.split(x[0], s), in_axes=(0, None))(
                key_envs, key_envs.shape[1]
            )
            env_state = reset_fn(key_envs) if training_config.num_resets_per_eval > 0 else env_state

        if process_id != 0:
            continue

        # Process id == 0.
        teacher_student_params = _utils.unpmap(training_state.agent_params)

        policy_params_fn(current_step, policy_factories, teacher_student_params)

        if num_evals > 0:
            run_evaluations(teacher_student_params, training_metrics)

    total_steps = current_step
    if not total_steps >= num_timesteps:
        raise AssertionError(
            f"Total steps {total_steps} is less than `num_timesteps`=" f" {num_timesteps}."
        )

    # If there were no mistakes, the training_state should still be identical on all devices.
    pmap.assert_is_replicated(training_state)
    teacher_student_params = _utils.unpmap(training_state.agent_params)
    logging.info("total steps: %s", total_steps)
    pmap.synchronize_hosts()
    return policy_factories, teacher_student_params, evaluators_metrics
