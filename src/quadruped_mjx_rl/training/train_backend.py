import functools
import logging
import time
from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax.struct import dataclass as flax_dataclass

from quadruped_mjx_rl import running_statistics, types
from quadruped_mjx_rl.physics_pipeline import Env, State
from quadruped_mjx_rl.models import AgentParams
from quadruped_mjx_rl.models.acting import GenerateUnrollFn
from quadruped_mjx_rl.models.types import RecurrentAgentState, AgentNetworkParams
from quadruped_mjx_rl.training import (
    pmap,
    training_utils as _utils,
    logger as metric_logger,
)
from quadruped_mjx_rl.training.configs import TrainingConfig
from quadruped_mjx_rl.training.fitting import Fitter, OptimizerState
from quadruped_mjx_rl.types import PRNGKey


@flax_dataclass
class TrainingState:
    """Contains the training state for the learner."""

    optimizer_state: OptimizerState
    agent_params: AgentParams
    env_steps: jnp.ndarray


def train(
    *,
    training_config: TrainingConfig,
    env: Env,
    fitter: Fitter,
    # acting_policy_factory,
    metrics_aggregator: metric_logger.EpisodeMetricsLogger,
    # callbacks
    policy_params_fn: Callable[..., None],
    run_evaluations,
    generate_unroll_factory: Callable[[AgentParams], GenerateUnrollFn],
    # numbers
    env_step_per_training_step: int,
    num_training_steps_per_epoch: int,
    local_devices_to_use: int,
    num_evals_after_init: int,
    process_id: int,
    training_state: TrainingState,
    reset_fn: Callable[[PRNGKey], State],
    local_key: PRNGKey,
    key_envs: PRNGKey,
    env_state: State,
    recurrent_agent_state: RecurrentAgentState | None,
    recurrent: bool,
):
    # Unpack hyperparams
    log_training_metrics = training_config.log_training_metrics
    num_envs = training_config.num_envs
    num_evals = training_config.num_evals
    num_timesteps = training_config.num_timesteps
    num_updates_per_batch = training_config.num_updates_per_batch
    num_minibatches = training_config.num_minibatches
    batch_size = training_config.batch_size
    unroll_length = training_config.unroll_length

    xt = time.time()

    def sgd_step(
        carry: tuple[OptimizerState, AgentNetworkParams, PRNGKey],
        _,
        data: types.Transition,
        agent_state: RecurrentAgentState,
        normalizer_params: running_statistics.RunningStatisticsState,
    ) -> tuple[
        tuple[OptimizerState, AgentNetworkParams, PRNGKey],
        tuple[RecurrentAgentState, types.Metrics],
    ]:
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
            if not recurrent:
                x = jax.random.permutation(key_perm, x)
            x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
            return x

        def restore_data(x: jnp.ndarray):
            return jnp.reshape(x, (-1,) + x.shape[2:])

        converted_data = jax.tree_util.tree_map(convert_data, data)
        agent_state_batched = jax.tree_util.tree_map(convert_data, agent_state)
        if recurrent:
            minibatch_data = (converted_data, agent_state_batched)
        else:
            minibatch_data = converted_data
        (opt_state, network_params, _), minibatch_acc = jax.lax.scan(
            functools.partial(fitter.minibatch_step, normalizer_params=normalizer_params),
            (opt_state, network_params, key_grad),
            minibatch_data,
            length=num_minibatches,
        )
        if recurrent:
            agent_state_batched, metrics = minibatch_acc
        else:
            metrics = minibatch_acc
        agent_state_updated = jax.tree_util.tree_map(restore_data, agent_state_batched)
        return (opt_state, network_params, key),  (agent_state_updated, metrics)

    def training_step(
        carry: tuple[TrainingState, State, PRNGKey, RecurrentAgentState], _
    ) -> tuple[tuple[TrainingState, State, PRNGKey, RecurrentAgentState], types.Metrics]:
        training_state, state, key, agent_state = carry
        key_sgd, key_generate_unroll, new_key = jax.random.split(key, 3)

        unroll_repeat = batch_size * num_minibatches // num_envs
        generate_unroll = generate_unroll_factory(training_state.agent_params)
        state, data = generate_unroll(
            env_state=state,
            key=key_generate_unroll,
            env=env,
            unroll_length=unroll_length * unroll_repeat,
            extra_fields=("truncation", "episode_metrics", "episode_done"),
        )
        # Have leading dimensions (batch_size * num_minibatches, unroll_length)
        def convert_data(x: jax.Array):
            x = jnp.reshape(x, (unroll_repeat, unroll_length) + x.shape[1:])
            x = jnp.swapaxes(x, 1, 2)
            return jnp.reshape(x, (-1,) + x.shape[2:])


        data = jax.tree_util.tree_map(lambda x: convert_data(x), data)
        assert data.discount.shape[-1] % unroll_length == 0  # without the substeps

        # Update normalization params and normalize observations.
        normalizer_params = running_statistics.update(
            training_state.agent_params.preprocessor_params,
            _utils.remove_pixels(data.observation),
            pmap_axis_name=_utils.PMAP_AXIS_NAME,
        )

        (opt_state, network_params, _), (new_agent_states, metrics) = jax.lax.scan(
            functools.partial(
                sgd_step,
                data=data,
                agent_state=agent_state,
                normalizer_params=normalizer_params,
            ),
            (
                training_state.optimizer_state,
                training_state.agent_params.network_params,
                key_sgd,
            ),
            (),
            length=num_updates_per_batch,
        )
        new_agent_state = jax.tree_util.tree_map(lambda x: x[-1], new_agent_states)

        new_training_state = TrainingState(
            optimizer_state=opt_state,
            agent_params=type(training_state.agent_params)(
                network_params=network_params,
                preprocessor_params=normalizer_params,
            ),
            env_steps=training_state.env_steps + env_step_per_training_step,
        )

        if log_training_metrics:  # log unroll metrics
            jax.debug.callback(
                metrics_aggregator.update_episode_metrics,
                data.extras["state_extras"]["episode_metrics"],
                data.extras["state_extras"]["episode_done"],
            )

        return (new_training_state, state, new_key, new_agent_state), metrics

    def training_epoch(
        training_state: TrainingState,
        state: State,
        key: PRNGKey,
        agent_state: RecurrentAgentState,
    ) -> tuple[TrainingState, State, types.Metrics, RecurrentAgentState]:
        (training_state, state, ignored_key, agent_state), loss_metrics = jax.lax.scan(
            training_step,
            (training_state, state, key, agent_state),
            (),
            length=num_training_steps_per_epoch,
        )
        loss_metrics = jax.tree_util.tree_map(jnp.mean, loss_metrics)
        return training_state, state, loss_metrics, agent_state

    training_epoch = jax.pmap(training_epoch, axis_name=_utils.PMAP_AXIS_NAME)

    # Note that this is NOT a pure jittable method.
    def training_epoch_with_timing(
        training_state: TrainingState,
        env_state: State,
        key: PRNGKey,
        agent_state: RecurrentAgentState,
    ) -> tuple[TrainingState, State, types.Metrics, RecurrentAgentState]:
        nonlocal training_walltime
        t = time.time()
        training_state, env_state, agent_state = _utils.strip_weak_type(
            (training_state, env_state, agent_state)
        )
        result = training_epoch(training_state, env_state, key, agent_state)
        training_state, env_state, metrics, agent_state = _utils.strip_weak_type(result)

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
        return training_state, env_state, metrics, agent_state

    training_metrics = {}
    training_walltime = 0
    current_step = 0

    # Run initial policy params fn
    params = _utils.unpmap(training_state.agent_params)
    policy_params_fn(current_step, params)
    # Run initial eval
    if process_id == 0 and num_evals > 1:
        logging.info("Running initial eval...")
        run_evaluations(current_step, params, training_metrics={})

    for it in range(num_evals_after_init):
        logging.info("starting iteration %s %s", it, time.time() - xt)

        for _ in range(max(training_config.num_resets_per_eval, 1)):
            # optimization
            epoch_key, local_key = jax.random.split(local_key)
            epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
            training_state, env_state, training_metrics, recurrent_agent_state = (
                training_epoch_with_timing(
                    training_state,
                    env_state,
                    epoch_keys,
                    recurrent_agent_state,
                )
            )
            current_step = int(_utils.unpmap(training_state.env_steps))

            key_envs = jax.vmap(lambda x, s: jax.random.split(x[0], s), in_axes=(0, None))(
                key_envs, key_envs.shape[1]
            )
            env_state = (
                reset_fn(key_envs) if training_config.num_resets_per_eval > 0 else env_state
            )

        if process_id != 0:
            continue

        # Process id == 0.
        params = _utils.unpmap(training_state.agent_params)

        policy_params_fn(current_step, params)

        if num_evals > 0:
            logging.info(f"training_metrics: {training_metrics}")
            # logging.info(f"nan in params: {jnp.any(jnp.isnan(params))}")
            run_evaluations(current_step, params, training_metrics)

    total_steps = current_step
    if not total_steps >= num_timesteps:
        raise AssertionError(
            f"Total steps {total_steps} is less than `num_timesteps`=" f" {num_timesteps}."
        )

    # If there were no mistakes, the training_state should still be identical on all devices.
    pmap.assert_is_replicated(training_state)
    final_params = _utils.unpmap(training_state.agent_params)
    logging.info("total steps: %s", total_steps)
    pmap.synchronize_hosts()

    return final_params
