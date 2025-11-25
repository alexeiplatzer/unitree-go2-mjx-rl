import functools
import logging
import time
from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax.struct import dataclass as flax_dataclass

from quadruped_mjx_rl import running_statistics, types
from quadruped_mjx_rl.environments.physics_pipeline import Env, State
from quadruped_mjx_rl.models.networks_utils import AgentParams
from quadruped_mjx_rl.training import (
    acting,
    logger as metric_logger,
    pmap,
    training_utils as _utils,
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


@flax_dataclass
class AgentState:
    recurrent_carry: types.RecurrentHiddenState
    recurrent_buffer: jax.Array


def train(
    training_config: TrainingConfig,
    env: Env,
    fitter: Fitter,
    acting_policy_factory,
    # metrics_aggregator: metric_logger.EpisodeMetricsLogger,
    # callbacks
    policy_params_fn: Callable[..., None],
    run_evaluations,
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
):
    # Unpack hyperparams
    num_envs = training_config.num_envs
    num_evals = training_config.num_evals
    num_timesteps = training_config.num_timesteps
    num_updates_per_batch = training_config.num_updates_per_batch
    num_minibatches = training_config.num_minibatches
    batch_size = training_config.batch_size
    unroll_length = training_config.unroll_length

    bptt_truncation = 64
    latent_sizes = 64

    recurrent_buffer = jnp.zeros((num_minibatches*batch_size, bptt_truncation, latent_sizes))

    xt = time.time()

    def sgd_step(
        carry,
        unused_t,
        data: types.Transition,
        normalizer_params: running_statistics.RunningStatisticsState,
    ):
        opt_state, network_params, key, agent_state = carry
        key, key_perm, key_grad = jax.random.split(key, 3)

        # if training_config.augment_pixels:
        #     key, key_rt = jax.random.split(key)
        #     r_translate = functools.partial(_utils.random_translate_pixels, key=key_rt)
        #     data = types.Transition(
        #         observation=r_translate(data.observation),
        #         action=data.action,
        #         reward=data.reward,
        #         discount=data.discount,
        #         next_observation=r_translate(data.next_observation),
        #         extras=data.extras,
        #     )

        def convert_data(x: jnp.ndarray):
            # x = jax.random.permutation(key_perm, x)
            x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
            return x

        def restore_data(x: jnp.ndarray):
            return jnp.reshape(x, (-1,) + x.shape[2:])

        shuffled_data = jax.tree_util.tree_map(convert_data, data)
        agent_state_batched = jax.tree_util.tree_map(convert_data, agent_state)
        (opt_state, network_params, ignore_key), (agent_state_batched_updated, metrics) = (
            jax.lax.scan(
                functools.partial(fitter.minibatch_step, normalizer_params=normalizer_params),
                (opt_state, network_params, key_grad),
                (shuffled_data, agent_state_batched),
                length=num_minibatches,
            )
        )
        agent_state_updated = jax.tree_util.tree_map(
            restore_data, agent_state_batched_updated
        )
        return (opt_state, network_params, key, agent_state_updated), metrics

    # def recurrent_student_step(opt_state, network_params, key, data, normalizer_params):
    #     key_smth, key_perm = jax.random.split(key, 2)
    #     prev_carry = None
    #     recurrent_buffer = None
    #
    #     render_every = 5
    #     images = data.observation["pixels/frontal_ego/rgb_adjusted"][:, ::render_every]
    #     output, carry = fitter.network.student_encoder_network.apply(
    #         preprocessor_params=normalizer_params,
    #         params=network_params.student_encoder,
    #         key=key_smth,
    #         visual_inputs=images,
    #         proprio_inputs=data.observation,
    #         carry=prev_carry,
    #     )
    #     recurrent_buffer = jnp.roll(recurrent_buffer, 1, axis=-2)
    #     # now do a gradient update of the lstm

    def training_step(
        carry: tuple[TrainingState, State, PRNGKey, AgentState], _
    ) -> tuple[tuple[TrainingState, State, PRNGKey, AgentState], types.Metrics]:
        training_state, state, key, agent_state = carry
        key_sgd, key_generate_unroll, new_key = jax.random.split(key, 3)

        acting_policy = acting_policy_factory(training_state.agent_params, deterministic=False)
        recurrent_encoder = functools.partial(
            fitter.network.student_encoder_network.apply,
            preprocessor_params=training_state.agent_params.preprocessor_params,
            params=training_state.agent_params.network_params.student_encoder,
        )

        def roll(carry, _):
            current_state, current_key, recurrent_state = carry
            current_key, next_key = jax.random.split(current_key)
            next_state, data = acting.generate_unroll(
                env,
                current_state,
                acting_policy,
                current_key,
                unroll_length,
                extra_fields=("truncation", "episode_metrics", "episode_done"),
                add_vision_obs=True,
                proprioceptive_steps_per_vision_step=5,  # TODO configure
            )
            return (next_state, next_key), data

        (state, ignore_key, final_recurrent_state), data = jax.lax.scan(
            roll,
            (state, key_generate_unroll),
            (),
            length=batch_size * num_minibatches // num_envs,
        )
        # Have leading dimensions (batch_size * num_minibatches, unroll_length)
        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
        data = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data)
        assert data.discount.shape[1:] == (unroll_length,)

        # Update normalization params and normalize observations.
        normalizer_params = running_statistics.update(
            training_state.agent_params.preprocessor_params,
            _utils.remove_pixels(data.observation),
            pmap_axis_name=_utils.PMAP_AXIS_NAME,
        )

        (opt_state, network_params, ignore_key, new_agent_state), metrics = jax.lax.scan(
            functools.partial(sgd_step, data=data, normalizer_params=normalizer_params),
            (
                training_state.optimizer_state,
                training_state.agent_params.network_params,
                key_sgd,
                agent_state,
            ),
            (),
            length=num_updates_per_batch,
        )

        new_training_state = TrainingState(
            optimizer_state=opt_state,
            agent_params=type(training_state.agent_params)(
                network_params=network_params,
                preprocessor_params=normalizer_params,
            ),
            env_steps=training_state.env_steps + env_step_per_training_step,
        )
        return (new_training_state, state, new_key, new_agent_state), metrics

    # TODO update carries with the agent state
    def training_epoch(
        training_state: TrainingState, state: State, key: PRNGKey
    ) -> tuple[TrainingState, State, types.Metrics]:
        (training_state, state, ignored_key), loss_metrics = (
            jax.lax.scan(
                training_step,
                (training_state, state, key),
                (),
                length=num_training_steps_per_epoch,
            )
        )
        loss_metrics = jax.tree_util.tree_map(jnp.mean, loss_metrics)
        return training_state, state, loss_metrics

    training_epoch = jax.pmap(training_epoch, axis_name=_utils.PMAP_AXIS_NAME)

    # Note that this is NOT a pure jittable method.
    def training_epoch_with_timing(
        training_state: TrainingState,
        env_state: State,
        key: PRNGKey,
    ) -> tuple[TrainingState, State, types.Metrics]:
        nonlocal training_walltime
        t = time.time()
        training_state, env_state = _utils.strip_weak_type((training_state, env_state))
        result = training_epoch(training_state, env_state, key)
        training_state, env_state, metrics = _utils.strip_weak_type(
            result
        )

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

    training_metrics = {}
    training_walltime = 0
    current_step = 0

    # Run initial policy params fn
    params = _utils.unpmap(training_state.agent_params)
    policy_params_fn(current_step, params)
    # Run initial eval
    if process_id == 0 and num_evals > 1:
        run_evaluations(current_step, params, training_metrics={})

    for it in range(num_evals_after_init):
        logging.info("starting iteration %s %s", it, time.time() - xt)

        for _ in range(max(training_config.num_resets_per_eval, 1)):
            # optimization
            epoch_key, local_key = jax.random.split(local_key)
            epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
            (training_state, env_state, training_metrics) = (
                training_epoch_with_timing(
                    training_state, env_state, epoch_keys
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
