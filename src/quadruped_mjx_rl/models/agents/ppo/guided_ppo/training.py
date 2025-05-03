import functools
import time
from collections.abc import Callable, Mapping

from absl import logging
from brax import base
from brax import envs
from brax.training import acting
from brax.training import gradients
from brax.training import logger as metric_logger
from brax.training import pmap
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.agents.ppo import checkpoint
from brax.training.types import Params
from brax.training.types import PRNGKey
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

from quadruped_mjx_rl.models.architectures import guided_actor_critic as ppo_networks
from quadruped_mjx_rl.models.architectures.guided_actor_critic import TeacherNetworkParams
from quadruped_mjx_rl.models.architectures.guided_actor_critic import StudentNetworkParams
from quadruped_mjx_rl.models.agents.ppo.guided_ppo.losses import compute_teacher_loss
from quadruped_mjx_rl.models.agents.ppo.guided_ppo.losses import compute_student_loss
from quadruped_mjx_rl.models.agents.ppo import _training_utils as utils

InferenceParams = tuple[running_statistics.NestedMeanStd, Params]


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""

    teacher_optimizer_state: optax.OptState
    teacher_params: TeacherNetworkParams
    student_optimizer_state: optax.OptState
    student_params: StudentNetworkParams
    normalizer_params: running_statistics.RunningStatisticsState
    env_steps: jnp.ndarray


def train(
    environment: envs.Env,
    num_timesteps: int,
    max_devices_per_host: int | None = None,
    # high-level control flow
    wrap_env: bool = True,
    madrona_backend: bool = False,
    augment_pixels: bool = False,
    # environment wrapper
    num_envs: int = 1,
    episode_length: int | None = None,
    action_repeat: int = 1,
    wrap_env_fn: Callable | None = None,
    randomization_fn: (
        Callable[[base.System, jnp.ndarray], tuple[base.System, base.System]] | None
    ) = None,
    # Teacher network
    teacher_network_factory: types.NetworkFactory[
        ppo_networks.TeacherNetworks
    ] = ppo_networks.make_teacher_networks,
    teacher_learning_rate: float = 1e-4,
    # Student network
    student_network_factory: types.NetworkFactory[
        ppo_networks.StudentNetworks
    ] = ppo_networks.make_student_networks,
    student_learning_rate: float = 1e-4,
    student_steps_per_teacher_step: int = 2,
    # PPO parameters
    entropy_cost: float = 1e-4,
    discounting: float = 0.9,
    unroll_length: int = 10,
    batch_size: int = 32,
    num_minibatches: int = 16,
    num_updates_per_batch: int = 2,
    num_resets_per_eval: int = 0,
    normalize_observations: bool = False,
    reward_scaling: float = 1.0,
    clipping_epsilon: float = 0.3,
    gae_lambda: float = 0.95,
    max_grad_norm: float | None = None,
    normalize_advantage: bool = True,
    seed: int = 0,
    # evaluation settings
    num_evals: int = 1,
    eval_env: envs.Env | None = None,
    num_eval_envs: int = 128,
    deterministic_eval: bool = False,
    # training metrics
    log_training_metrics: bool = False,
    training_metrics_steps: int | None = None,
    # callbacks
    progress_fn: Callable[[int, ...], None] = lambda *args: None,
    policy_params_fn: Callable[..., None] = lambda *args: None,
    # checkpointing
    save_checkpoint_path: str | None = None,
    restore_checkpoint_path: str | None = None,
    restore_params=None,
    restore_value_fn: bool = True,
):
    # Check arguments
    assert batch_size * num_minibatches % num_envs == 0
    utils.validate_madrona_args(
        madrona_backend, num_envs, num_eval_envs, action_repeat, eval_env
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
    # Equals to ceil(num_timesteps / (num_evals * env_step_per_training_step *
    #                                 num_resets_per_eval))
    num_training_steps_per_epoch = np.ceil(
        num_timesteps
        / (num_evals_after_init * env_step_per_training_step * max(num_resets_per_eval, 1))
    ).astype(int)

    key = jax.random.PRNGKey(seed)
    global_key, local_key = jax.random.split(key)
    del key
    local_key = jax.random.fold_in(local_key, process_id)
    local_key, key_env, eval_key = jax.random.split(local_key, 3)
    # key_networks should be global, so that networks are initialized the same
    # way for different processes.
    key_encoder, key_adapter, key_policy, key_value = jax.random.split(global_key, 4)
    del global_key

    assert num_envs % device_count == 0

    env = utils.maybe_wrap_env(
        environment,
        wrap_env,
        num_envs,
        episode_length,
        action_repeat,
        device_count,
        key_env,
        wrap_env_fn,
        randomization_fn,
    )
    reset_fn = jax.jit(jax.vmap(env.reset))
    key_envs = jax.random.split(key_env, num_envs // process_count)
    key_envs = jnp.reshape(key_envs, (local_devices_to_use, -1) + key_envs.shape[1:])
    env_state = reset_fn(key_envs)

    if not isinstance(env_state.obs, Mapping):
        raise TypeError(
            f"Environment observations must be a dictionary (Mapping), got {type(env_state.obs)}"
        )
    required_keys = {'state', 'privileged_state', 'state_history'}
    if not required_keys.issubset(env_state.obs.keys()):
        raise ValueError(
            f"Environment observation dict missing required keys. "
            f"Expected: {required_keys}, Got: {env_state.obs.keys()}"
        )

    # Shapes of different observation tensors
    # Discard the batch axes over devices and envs.
    obs_shape = jax.tree_util.tree_map(lambda x: x.shape[2:], env_state.obs)

    # --- Networks ---
    preprocess_fn = running_statistics.normalize if normalize_observations else lambda x, y: x

    teacher_network = teacher_network_factory(
        observation_size=obs_shape,
        action_size=env.action_size,
        preprocess_observations_fn=preprocess_fn,
    )
    make_teacher_policy = ppo_networks.make_teacher_inference_fn(teacher_network)

    student_network = student_network_factory(
        observation_size=obs_shape,
        action_size=env.action_size,
        preprocess_observations_fn=preprocess_fn,
    )
    make_student_policy = ppo_networks.make_student_inference_fn(
        teacher_network, student_network
    )

    # --- Optimizers ---
    def make_optmizer(learning_rate: float):
        opt = optax.adam(learning_rate=learning_rate)
        return opt if max_grad_norm is None else optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate=learning_rate),
        )

    teacher_optimizer = make_optmizer(teacher_learning_rate)
    student_optimizer = make_optmizer(student_learning_rate)

    teacher_loss_fn = functools.partial(
        compute_teacher_loss,
        teacher_network=teacher_network,
        entropy_cost=entropy_cost,
        discounting=discounting,
        reward_scaling=reward_scaling,
        gae_lambda=gae_lambda,
        clipping_epsilon=clipping_epsilon,
        normalize_advantage=normalize_advantage,
    )
    student_loss_fn = functools.partial(
        compute_student_loss,
        teacher_network=teacher_network,
        student_network=student_network,
    )

    teacher_gradient_update_fn = gradients.gradient_update_fn(
        teacher_loss_fn, teacher_optimizer, pmap_axis_name=utils.PMAP_AXIS_NAME, has_aux=True
    )
    student_gradient_update_fn = gradients.gradient_update_fn(
        student_loss_fn, student_optimizer, pmap_axis_name=utils.PMAP_AXIS_NAME, has_aux=True
    )

    metrics_aggregator = metric_logger.EpisodeMetricsLogger(
        steps_between_logging=training_metrics_steps or env_step_per_training_step,
        progress_fn=progress_fn,
    )

    def minibatch_step(
        carry,
        data: types.Transition,
        normalizer_params: running_statistics.RunningStatisticsState,
    ):
        t_opt_state, t_params, s_opt_state, s_params, key = carry
        key, key_loss = jax.random.split(key)
        (t_loss, t_metrics), t_params, t_opt_state = teacher_gradient_update_fn(
            t_params,
            normalizer_params,
            data,
            key_loss,
            optimizer_state=t_opt_state,
        )

        def student_step(
            carry,
            data: types.Transition,
            normalizer_params: running_statistics.RunningStatisticsState,
        ):
            s_opt_state, s_params = carry
            (s_loss, s_metrics), s_params, s_opt_state = student_gradient_update_fn(
                s_params,
                t_params,
                normalizer_params,
                data,
                optimizer_state=s_opt_state,
            )
            return (s_opt_state, s_params), s_metrics

        (s_opt_state, s_params), s_metrics = jax.lax.scan(
            functools.partial(student_step, normalizer_params=normalizer_params),
            (s_opt_state, s_params),
            data,
            length=student_steps_per_teacher_step,
        )

        return (t_opt_state, t_params, s_opt_state, s_params, key), (t_metrics, s_metrics)

    def sgd_step(
        carry,
        unused_t,
        data: types.Transition,
        normalizer_params: running_statistics.RunningStatisticsState,
    ):
        t_opt_state, t_params, s_opt_state, s_params, key = carry
        key, key_perm, key_grad = jax.random.split(key, 3)

        if augment_pixels:
            key, key_rt = jax.random.split(key)
            r_translate = functools.partial(utils.random_translate_pixels, key=key_rt)
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
        (t_opt_state, t_params, s_opt_state, s_params, ignore_key), metrics = jax.lax.scan(
            functools.partial(minibatch_step, normalizer_params=normalizer_params),
            (t_opt_state, t_params, s_opt_state, s_params, key_grad),
            shuffled_data,
            length=num_minibatches,
        )
        return (t_opt_state, t_params, s_opt_state, s_params, key), metrics

    def training_step(
        carry: tuple[TrainingState, envs.State, PRNGKey], unused_t
    ) -> tuple[tuple[TrainingState, envs.State, PRNGKey], utils.Metrics]:
        training_state, state, key = carry
        key_sgd, key_generate_unroll, new_key = jax.random.split(key, 3)

        policy = make_teacher_policy(
            (
                training_state.normalizer_params,
                training_state.teacher_params.encoder,
                training_state.teacher_params.policy,
                training_state.teacher_params.value,
            )
        )

        def roll(carry, unused_t):
            current_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            next_state, data = acting.generate_unroll(
                env,
                current_state,
                policy,
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

        if log_training_metrics:  # log unroll metrics
            jax.debug.callback(
                metrics_aggregator.update_episode_metrics,
                data.extras["state_extras"]["episode_metrics"],
                data.extras["state_extras"]["episode_done"],
            )

        # Update normalization params and normalize observations.
        normalizer_params = running_statistics.update(
            training_state.normalizer_params,
            utils.remove_pixels(data.observation),
            pmap_axis_name=utils.PMAP_AXIS_NAME,
        )

        (t_opt_state, t_params, s_opt_state, s_params, ignore_key), metrics = jax.lax.scan(
            functools.partial(sgd_step, data=data, normalizer_params=normalizer_params),
            (
                training_state.teacher_optimizer_state,
                training_state.teacher_params,
                training_state.student_optimizer_state,
                training_state.student_params,
                key_sgd,
            ),
            (),
            length=num_updates_per_batch,
        )

        new_training_state = TrainingState(
            teacher_optimizer_state=t_opt_state,
            teacher_params=t_params,
            student_optimizer_state=s_opt_state,
            student_params=s_params,
            normalizer_params=normalizer_params,
            env_steps=training_state.env_steps + env_step_per_training_step,
        )
        return (new_training_state, state, new_key), metrics

    def training_epoch(
        training_state: TrainingState, state: envs.State, key: PRNGKey
    ) -> tuple[TrainingState, envs.State, utils.Metrics]:
        (training_state, state, ignored_key), loss_metrics = jax.lax.scan(
            training_step,
            (training_state, state, key),
            (),
            length=num_training_steps_per_epoch,
        )
        loss_metrics = jax.tree_util.tree_map(jnp.mean, loss_metrics)
        return training_state, state, loss_metrics

    training_epoch = jax.pmap(training_epoch, axis_name=utils.PMAP_AXIS_NAME)

    # Note that this is NOT a pure jittable method.
    def training_epoch_with_timing(
        training_state: TrainingState, env_state: envs.State, key: PRNGKey
    ) -> tuple[TrainingState, envs.State, utils.Metrics]:
        nonlocal training_walltime
        t = time.time()
        training_state, env_state = utils.strip_weak_type((training_state, env_state))
        result = training_epoch(training_state, env_state, key)
        training_state, env_state, metrics = utils.strip_weak_type(result)

        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

        epoch_training_time = time.time() - t
        training_walltime += epoch_training_time
        sps = (
                  num_training_steps_per_epoch
                  * env_step_per_training_step
                  * max(num_resets_per_eval, 1)
              ) / epoch_training_time
        metrics = {
            "training/sps": sps,
            "training/walltime": training_walltime,
            **{f"training/teacher/{name}": value for name, value in metrics[0].items()},
            **{f"training/student/{name}": value for name, value in metrics[1].items()},
        }
        return (training_state, env_state, metrics)

    # Initialize model params and training state.
    teacher_init_params = TeacherNetworkParams(
        encoder=ppo_networks.TeacherNetworks.policy_network.init(key_encoder),
        policy=ppo_networks.TeacherNetwork.policy_network.init(key_policy),
        value=ppo_networks.TeacherNetwork.value_network.init(key_value),
    )
    student_init_params = StudentNetworkParams(
        encoder=ppo_networks.StudentNetworks.encoder_network.init(key_adapter),
    )

    obs_shape = jax.tree_util.tree_map(
        lambda x: specs.Array(x.shape[-1:], jnp.dtype("float32")), env_state.obs
    )
    training_state = TrainingState(
        teacher_optimizer_state=teacher_optimizer.init(teacher_init_params),
        teacher_params=teacher_init_params,
        student_optimizer_state=student_optimizer.init(student_init_params),
        student_params=student_init_params,
        normalizer_params=running_statistics.init_state(utils.remove_pixels(obs_shape)),
        env_steps=jnp.array(0, dtype=jnp.int64),
    )

    # TODO implement checkpoint support
    # if restore_checkpoint_path is not None:
    #     params = checkpoint.load(restore_checkpoint_path)
    #     value_params = params[2] if restore_value_fn else init_params.value
    #     training_state = training_state.replace(
    #         normalizer_params=params[0],
    #         params=training_state.params.replace(policy=params[1], value=value_params),
    #     )
    #
    # if restore_params is not None:
    #     logging.info("Restoring TrainingState from `restore_params`.")
    #     value_params = restore_params[2] if restore_value_fn else init_params.value
    #     training_state = training_state.replace(
    #         normalizer_params=restore_params[0],
    #         params=training_state.params.replace(policy=restore_params[1], value=value_params),
    #     )

    if num_timesteps == 0:
        return (
            make_student_policy,
            (
                training_state.normalizer_params,
                training_state.teacher_params.encoder,
                training_state.teacher_params.policy,
                training_state.teacher_params.value,
                training_state.student_params.encoder,
            ),
            {},
        )

    training_state = jax.device_put_replicated(
        training_state, jax.local_devices()[:local_devices_to_use]
    )

    eval_env = utils.maybe_wrap_env(
        eval_env or environment,
        wrap_env,
        num_eval_envs,
        episode_length,
        action_repeat,
        device_count=1,  # eval on the host only
        key_env=eval_key,
        wrap_env_fn=wrap_env_fn,
        randomization_fn=randomization_fn,
    )
    teacher_evaluator = acting.Evaluator(
        eval_env,
        functools.partial(make_teacher_policy, deterministic=deterministic_eval),
        num_eval_envs=num_eval_envs,
        episode_length=episode_length,
        action_repeat=action_repeat,
        key=eval_key,
    )
    student_evaluator = acting.Evaluator(
        eval_env,
        functools.partial(make_student_policy, deterministic=deterministic_eval),
        num_eval_envs=num_eval_envs,
        episode_length=episode_length,
        action_repeat=action_repeat,
        key=eval_key,
    )

    # Run initial eval
    teacher_metrics = {}
    student_metrics = {}
    if process_id == 0 and num_evals > 1:
        teacher_metrics = teacher_evaluator.run_evaluation(
            utils.unpmap(
                (
                    training_state.normalizer_params,
                    training_state.teacher_params.encoder,
                    training_state.teacher_params.policy,
                    training_state.teacher_params.value,
                )
            ),
            training_metrics={},
        )
        logging.info(teacher_metrics)
        progress_fn(0, teacher_metrics)

        student_metrics = student_evaluator.run_evaluation(
            utils.unpmap(
                (
                    training_state.normalizer_params,
                    training_state.student_params.encoder,
                    training_state.teacher_params.policy,
                )
            ),
            training_metrics={},
        )
        logging.info(student_metrics)
        progress_fn(0, student_metrics)

    # TODO VVV

    training_metrics = {}
    training_walltime = 0
    current_step = 0
    for it in range(num_evals_after_init):
        logging.info("starting iteration %s %s", it, time.time() - xt)

        for _ in range(max(num_resets_per_eval, 1)):
            # optimization
            epoch_key, local_key = jax.random.split(local_key)
            epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
            (training_state, env_state, training_metrics) = training_epoch_with_timing(
                training_state, env_state, epoch_keys
            )
            current_step = int(utils.unpmap(training_state.env_steps))

            key_envs = jax.vmap(lambda x, s: jax.random.split(x[0], s), in_axes=(0, None))(
                key_envs, key_envs.shape[1]
            )
            # TODO: move extra reset logic to the AutoResetWrapper.
            env_state = reset_fn(key_envs) if num_resets_per_eval > 0 else env_state

        if process_id != 0:
            continue

        # Process id == 0.
        teacher_params = utils.unpmap(
            (
                training_state.normalizer_params,
                training_state.teacher_params.encoder,
                training_state.teacher_params.policy,
                training_state.teacher_params.value,
            )
        )
        student_params = utils.unpmap(
            (
                training_state.normalizer_params,
                training_state.student_params.encoder,
                training_state.teacher_params.policy,
            )
        )

        policy_params_fn(current_step, make_teacher_policy, teacher_params)  # TODO: what?

        # if save_checkpoint_path is not None:
        #     ckpt_config = checkpoint.network_config(
        #         observation_size=obs_shape,
        #         action_size=env.action_size,
        #         normalize_observations=normalize_observations,
        #         network_factory=network_factory,
        #     )
        #     checkpoint.save(save_checkpoint_path, current_step, params, ckpt_config)

        if num_evals > 0:
            teacher_metrics = teacher_evaluator.run_evaluation(
                teacher_params,
                training_metrics,
            )
            logging.info(teacher_metrics)
            progress_fn(current_step, teacher_metrics)
            student_metrics = student_evaluator.run_evaluation(
                student_params,
                training_metrics,
            )
            logging.info(student_metrics)
            progress_fn(current_step, student_metrics)

    total_steps = current_step
    if not total_steps >= num_timesteps:
        raise AssertionError(
            f"Total steps {total_steps} is less than `num_timesteps`=" f" {num_timesteps}."
        )

    # If there were no mistakes the training_state should still be identical on all
    # devices.
    pmap.assert_is_replicated(training_state)
    teacher_params = utils.unpmap(
        (
            training_state.normalizer_params,
            training_state.teacher_params.encoder,
            training_state.teacher_params.policy,
            training_state.teacher_params.value,
        )
    )
    student_params = utils.unpmap(
        (
            training_state.normalizer_params,
            training_state.student_params.encoder,
            training_state.teacher_params.policy,
        )
    )
    logging.info("total steps: %s", total_steps)
    pmap.synchronize_hosts()
    metrics = (teacher_metrics, student_metrics)
    return make_student_policy, teacher_params, student_params, metrics