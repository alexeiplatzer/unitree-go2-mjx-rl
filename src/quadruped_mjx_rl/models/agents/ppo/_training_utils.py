import functools
from collections.abc import Callable, Mapping

import flax
import jax
import jax.numpy as jnp
import optax
from brax import base
from brax import envs
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.types import Params
from brax.training.types import PRNGKey

from quadruped_mjx_rl.models.agents.ppo.guided_ppo.losses import StudentNetworkParams
from quadruped_mjx_rl.models.agents.ppo.guided_ppo.losses import TeacherNetworkParams
from quadruped_mjx_rl.environments.wrappers import MadronaWrapper

Metrics = types.Metrics

PMAP_AXIS_NAME = "i"


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""

    teacher_optimizer_state: optax.OptState
    teacher_params: TeacherNetworkParams
    student_optimizer_state: optax.OptState
    student_params: StudentNetworkParams
    normalizer_params: running_statistics.RunningStatisticsState
    env_steps: jnp.ndarray


def unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)


def strip_weak_type(tree):
    # brax user code is sometimes ambiguous about weak_type.
    # to avoid extra jit recompilations, we strip all weak types from user input
    def f(leaf):
        leaf = jnp.asarray(leaf)
        return leaf.astype(leaf.dtype)

    return jax.tree_util.tree_map(f, tree)


def validate_madrona_args(
    madrona_backend: bool,
    num_envs: int,
    num_eval_envs: int,
    action_repeat: int,
    eval_env: envs.Env | None = None,
):
    """Validates arguments for Madrona-MJX."""
    if madrona_backend:
        if eval_env:
            raise ValueError("Madrona-MJX doesn't support multiple env instances")
        if num_eval_envs != num_envs:
            raise ValueError("Madrona-MJX requires a fixed batch size")
        if action_repeat != 1:
            raise ValueError(
                "Implement action_repeat using PipelineEnv's _n_frames to avoid"
                " unnecessary rendering!"
            )


def maybe_wrap_env(
    env: envs.Env,
    wrap_env: bool,
    num_envs: int,
    episode_length: int | None,
    action_repeat: int,
    device_count: int,
    key_env: PRNGKey,
    wrap_env_fn: Callable | None = None,
    randomization_fn: Callable[
        [base.System, jnp.ndarray], tuple[base.System, base.System]
    ] | None = None,
    vision: bool = False,
    num_vision_envs: int = 1,
):
    """Wraps the environment for training/eval if wrap_env is True."""
    if not wrap_env:
        return env
    if episode_length is None:
        raise ValueError("episode_length must be specified in train")
    v_randomization_fn = None
    if randomization_fn is not None:
        randomization_batch_size = num_envs // device_count
        # all devices get the same randomization rng
        randomization_rng = jax.random.split(key_env, randomization_batch_size)
        v_randomization_fn = functools.partial(randomization_fn, rng=randomization_rng)
    wrap_for_training = wrap_env_fn or envs.training.wrap
    if vision:
        env = MadronaWrapper(env, num_vision_envs, randomization_fn)
    return wrap_for_training(
        env,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=v_randomization_fn,
    )


def random_translate_pixels(
    obs: Mapping[str, jax.Array], key: PRNGKey
) -> Mapping[str, jax.Array]:
    """Apply random translations to B x T x ... pixel observations.

    The same shift is applied across the unroll_length (T) dimension.

    Args:
      obs: a dictionary of observations
      key: a PRNGKey

    Returns:
      A dictionary of observations with translated pixels
    """

    @jax.vmap
    def rt_all_views(ub_obs: Mapping[str, jax.Array], key: PRNGKey) -> Mapping[str, jax.Array]:
        # Expects a dictionary of unbatched observations.
        def rt_view(img: jax.Array, padding: int, key: PRNGKey) -> jax.Array:  # TxHxWxC
            # Randomly translates a set of pixel inputs.
            # Adapted from https://github.com/ikostrikov/jaxrl/blob/main/jaxrl/agents/drq/augmentations.py
            crop_from = jax.random.randint(key, (2,), 0, 2 * padding + 1)
            zero = jnp.zeros((1,), dtype=jnp.int32)
            crop_from = jnp.concatenate([zero, crop_from, zero])
            padded_img = jnp.pad(
                img,
                ((0, 0), (padding, padding), (padding, padding), (0, 0)),
                mode="edge",
            )
            return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)

        out = {}
        for k_view, v_view in ub_obs.items():
            if k_view.startswith("pixels/"):
                key, key_shift = jax.random.split(key)
                out[k_view] = rt_view(v_view, 4, key_shift)
        return {**ub_obs, **out}

    bdim = next(iter(obs.items()), None)[1].shape[0]
    keys = jax.random.split(key, bdim)
    obs = rt_all_views(obs, keys)
    return obs


def remove_pixels(
    obs: jnp.ndarray | Mapping[str, jax.Array],
) -> jnp.ndarray | Mapping[str, jax.Array]:
    """Removes pixel observations from the observation dict."""
    if not isinstance(obs, Mapping):
        return obs
    return {k: v for k, v in obs.items() if not k.startswith("pixels/")}
