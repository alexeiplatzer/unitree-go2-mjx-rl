# Typing
from collections.abc import Callable

# Supporting
import functools

# Math
import jax
from jax import numpy as jnp
from flax import struct as flax_struct

# Sim
from quadruped_mjx_rl.environments.base import (
    System,
    Env,
    State,
    Wrapper,
)


def wrap_for_training(
    env: Env,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn: Callable[[System], tuple[System, System]] | None = None,
    vision: bool = False,
    num_vision_envs: int = 1,
) -> Wrapper:
    """
    Common wrapper pattern for all training agents.

    Args:
        env: environment to be wrapped
        episode_length: length of episode
        action_repeat: how many repeated actions to take per step
        randomization_fn: randomization function that produces a vectorized system
            and in_axes to vmap over
        vision: whether the environment will be vision-based
        num_vision_envs: number of environments the renderer should generate,
            should equal the number of batched envs

    Returns:
        An environment that is wrapped with Episode and AutoReset wrappers.  If the
        environment did not already have batch dimensions, it is additionally Vmap-wrapped.
  """
    if vision:
        env = MadronaWrapper(env, num_vision_envs, randomization_fn)
    elif randomization_fn is None:
        env = VmapWrapper(env)
    else:
        env = DomainRandomizationVmapWrapper(env, randomization_fn)
    env = EpisodeWrapper(env, episode_length, action_repeat)
    env = AutoResetWrapper(env)
    return env


def _identity_vision_randomization_fn(sys: System, num_worlds: int) -> tuple[System, System]:
    """Tile the necessary fields for the Madrona memory buffer copy."""
    in_axes = jax.tree_util.tree_map(lambda x: None, sys)
    in_axes = in_axes.tree_replace(
        {
            "geom_rgba": 0,
            "geom_matid": 0,
            "geom_size": 0,
            "light_pos": 0,
            "light_dir": 0,
            "light_directional": 0,
            "light_castshadow": 0,
            "light_cutoff": 0,
        }
    )
    sys = sys.tree_replace(
        {
            "geom_rgba": jnp.repeat(jnp.expand_dims(sys.geom_rgba, 0), num_worlds, axis=0),
            "geom_matid": jnp.repeat(
                jnp.expand_dims(jnp.repeat(-1, sys.geom_matid.shape[0], 0), 0),
                num_worlds,
                axis=0,
            ),
            "geom_size": jnp.repeat(jnp.expand_dims(sys.geom_size, 0), num_worlds, axis=0),
            "light_pos": jnp.repeat(jnp.expand_dims(sys.light_pos, 0), num_worlds, axis=0),
            "light_dir": jnp.repeat(jnp.expand_dims(sys.light_dir, 0), num_worlds, axis=0),
            "light_directional": jnp.repeat(
                jnp.expand_dims(sys.light_directional, 0), num_worlds, axis=0
            ),
            "light_castshadow": jnp.repeat(
                jnp.expand_dims(sys.light_castshadow, 0), num_worlds, axis=0
            ),
            "light_cutoff": jnp.repeat(
                jnp.expand_dims(sys.light_cutoff, 0), num_worlds, axis=0
            ),
        }
    )
    return sys, in_axes


def _supplement_vision_randomization_fn(
    sys: System,
    randomization_fn: Callable[[System], tuple[System, System]],
    num_worlds: int,
) -> tuple[System, System]:
    """Tile the necessary missing fields for the Madrona memory buffer copy."""
    sys, in_axes = randomization_fn(sys)

    required_fields = [
        "geom_rgba",
        "geom_matid",
        "geom_size",
        "light_pos",
        "light_dir",
        "light_directional",
        "light_castshadow",
        "light_cutoff",
    ]

    for field in required_fields:
        if getattr(in_axes, field) is None:
            in_axes = in_axes.tree_replace({field: 0})
            val = -2 if field == "geom_matid" else getattr(sys, field)
            sys = sys.tree_replace(
                {
                    field: jnp.repeat(jnp.expand_dims(val, 0), num_worlds, axis=0),
                }
            )
    return sys, in_axes


class MadronaWrapper(Wrapper):
    """Wraps an Environment to be used in Brax with Madrona."""

    def __init__(
        self,
        env: Env,
        num_worlds: int,
        randomization_fn: Callable[[System], tuple[System, System]] | None = None,
    ):
        if not randomization_fn:
            randomization_fn = functools.partial(
                _identity_vision_randomization_fn, num_worlds=num_worlds
            )
        else:
            randomization_fn = functools.partial(
                _supplement_vision_randomization_fn,
                randomization_fn=randomization_fn,
                num_worlds=num_worlds,
            )
        self.env = DomainRandomizationVmapWrapper(env, randomization_fn)
        self.num_worlds = num_worlds

        # For user-made DR functions, ensure that the output model includes the
        # necessary in_axes and has the correct shape for madrona initialization.
        required_fields = [
            "geom_rgba",
            "geom_matid",
            "geom_size",
            "light_pos",
            "light_dir",
            "light_directional",
            "light_castshadow",
            "light_cutoff",
        ]
        for field in required_fields:
            assert hasattr(self.env._in_axes, field), f"{field} not in in_axes"
            assert (
                getattr(self.env._sys_v, field).shape[0] == num_worlds
            ), f"{field} shape does not match num_worlds"

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        return self.env.reset(rng)

    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        return self.env.step(state, action)


class VmapWrapper(Wrapper):
    """Vectorizes Brax env."""

    def __init__(self, env: Env, batch_size: int | None = None):
        super().__init__(env)
        self.batch_size = batch_size

    def reset(self, rng: jax.Array) -> State:
        if self.batch_size is not None:
            rng = jax.random.split(rng, self.batch_size)
        return jax.vmap(self.env.reset)(rng)

    def step(self, state: State, action: jax.Array) -> State:
        return jax.vmap(self.env.step)(state, action)


class EpisodeWrapper(Wrapper):
    """Maintains episode step count and sets done at the episode end."""

    def __init__(self, env: Env, episode_length: int, action_repeat: int):
        super().__init__(env)
        self.episode_length = episode_length
        self.action_repeat = action_repeat

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info['steps'] = jnp.zeros(rng.shape[:-1])
        state.info['truncation'] = jnp.zeros(rng.shape[:-1])
        # Keep a separate record of episode-done as state.info['done'] can be erased
        # by AutoResetWrapper
        state.info['episode_done'] = jnp.zeros(rng.shape[:-1])
        episode_metrics = dict()
        episode_metrics['sum_reward'] = jnp.zeros(rng.shape[:-1])
        episode_metrics['length'] = jnp.zeros(rng.shape[:-1])
        for metric_name in state.metrics.keys():
            episode_metrics[metric_name] = jnp.zeros(rng.shape[:-1])
        state.info['episode_metrics'] = episode_metrics
        return state

    def step(self, state: State, action: jax.Array) -> State:
        def f(state, _):
            nstate = self.env.step(state, action)
            return nstate, nstate.reward

        state, rewards = jax.lax.scan(f, state, (), self.action_repeat)
        state = state.replace(reward=jnp.sum(rewards, axis=0))
        steps = state.info['steps'] + self.action_repeat
        one = jnp.ones_like(state.done)
        zero = jnp.zeros_like(state.done)
        episode_length = jnp.array(self.episode_length, dtype=jnp.int32)
        done = jnp.where(steps >= episode_length, one, state.done)
        state.info['truncation'] = jnp.where(
            steps >= episode_length, 1 - state.done, zero
        )
        state.info['steps'] = steps

        # Aggregate state metrics into episode metrics
        prev_done = state.info['episode_done']
        state.info['episode_metrics']['sum_reward'] += jnp.sum(rewards, axis=0)
        state.info['episode_metrics']['sum_reward'] *= (1 - prev_done)
        state.info['episode_metrics']['length'] += self.action_repeat
        state.info['episode_metrics']['length'] *= (1 - prev_done)
        for metric_name in state.metrics.keys():
            if metric_name != 'reward':
                state.info['episode_metrics'][metric_name] += state.metrics[metric_name]
                state.info['episode_metrics'][metric_name] *= (1 - prev_done)
        state.info['episode_done'] = done
        return state.replace(done=done)


class AutoResetWrapper(Wrapper):
    """Automatically resets Brax envs that are done."""

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info['first_pipeline_state'] = state.pipeline_state
        state.info['first_obs'] = state.obs
        return state

    def step(self, state: State, action: jax.Array) -> State:
        if 'steps' in state.info:
            steps = state.info['steps']
            steps = jnp.where(state.done, jnp.zeros_like(steps), steps)
            state.info.update(steps=steps)
        state = state.replace(done=jnp.zeros_like(state.done))
        state = self.env.step(state, action)

        def where_done(x, y):
            done = state.done
            if done.shape:
                done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
            return jnp.where(done, x, y)

        pipeline_state = jax.tree.map(
            where_done, state.info['first_pipeline_state'], state.pipeline_state
        )
        obs = jax.tree.map(where_done, state.info['first_obs'], state.obs)
        return state.replace(pipeline_state=pipeline_state, obs=obs)


@flax_struct.dataclass
class EvalMetrics:
    """
    Dataclass holding evaluation metrics for Brax.

    Attributes:
        episode_metrics: Aggregated episode metrics since the beginning of the episode.
        active_episodes: Boolean vector tracking which episodes are not done yet.
        episode_steps: Integer vector tracking the number of steps in the episode.
    """

    episode_metrics: dict[str, jax.Array]
    active_episodes: jax.Array
    episode_steps: jax.Array


class EvalWrapper(Wrapper):
    """Brax env with eval metrics."""

    def reset(self, rng: jax.Array) -> State:
        reset_state = self.env.reset(rng)
        reset_state.metrics['reward'] = reset_state.reward
        eval_metrics = EvalMetrics(
            episode_metrics=jax.tree_util.tree_map(
                jnp.zeros_like, reset_state.metrics
            ),
            active_episodes=jnp.ones_like(reset_state.reward),
            episode_steps=jnp.zeros_like(reset_state.reward),
        )
        reset_state.info['eval_metrics'] = eval_metrics
        return reset_state

    def step(self, state: State, action: jax.Array) -> State:
        state_metrics = state.info['eval_metrics']
        if not isinstance(state_metrics, EvalMetrics):
            raise ValueError(
                f'Incorrect type for state_metrics: {type(state_metrics)}'
            )
        del state.info['eval_metrics']
        nstate = self.env.step(state, action)
        nstate.metrics['reward'] = nstate.reward
        episode_steps = jnp.where(
            state_metrics.active_episodes,
            nstate.info['steps'],
            state_metrics.episode_steps,
        )
        episode_metrics = jax.tree_util.tree_map(
            lambda a, b: a + b * state_metrics.active_episodes,
            state_metrics.episode_metrics,
            nstate.metrics,
        )
        active_episodes = state_metrics.active_episodes * (1 - nstate.done)

        eval_metrics = EvalMetrics(
            episode_metrics=episode_metrics,
            active_episodes=active_episodes,
            episode_steps=episode_steps,
        )
        nstate.info['eval_metrics'] = eval_metrics
        return nstate


class DomainRandomizationVmapWrapper(Wrapper):
    """Wrapper for domain randomization."""

    def __init__(
        self,
        env: Env,
        randomization_fn: Callable[[System], tuple[System, System]],
    ):
        super().__init__(env)
        self._sys_v, self._in_axes = randomization_fn(self.sys)

    def _env_fn(self, sys: System) -> Env:
        env = self.env
        env.unwrapped.sys = sys
        return env

    def reset(self, rng: jax.Array) -> State:
        def reset(sys, rng):
            env = self._env_fn(sys=sys)
            return env.reset(rng)

        state = jax.vmap(reset, in_axes=[self._in_axes, 0])(self._sys_v, rng)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        def step(sys, s, a):
            env = self._env_fn(sys=sys)
            return env.step(s, a)

        res = jax.vmap(step, in_axes=[self._in_axes, 0, 0])(
            self._sys_v, state, action
        )
        return res
