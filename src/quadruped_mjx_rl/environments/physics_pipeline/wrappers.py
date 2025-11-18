from collections.abc import Callable

import jax
from flax import struct as flax_struct
from jax import numpy as jnp

from quadruped_mjx_rl.environments.physics_pipeline.base import (
    EnvModel,
    PipelineModel,
    PipelineState,
)
from quadruped_mjx_rl.environments.physics_pipeline.environments import Env, State, Wrapper


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
        state.info["steps"] = jnp.zeros(rng.shape[:-1])
        state.info["truncation"] = jnp.zeros(rng.shape[:-1])
        # Keep a separate record of "episode-done" as state.info['done'] can be erased
        # by AutoResetWrapper
        state.info["episode_done"] = jnp.zeros(rng.shape[:-1])
        episode_metrics = dict()
        episode_metrics["sum_reward"] = jnp.zeros(rng.shape[:-1])
        episode_metrics["length"] = jnp.zeros(rng.shape[:-1])
        for metric_name in state.metrics.keys():
            episode_metrics[metric_name] = jnp.zeros(rng.shape[:-1])
        state.info["episode_metrics"] = episode_metrics
        return state

    def step(self, state: State, action: jax.Array) -> State:
        def f(state, _):
            next_state = self.env.step(state, action)
            return next_state, next_state.reward

        state, rewards = jax.lax.scan(f, state, (), self.action_repeat)
        state = state.replace(reward=jnp.sum(rewards, axis=0))
        steps = state.info["steps"] + self.action_repeat
        one = jnp.ones_like(state.done)
        zero = jnp.zeros_like(state.done)
        episode_length = jnp.array(self.episode_length, dtype=jnp.int32)
        done = jnp.where(steps >= episode_length, one, state.done)
        state.info["truncation"] = jnp.where(steps >= episode_length, 1 - state.done, zero)
        state.info["steps"] = steps

        # Aggregate state metrics into episode metrics
        prev_done = state.info["episode_done"]
        state.info["episode_metrics"]["sum_reward"] += jnp.sum(rewards, axis=0)
        state.info["episode_metrics"]["sum_reward"] *= 1 - prev_done
        state.info["episode_metrics"]["length"] += self.action_repeat
        state.info["episode_metrics"]["length"] *= 1 - prev_done
        for metric_name in state.metrics.keys():
            if metric_name != "reward":
                state.info["episode_metrics"][metric_name] += state.metrics[metric_name]
                state.info["episode_metrics"][metric_name] *= 1 - prev_done
        state.info["episode_done"] = done
        return state.replace(done=done)


class AutoResetWrapper(Wrapper):
    """Automatically resets Brax envs that are done."""

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info["first_pipeline_state"] = state.pipeline_state
        state.info["first_obs"] = state.obs
        return state

    def step(self, state: State, action: jax.Array) -> State:
        if "steps" in state.info:
            steps = state.info["steps"]
            steps = jnp.where(state.done, jnp.zeros_like(steps), steps)
            state.info.update(steps=steps)
        state = state.replace(done=jnp.zeros_like(state.done))
        state = self.env.step(state, action)

        def where_done(x, y):
            done = state.done
            if done.shape:
                done = jnp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
            return jnp.where(done, x, y)

        pipeline_state = jax.tree.map(
            where_done, state.info["first_pipeline_state"], state.pipeline_state
        )
        obs = jax.tree.map(where_done, state.info["first_obs"], state.obs)
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
        reset_state.metrics["reward"] = reset_state.reward
        eval_metrics = EvalMetrics(
            episode_metrics=jax.tree_util.tree_map(jnp.zeros_like, reset_state.metrics),
            active_episodes=jnp.ones_like(reset_state.reward),
            episode_steps=jnp.zeros_like(reset_state.reward),
        )
        reset_state.info["eval_metrics"] = eval_metrics
        return reset_state

    def step(self, state: State, action: jax.Array) -> State:
        state_metrics = state.info["eval_metrics"]
        if not isinstance(state_metrics, EvalMetrics):
            raise ValueError(f"Incorrect type for state_metrics: {type(state_metrics)}")
        del state.info["eval_metrics"]
        nstate = self.env.step(state, action)
        nstate.metrics["reward"] = nstate.reward
        episode_steps = jnp.where(
            state_metrics.active_episodes,
            nstate.info["steps"],
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
        nstate.info["eval_metrics"] = eval_metrics
        return nstate


class DomainRandomizationVmapWrapper(Wrapper):
    """Wrapper for domain randomization."""

    def __init__(
        self,
        env: Env,
        randomization_fn: Callable[[PipelineModel], tuple[PipelineModel, PipelineModel]],
    ):
        super().__init__(env)
        self._sys_v, self._in_axes = randomization_fn(self.pipeline_model)

    def _env_fn(self, pipeline_model: PipelineModel) -> Env:
        env = self.env
        env.unwrapped._pipeline_model = pipeline_model
        return env

    def reset(self, rng: jax.Array) -> State:
        def reset(pipeline_model, rng):
            env = self._env_fn(pipeline_model=pipeline_model)
            return env.reset(rng)

        state = jax.vmap(reset, in_axes=[self._in_axes, 0])(self._sys_v, rng)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        def step(pipeline_model, s, a):
            env = self._env_fn(pipeline_model=pipeline_model)
            return env.step(s, a)

        res = jax.vmap(step, in_axes=[self._in_axes, 0, 0])(self._sys_v, state, action)
        return res
