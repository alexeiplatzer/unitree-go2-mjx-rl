import functools

import jax
from flax import struct as flax_struct
from jax import numpy as jnp

from quadruped_mjx_rl.domain_randomization import (
    DomainRandomizationFn,
    TerrainMapRandomizationFn,
)
from quadruped_mjx_rl.environments.physics_pipeline.base import PipelineModel, EnvModel
from quadruped_mjx_rl.environments.physics_pipeline.environments import Env, State, Wrapper
from quadruped_mjx_rl.types import PRNGKey


def wrap_for_training(
    env: Env,
    num_envs: int = 1,
    device_count: int = 1,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn: DomainRandomizationFn | TerrainMapRandomizationFn | None = None,
    rng_key: PRNGKey | None = None,
    vision: bool = False,
    num_terrain_colors: int = 1,
) -> Wrapper:
    """
    Common wrapper pattern for all training agents.

    Args:
        env: environment to be wrapped
        num_envs: total number of environments running in parallel
        device_count: number of devices across which the parallel environments are split
        episode_length: length of episode
        action_repeat: how many repeated actions to take per step
        randomization_fn: randomization function that produces a vectorized system
            and in_axes to vmap over
        rng_key: an RNG key used for domain randomization
        vision: whether the environment will be vision-based
        num_terrain_colors: number of terrain colors to use for terrain map randomization

    Returns:
        An environment that is wrapped with Episode and AutoReset wrappers.  If the
        environment did not already have batch dimensions, it is additionally Vmap-wrapped.
    """
    if randomization_fn is None and not vision:
        env = VmapWrapper(env)
    else:
        # all the devices get the same randomization seed
        local_num_envs = num_envs // device_count
        # envs_key = jax.random.split(rng_key, local_num_envs)
        env = _vmap_wrap_with_randomization(
            env,
            vision=vision,
            num_worlds=local_num_envs,
            randomization_fn=randomization_fn,
            worlds_random_key=rng_key,
            num_terrain_colors=num_terrain_colors,
        )
    env = EpisodeWrapper(env, episode_length, action_repeat)
    env = AutoResetWrapper(env)
    return env


def _vmap_wrap_with_randomization(
    env: Env,
    vision: bool = False,
    num_worlds: int = 1,
    randomization_fn: DomainRandomizationFn | TerrainMapRandomizationFn | None = None,
    worlds_random_key: PRNGKey | None = None,
    num_terrain_colors: int = 1,
) -> Wrapper:
    assert vision or randomization_fn is not None
    assert not randomization_fn or worlds_random_key is not None

    if num_terrain_colors > 1:
        assert randomization_fn is not None
        # let the randomization function have the same interface as in the single color case
        # randomization_fn = functools.partial(randomization_fn, num_colors=num_terrain_colors)
        wrapper = functools.partial(TerrainMapWrapper, num_colors=num_terrain_colors)
    else:
        wrapper = DomainRandomizationVmapWrapper

    if vision:
        if not randomization_fn:
            randomization_fn = _identity_vision_randomization_fn
        else:
            randomization_fn = functools.partial(
                _supplement_vision_randomization_fn,
                randomization_fn=randomization_fn,
            )

    wrapped_env = wrapper(
        env=env,
        randomization_fn=randomization_fn,
        key_envs=worlds_random_key,
        num_envs=num_worlds,
    )

    if vision:
        # For user-made DR functions, ensure that the output model includes the
        # necessary in_axes and has the correct shape for madrona initialization.
        required_fields = [
            "geom_rgba",
            "geom_matid",
            "geom_size",
            "light_pos",
            "light_dir",
            "light_type",
            "light_castshadow",
            "light_cutoff",
        ]
        for field in required_fields:
            assert hasattr(wrapped_env._in_axes.model, field), f"{field} not in in_axes"
            assert (
                getattr(wrapped_env._sys_v.model, field).shape[0] == num_worlds
            ), f"{field} shape does not match num_worlds"

    return wrapped_env


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
        randomization_fn: DomainRandomizationFn,
        key_envs: PRNGKey,
        num_envs: int,
    ):
        super().__init__(env)
        self._sys_v, self._in_axes = randomization_fn(
            pipeline_model=self.pipeline_model,
            env_model=self.env_model,
            rng_key=key_envs,
            num_worlds=num_envs,
        )

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


def _identity_vision_randomization_fn(
    pipeline_model: PipelineModel,
    env_model: EnvModel,
    rng_key: PRNGKey,
    num_worlds: int,
) -> tuple[PipelineModel, PipelineModel]:
    """Tile the necessary fields for the Madrona memory buffer copy."""
    in_axes = jax.tree_util.tree_map(lambda x: None, pipeline_model)
    in_axes = in_axes.replace(
        model=in_axes.model.tree_replace(
            {
                "geom_rgba": 0,
                "geom_matid": 0,
                "geom_size": 0,
                "light_pos": 0,
                "light_dir": 0,
                "light_type": 0,
                "light_castshadow": 0,
                "light_cutoff": 0,
            }
        )
    )
    pipeline_model = pipeline_model.replace(
        model=pipeline_model.model.tree_replace(
            {
                "geom_rgba": jnp.repeat(
                    jnp.expand_dims(pipeline_model.model.geom_rgba, 0), num_worlds, axis=0
                ),
                "geom_matid": jnp.repeat(
                    jnp.expand_dims(
                        jnp.repeat(-1, pipeline_model.model.geom_matid.shape[0], 0), 0
                    ),
                    num_worlds,
                    axis=0,
                ),
                "geom_size": jnp.repeat(
                    jnp.expand_dims(pipeline_model.model.geom_size, 0), num_worlds, axis=0
                ),
                "light_pos": jnp.repeat(
                    jnp.expand_dims(pipeline_model.model.light_pos, 0), num_worlds, axis=0
                ),
                "light_dir": jnp.repeat(
                    jnp.expand_dims(pipeline_model.model.light_dir, 0), num_worlds, axis=0
                ),
                "light_type": jnp.repeat(
                    jnp.expand_dims(pipeline_model.model.light_type, 0), num_worlds, axis=0
                ),
                "light_castshadow": jnp.repeat(
                    jnp.expand_dims(pipeline_model.model.light_castshadow, 0),
                    num_worlds,
                    axis=0,
                ),
                "light_cutoff": jnp.repeat(
                    jnp.expand_dims(pipeline_model.model.light_cutoff, 0), num_worlds, axis=0
                ),
            }
        )
    )
    return pipeline_model, in_axes


def _supplement_vision_randomization_fn(
    pipeline_model: PipelineModel,
    env_model: EnvModel,
    rng_key: PRNGKey,
    num_worlds: int,
    randomization_fn: DomainRandomizationFn | TerrainMapRandomizationFn,
    **randomization_fn_kwargs,
) -> tuple[PipelineModel, PipelineModel] | tuple[PipelineModel, PipelineModel, tuple]:
    """Tile the necessary missing fields for the Madrona memory buffer copy."""
    randomization_outputs = randomization_fn(
        pipeline_model=pipeline_model,
        env_model=env_model,
        rng_key=rng_key,
        num_worlds=num_worlds,
        **randomization_fn_kwargs,
    )
    pipeline_model, in_axes, *rest_outputs = randomization_outputs

    required_fields = [
        "geom_rgba",
        "geom_matid",
        "geom_size",
        "light_pos",
        "light_dir",
        "light_type",
        "light_castshadow",
        "light_cutoff",
    ]

    for field in required_fields:
        if getattr(in_axes.model, field) is None:
            in_axes = in_axes.replace(model=in_axes.model.tree_replace({field: 0}))
            val = -2 if field == "geom_matid" else getattr(pipeline_model.model, field)
            pipeline_model = pipeline_model.replace(
                model=pipeline_model.model.tree_replace(
                    {
                        field: jnp.repeat(jnp.expand_dims(val, 0), num_worlds, axis=0),
                    }
                )
            )
    return pipeline_model, in_axes, *rest_outputs


class TerrainMapWrapper(Wrapper):
    """Wrapper for domain randomization with colored terrain, preserving the terrain info."""

    def __init__(
        self,
        env: Env,
        randomization_fn: TerrainMapRandomizationFn,
        key_envs: jax.Array,
        num_envs: int,
        num_colors: int = 2,
    ):
        super().__init__(env)
        (
            self._sys_v,
            self._in_axes,
            (self._rgba_table_v, self._friction_table_v, self._stiffness_table_v),
        ) = randomization_fn(
            pipeline_model=self.pipeline_model,
            env_model=self.env_model,
            rng_key=key_envs,
            num_worlds=num_envs,
            num_colors=num_colors,
        )
        self._rgba_table = jnp.zeros((num_colors, 4))
        self._friction_table = jnp.zeros((num_colors,))
        self._stiffness_table = jnp.zeros((num_colors,))

    def _env_fn(
        self,
        pipeline_model: PipelineModel,
        rgba_table: jax.Array,
        friction_table: jax.Array,
        stiffness_table: jax.Array,
    ) -> Env:
        env = self.env
        env.unwrapped._pipeline_model = pipeline_model
        env.unwrapped._rgba_table = rgba_table
        env.unwrapped._friction_table = friction_table
        env.unwrapped._stiffness_table = stiffness_table
        return env

    def reset(self, rng: jax.Array) -> State:
        def reset(pipeline_model, rgba_table, friction_table, stiffness_table, rng):
            env = self._env_fn(
                pipeline_model=pipeline_model,
                rgba_table=rgba_table,
                friction_table=friction_table,
                stiffness_table=stiffness_table,
            )
            return env.reset(rng)

        state = jax.vmap(reset, in_axes=[self._in_axes, 0])(
            self._sys_v,
            self._rgba_table_v,
            self._friction_table_v,
            self._stiffness_table_v,
            rng,
        )
        return state

    def step(self, state: State, action: jax.Array) -> State:
        def step(
            pipeline_model,
            rgba_table,
            friction_table,
            stiffness_table,
            state_local,
            action_local,
        ):
            env = self._env_fn(
                pipeline_model=pipeline_model,
                rgba_table=rgba_table,
                friction_table=friction_table,
                stiffness_table=stiffness_table,
            )
            return env.step(state_local, action_local)

        res = jax.vmap(step, in_axes=[self._in_axes, 0, 0])(
            self._sys_v,
            self._rgba_table_v,
            self._friction_table_v,
            self._stiffness_table_v,
            state,
            action,
        )
        return res
