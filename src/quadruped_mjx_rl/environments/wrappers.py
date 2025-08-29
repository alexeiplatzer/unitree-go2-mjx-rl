import functools
from collections.abc import Callable

import jax
from jax import numpy as jnp

from quadruped_mjx_rl.environments.physics_pipeline import (
    AutoResetWrapper,
    DomainRandomizationVmapWrapper,
    Env,
    EpisodeWrapper,
    PipelineModel,
    State,
    VmapWrapper,
    Wrapper,
)


def wrap_for_training(
    env: Env,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn: (
        Callable[[PipelineModel], tuple[PipelineModel, PipelineModel]] | None
    ) = None,
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


def _identity_vision_randomization_fn(
    pipeline_model: PipelineModel, num_worlds: int
) -> tuple[PipelineModel, PipelineModel]:
    """Tile the necessary fields for the Madrona memory buffer copy."""
    in_axes = jax.tree_util.tree_map(lambda x: None, pipeline_model)
    in_axes = in_axes.tree_replace(
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
    randomization_fn: Callable[[PipelineModel], tuple[PipelineModel, PipelineModel]],
    num_worlds: int,
) -> tuple[PipelineModel, PipelineModel]:
    """Tile the necessary missing fields for the Madrona memory buffer copy."""
    pipeline_model, in_axes = randomization_fn(pipeline_model)

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
        if getattr(in_axes, field) is None:
            in_axes = in_axes.tree_replace({field: 0})
            val = -2 if field == "geom_matid" else getattr(pipeline_model, field)
            pipeline_model = pipeline_model.replace(
                model=pipeline_model.model.tree_replace(
                    {
                        field: jnp.repeat(jnp.expand_dims(val, 0), num_worlds, axis=0),
                    }
                )
            )
    return pipeline_model, in_axes


class MadronaWrapper(Wrapper):
    """Wraps an Environment to be used in Brax with Madrona."""

    def __init__(
        self,
        env: Env,
        num_worlds: int,
        randomization_fn: (
            Callable[[PipelineModel], tuple[PipelineModel, PipelineModel]] | None
        ) = None,
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
            "light_type",
            "light_castshadow",
            "light_cutoff",
        ]
        for field in required_fields:
            assert hasattr(self.env._in_axes, field), f"{field} not in in_axes"
            assert (
                getattr(self.env._sys_v.model, field).shape[0] == num_worlds
            ), f"{field} shape does not match num_worlds"

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        return self.env.reset(rng)

    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        return self.env.step(state, action)
