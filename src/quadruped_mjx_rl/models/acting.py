"""Brax training acting functions."""

import functools
from collections.abc import Callable, Sequence
from typing import Any, Protocol

import jax
from jax import numpy as jnp

from quadruped_mjx_rl.environments.vision.vision_wrappers import VisionWrapper
from quadruped_mjx_rl.models.types import (
    Policy,
    RecurrentCarry,
    RecurrentEncoder,
)
from quadruped_mjx_rl.physics_pipeline import Env, State
from quadruped_mjx_rl.types import Observation, PRNGKey, Transition


class GenerateUnrollFn(Protocol):
    def __call__(
        self,
        env_state: State,
        key: PRNGKey,
        env: Env,
        unroll_length: int,
        extra_fields: Sequence[str] = (),
    ) -> tuple[State, Any]:
        pass


def unroll_factory(
    fun: Callable[..., tuple],
    initial_carry_extras: tuple = (),
    remove_roll_axis: bool = False,
    **kwargs,
) -> GenerateUnrollFn:
    def generate_unroll(
        env_state: State,
        key: PRNGKey,
        env: Env,
        unroll_length: int,
        extra_fields: Sequence[str] = (),
    ):
        def roll(
            carry: tuple[State, ..., PRNGKey], _
        ) -> tuple[tuple[State, ..., PRNGKey], Any]:
            *args, key = carry
            key, next_key = jax.random.split(key)
            next_state, *next_carry, output = fun(
                *args, key=next_key, env=env, extra_fields=extra_fields, **kwargs
            )
            return (next_state, *next_carry, next_key), output

        (env_state, *_), transitions = jax.lax.scan(
            roll,
            (env_state, *initial_carry_extras, key),
            (),
            length=unroll_length,
        )
        if remove_roll_axis:
            transitions = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), transitions
            )
        return env_state, transitions

    return generate_unroll


def _actor_step(
    env_state: State,
    key: PRNGKey,
    *,
    env: Env,
    policy: Policy,
    extra_fields: Sequence[str] = (),
    accumulate_pipeline_states: bool = False,
) -> tuple[State, Transition]:
    """Collect data."""
    actions, policy_extras = policy(observation=env_state.obs, sample_key=key)
    next_state = env.step(env_state, actions)
    extras = {
        "policy_extras": policy_extras,
        "state_extras": {x: next_state.info[x] for x in extra_fields},
    }
    if accumulate_pipeline_states:
        extras |= {"pipeline_states": env_state.pipeline_state}
    transition = Transition(
        observation=env_state.obs,
        action=actions,
        reward=next_state.reward,
        discount=1 - next_state.done,
        next_observation=next_state.obs,
        extras=extras,
    )
    return next_state, transition


def proprioceptive_unroll_factory(
    policy: Policy, accumulate_pipeline_states: bool = False
) -> GenerateUnrollFn:
    return unroll_factory(
        _actor_step, policy=policy, accumulate_pipeline_states=accumulate_pipeline_states
    )


def _vision_actor_step(
    env_state: State,
    key: PRNGKey,
    *,
    env: VisionWrapper,
    policy: Policy,
    vision_encoder: Callable[[Observation], jax.Array] | None = None,
    extra_fields: Sequence[str] = (),
    proprio_substeps: int = 1,
    accumulate_pipeline_states: bool = False,
) -> tuple[State, Transition]:
    vision_obs = env.get_vision_obs(env_state.pipeline_state, env_state.info)
    if vision_encoder is not None:
        latent_encoding = vision_encoder(observation=vision_obs)
        policy = functools.partial(policy, latent_encoding=latent_encoding)
    generate_proprioceptive_unroll = proprioceptive_unroll_factory(
        policy=policy, accumulate_pipeline_states=accumulate_pipeline_states
    )
    next_state, transitions = generate_proprioceptive_unroll(
        env_state, key, env, proprio_substeps, extra_fields
    )
    vision_obs = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), vision_obs)
    # add vision observations to transitions
    transitions = Transition(
        observation=transitions.observation | vision_obs,
        action=transitions.action,
        reward=transitions.reward,
        discount=transitions.discount,
        next_observation=transitions.next_observation | vision_obs,
        extras=transitions.extras,
    )
    return next_state, transitions


def vision_unroll_factory(
    policy: Policy,
    vision_encoder: Callable[[Observation], jax.Array] | None = None,
    proprio_substeps: int = 1,
    accumulate_pipeline_states: bool = False,
) -> GenerateUnrollFn:
    return unroll_factory(
        _vision_actor_step,
        policy=policy,
        vision_encoder=vision_encoder,
        proprio_substeps=proprio_substeps,
        remove_roll_axis=True,
        accumulate_pipeline_states=accumulate_pipeline_states,
    )


def _recurrent_actor_step(
    env_state: State,
    recurrent_carry: RecurrentCarry,
    initial_encoding: jax.Array,
    key: PRNGKey,
    *,
    env: VisionWrapper,
    policy: Policy,
    recurrent_encoder: RecurrentEncoder,
    extra_fields: Sequence[str] = (),
    vision_substeps: int = 0,
    proprio_substeps: int = 1,
    accumulate_pipeline_states: bool = False,
) -> tuple[State, RecurrentCarry, jax.Array, Transition]:
    enriched_policy = functools.partial(policy, latent_encoding=initial_encoding)
    if vision_substeps > 0:
        generate_unroll = vision_unroll_factory(
            policy=enriched_policy,
            proprio_substeps=proprio_substeps,
            accumulate_pipeline_states=accumulate_pipeline_states,
        )
        n_substeps = vision_substeps
    else:
        generate_unroll = proprioceptive_unroll_factory(
            policy=enriched_policy, accumulate_pipeline_states=accumulate_pipeline_states
        )
        n_substeps = proprio_substeps
    key, recurrent_key = jax.random.split(key)
    next_state, transitions = generate_unroll(env_state, key, env, n_substeps, extra_fields)
    recurrent_key = jax.random.split(recurrent_key, initial_encoding.shape[0])
    transitions = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), transitions)
    next_encoding, recurrent_carry = recurrent_encoder(
        observation=transitions.observation,
        done=1 - transitions.discount,
        key=recurrent_key,
        recurrent_carry=recurrent_carry,
    )
    transitions = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), transitions)
    return next_state, recurrent_carry, next_encoding, transitions


def recurrent_unroll_factory(
    *,
    policy: Policy,
    recurrent_encoder: RecurrentEncoder,
    encoding_size: int,
    init_carry_fn: Callable[[PRNGKey], RecurrentCarry],
    vision_substeps: int = 0,
    proprio_substeps: int = 1,
    recurrent_carry: RecurrentCarry | None = None,
    accumulate_pipeline_states: bool = False,
) -> GenerateUnrollFn:
    def generate_unroll(
        env_state: State,
        key: PRNGKey,
        env: Env,
        unroll_length: int,
        extra_fields: Sequence[str] = (),
    ) -> tuple[State, Transition]:
        key, init_carry_key = jax.random.split(key)
        num_envs = env_state.done.shape[0] if env_state.done.shape else 1
        if recurrent_carry is None:
            init_carry_keys = jax.random.split(init_carry_key, num_envs)
            init_carry = jax.vmap(init_carry_fn)(init_carry_keys)
        else:
            init_carry = recurrent_carry
        initial_encoding = jnp.zeros((num_envs, encoding_size))
        gen_unroll = unroll_factory(
            _recurrent_actor_step,
            initial_carry_extras=(init_carry, initial_encoding),
            policy=policy,
            recurrent_encoder=recurrent_encoder,
            vision_substeps=vision_substeps,
            proprio_substeps=proprio_substeps,
            remove_roll_axis=True,
            accumulate_pipeline_states=accumulate_pipeline_states,
        )
        env_state, transitions = gen_unroll(env_state, key, env, unroll_length, extra_fields)
        return env_state, transitions

    return generate_unroll
