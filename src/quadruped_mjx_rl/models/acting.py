"""Brax training acting functions."""

import functools
from collections.abc import Callable, Sequence
from typing import Any

import jax
from jax import numpy as jnp

from quadruped_mjx_rl.environments.physics_pipeline import Env, State
from quadruped_mjx_rl.environments.vision.vision_wrappers import VisionWrapper
from quadruped_mjx_rl.types import (
    PRNGKey,
    Observation,
    Transition,
)
from quadruped_mjx_rl.models.types import (
    Policy,
    RecurrentPolicyMaker,
    RecurrentCarry,
)


def actor_step(
    env_state_and_key: tuple[State, PRNGKey],
    _,
    *,
    env: Env,
    policy: Policy,
    extra_fields: Sequence[str] = (),
) -> tuple[tuple[State, PRNGKey], Transition]:
    """Collect data."""
    env_state, key = env_state_and_key
    key, next_key = jax.random.split(key)
    actions, policy_extras = policy(observation=env_state.obs, key=key)
    nstate = env.step(env_state, actions)
    state_extras = {x: nstate.info[x] for x in extra_fields}
    transition = Transition(
        observation=env_state.obs,
        action=actions,
        reward=nstate.reward,
        discount=1 - nstate.done,
        next_observation=nstate.obs,
        extras={"policy_extras": policy_extras, "state_extras": state_extras},
    )
    return (nstate, next_key), transition


def vision_actor_step(
    env_state_and_key: tuple[State, PRNGKey],
    _,
    *,
    env: VisionWrapper,
    policy: Policy,
    extra_fields: Sequence[str] = (),
    proprio_substeps: int = 1,
) -> tuple[tuple[State, PRNGKey], Transition]:
    env_state, key = env_state_and_key
    (next_state, next_key), transitions = jax.lax.scan(
        functools.partial(actor_step, env=env, policy=policy, extra_fields=extra_fields),
        (env_state, key),
        (),
        length=proprio_substeps,
    )
    vision_obs = env.get_vision_obs(next_state.pipeline_state, next_state.info)
    next_state = next_state.replace(obs=next_state.obs | vision_obs)
    return (next_state, next_key), transitions


def recurrent_actor_step(
    carry: tuple[State, PRNGKey, Policy, RecurrentCarry],
    _,
    *,
    env: VisionWrapper,
    extra_fields: Sequence[str] = (),
    recurrent_policy_maker: RecurrentPolicyMaker,
    vision_substeps: int = 0,
    proprio_substeps: int = 1,
) -> tuple[tuple[State, PRNGKey, Policy, RecurrentCarry], Transition]:
    env_state, key, current_policy, recurrent_carry = carry
    if vision_substeps > 0:
        step_fn = functools.partial(
            vision_actor_step, proprio_substeps=proprio_substeps
        )
        n_substeps = vision_substeps
    else:
        step_fn = actor_step
        n_substeps = proprio_substeps
    (next_state, next_key), transitions = jax.lax.scan(
        functools.partial(step_fn, env=env, policy=current_policy, extra_fields=extra_fields),
        (env_state, key),
        (),
        length=n_substeps,
    )
    next_key, policy_maker_key = jax.random.split(next_key)
    next_policy, recurrent_carry = recurrent_policy_maker(
        transitions.observation, 1 - transitions.discount, policy_maker_key, recurrent_carry
    )
    return (next_state, next_key, next_policy, recurrent_carry), transitions


def generate_unroll(
    carry: tuple[State, PRNGKey, Policy, RecurrentCarry],
    _,
    *,
    env: Env | VisionWrapper,
    unroll_length: int,
    extra_fields: Sequence[str] = (),
    add_vision_obs: bool = False,
    proprio_steps_per_vision_step: int = 1,
    recurrent: bool = False,
    recurrent_carry: RecurrentCarry = None,
    vis_steps_per_rec_step: int = 1,
    recurrent_policy_maker: RecurrentPolicyMaker | None = None,
) -> tuple[tuple[State, PRNGKey, Policy, RecurrentCarry], Transition]:
    """Collect trajectories of the given unroll_length."""
    env_state, key, policy, recurrent_carry = carry

    return (env_state, key, policy, recurrent_carry), Transition()
