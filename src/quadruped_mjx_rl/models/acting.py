"""Brax training acting functions."""

import functools
from collections.abc import Callable, Sequence
from typing import Protocol

import jax
from jax import numpy as jnp

from quadruped_mjx_rl.physics_pipeline import Env, State
from quadruped_mjx_rl.environments.vision.vision_wrappers import VisionWrapper
from quadruped_mjx_rl.models.types import (
    Policy,
    RecurrentCarry,
    RecurrentEncoder,
)
from quadruped_mjx_rl.types import (
    PRNGKey,
    Transition,
    Observation,
)


class GenerateUnrollFn(Protocol):
    def __call__(
        self,
        env_state: State,
        key: PRNGKey,
        env: Env,
        unroll_length: int,
        extra_keys: Sequence[str] = (),
    ) -> tuple[State, Transition]:
        pass


def wrap_roll(fun: Callable[..., tuple]) -> Callable[..., tuple]:
    def wrap(carry, _, **kwargs):
        *args, key = carry
        key, next_key = jax.random.split(key)
        *next_carry, output = fun(*args, key=next_key, **kwargs)
        return (*next_carry, next_key), output

    return wrap


@wrap_roll
def actor_step(
    env_state: State,
    key: PRNGKey,
    *,
    env: Env,
    policy: Policy,
    extra_fields: Sequence[str] = (),
) -> tuple[State, Transition]:
    """Collect data."""
    actions, policy_extras = policy(observation=env_state.obs, sample_key=key)
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
    return nstate, transition


@wrap_roll
def vision_actor_step(
    env_state: State,
    key: PRNGKey,
    last_vision_obs: Observation,
    *,
    env: VisionWrapper,
    policy: Policy,
    vision_encoder: Callable[[Observation], jax.Array] | None = None,
    extra_fields: Sequence[str] = (),
    proprio_substeps: int = 1,
) -> tuple[State, Observation, Transition]:
    if vision_encoder is not None:
        latent_encoding = vision_encoder(last_vision_obs)
        policy = functools.partial(policy, latent_encoding=latent_encoding)
    (next_state, _), transitions = jax.lax.scan(
        functools.partial(actor_step, env=env, policy=policy, extra_fields=extra_fields),
        (env_state, key),
        (),
        length=proprio_substeps,
    )
    # add vision observations to transitions
    last_vision_obs = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), last_vision_obs)
    transitions = Transition(
        observation=dict(transitions.observation) | dict(last_vision_obs),
        action=transitions.action,
        reward=transitions.reward,
        discount=transitions.discount,
        next_observation=dict(transitions.next_observation) | dict(last_vision_obs),
        extras=transitions.extras,
    )

    # update the vision observations
    next_vision_obs = jax.vmap(env.get_vision_obs)(next_state.pipeline_state, next_state.info)
    # next_state = next_state.replace(obs=dict(next_state.obs) | dict(vision_obs))
    return next_state, next_vision_obs, transitions


@wrap_roll
def recurrent_actor_step(
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
) -> tuple[State, RecurrentCarry, jax.Array, Transition]:
    enriched_policy = functools.partial(policy, latent_encoding=initial_encoding)
    if vision_substeps > 0:
        step_fn = functools.partial(vision_actor_step, proprio_substeps=proprio_substeps)
        n_substeps = vision_substeps
    else:
        step_fn = actor_step
        n_substeps = proprio_substeps
    (next_state, final_key), transitions = jax.lax.scan(
        functools.partial(step_fn, env=env, policy=enriched_policy, extra_fields=extra_fields),
        (env_state, key),
        (),
        length=n_substeps,
    )
    next_encoding, recurrent_carry = recurrent_encoder(
        transitions.observation, 1 - transitions.discount, final_key, recurrent_carry
    )
    return next_state, recurrent_carry, next_encoding, transitions
