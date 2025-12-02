"""Brax training acting functions."""

import functools
import time
from collections.abc import Callable, Sequence
from typing import Any

import jax
import numpy as np
from jax import numpy as jnp

from quadruped_mjx_rl.environments.physics_pipeline import Env, State
from quadruped_mjx_rl.environments.vision.vision_wrappers import VisionWrapper
from quadruped_mjx_rl.environments.wrappers import EvalWrapper
from quadruped_mjx_rl.types import (
    Metrics,
    Policy,
    PolicyParams,
    PRNGKey,
    Observation,
    RecurrentHiddenState,
    Transition,
    RecurrentEncoder,
)


def actor_step(
    env: Env,
    env_state: State,
    policy: Policy,
    key: PRNGKey,
    *,
    extra_fields: Sequence[str] = (),
) -> tuple[State, Transition]:
    """Collect data."""
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
    return nstate, transition

def generate_unroll(
    env: Env | VisionWrapper,
    env_state: State,
    policy: Policy,
    key: PRNGKey,
    unroll_length: int,
    extra_fields: Sequence[str] = (),
    add_vision_obs: bool = False,
    proprio_steps_per_vision_step: int = 1,
    recurrent: bool = False,
    recurrent_carry: RecurrentHiddenState = None,
    vis_steps_per_rec_step: int = 1,
    recurrent_encoder: RecurrentEncoder | None = None,
) -> tuple[State, Transition | tuple[Transition, Observation], RecurrentHiddenState]:
    """Collect trajectories of the given unroll_length."""

    @jax.jit
    def recurrent_superstep(
        carry, _,
    ):
        state, current_key, recurrent_encoding, recurrent_carry = carry
        current_key, next_key, encoder_key = jax.random.split(current_key, 3)
        current_policy = functools.partial(
            policy, recurrent_encoding=None
        )
        (next_state, _), transitions = jax.lax.scan(
            functools.partial(visually_enriched_step, policy=current_policy),
            (state, current_key),
            (),
            length=vis_steps_per_rec_step,
        )
        recurrent_encoding, recurrent_carry = recurrent_encoder(
            transitions.observation, transitions.done, encoder_key, recurrent_carry
        )
        return (next_state, next_key, recurrent_encoding, recurrent_carry), transitions

    @jax.jit
    def visually_enriched_step(
        carry: tuple[State, PRNGKey], _: Any, policy
    ) -> tuple[tuple[State, PRNGKey], Transition]:
        state, current_key = carry
        current_key, next_key = jax.random.split(current_key)

        (next_state, _), transitions = jax.lax.scan(
            proprioceptive_step,
            (state, current_key),
            (),
            length=proprio_steps_per_vision_step,
        )
        vision_obs = env.get_vision_obs(state.pipeline_state, state.info)
        next_state = next_state.replace(obs=next_state.obs | vision_obs)
        return (next_state, next_key), transitions

    @jax.jit
    def proprioceptive_step(
        carry: tuple[State, PRNGKey], _: Any, policy,
    ) -> tuple[tuple[State, PRNGKey], Transition]:
        state, current_key = carry
        current_key, next_key = jax.random.split(current_key)
        next_state, transition = actor_step(
            env,
            state,
            policy,
            current_key,
            extra_fields=extra_fields,
        )
        return (next_state, next_key), transition

    if recurrent:
        env_steps_per_rec_step = unroll_length // vis_steps_per_rec_step // proprio_steps_per_vision_step
        vis_steps_per_rec_step = unroll_length // vis_steps_per_rec_step
        dummy_state_obs = jax.tree.map(
            lambda x: jnp.zeros((env_steps_per_rec_step,) + x.shape), env_state.obs
        )
        example_vision_obs = env.get_vision_obs(env_state.pipeline_state, env_state.info)
        dummy_vision_obs = jax.tree.map(
            lambda x: jnp.zeros((vis_steps_per_rec_step,) + x.shape), example_vision_obs
        )
        dummy_done = jnp.zeros((env_steps_per_rec_step,) + env_state.done.shape)
        key, rec_encoder_key = jax.random.split(key)
        dummy_recurrent_encoding = recurrent_encoder(
            dummy_state_obs | dummy_vision_obs, dummy_done, rec_encoder_key, recurrent_carry
        )
        (final_state, _, recurrent_carry), data = jax.lax.scan(
            recurrent_superstep,
            (env_state, key, dummy_recurrent_encoding, recurrent_carry),
            (),
            length=unroll_length,
        )
    if add_vision_obs:
        (final_state, _, recurrent_carry), data = jax.lax.scan(
            visually_enriched_step,
            (env_state, key, recurrent_carry),
            (),
            length=unroll_length // proprio_steps_per_vision_step,
        )
    else:
        (final_state, _, recurrent_carry), data = jax.lax.scan(
            proprioceptive_step,
            (env_state, key, recurrent_carry),
            (),
            length=unroll_length,
        )
    return final_state, data, recurrent_carry


class Evaluator:
    """Class to run evaluations."""

    def __init__(
        self,
        eval_env: Env,
        eval_policy_fn: Callable[[PolicyParams], Policy],
        num_eval_envs: int,
        episode_length: int,
        action_repeat: int,
        key: PRNGKey,
        vision: bool = False,
        vision_supersteps: int = 1,
        recurrent: bool = False,
        init_carry_fn: Callable[[PRNGKey], RecurrentHiddenState] | None = None,
        vis_steps_per_rec_step: int = 1,
        recurrent_encoder: RecurrentEncoder | None = None,
    ):
        """Init.

        Args:
          eval_env: Batched environment to run evals on.
          eval_policy_fn: Function returning the policy from the policy parameters.
          num_eval_envs: Each env will run 1 episode in parallel for each eval.
          episode_length: Maximum length of an episode.
          action_repeat: Number of physics steps per env step.
          key: RNG key.
        """
        self._key = key
        self._eval_walltime = 0.0

        eval_env = EvalWrapper(eval_env)

        def generate_eval_unroll(
            policy_params: PolicyParams, key: PRNGKey
        ) -> State:
            key, init_carry_key = jax.random.split(key)
            reset_keys = jax.random.split(key, num_eval_envs)
            init_carry_key = jax.random.split(init_carry_key, num_eval_envs)
            eval_first_state = eval_env.reset(reset_keys)
            recurrent_carry = init_carry_fn(init_carry_key) if init_carry_fn is not None else None
            return generate_unroll(
                env=eval_env,
                env_state=eval_first_state,
                policy=eval_policy_fn(policy_params),
                key=key,
                unroll_length=episode_length // action_repeat,
                add_vision_obs=vision,
                proprio_steps_per_vision_step=vision_supersteps,
                recurrent_carry=recurrent_carry,
                recurrent=recurrent,
                recurrent_encoder=recurrent_encoder,
                vis_steps_per_rec_step=vis_steps_per_rec_step,
            )[0]

        self._generate_eval_unroll = jax.jit(generate_eval_unroll)
        self._steps_per_unroll = episode_length * num_eval_envs

    def run_evaluation(
        self,
        policy_params: PolicyParams,
        training_metrics: Metrics,
        aggregate_episodes: bool = True,
    ) -> Metrics:
        """Run one epoch of evaluation."""
        self._key, unroll_key = jax.random.split(self._key)

        t = time.time()
        eval_state = self._generate_eval_unroll(policy_params, unroll_key)
        eval_metrics = eval_state.info["eval_metrics"]
        eval_metrics.active_episodes.block_until_ready()
        epoch_eval_time = time.time() - t
        metrics = {}
        for fn in [np.mean, np.std]:
            suffix = "_std" if fn == np.std else ""
            metrics.update(
                {
                    f"eval/episode_{name}{suffix}": (fn(value) if aggregate_episodes else value)
                    for name, value in eval_metrics.episode_metrics.items()
                }
            )
        metrics["eval/avg_episode_length"] = np.mean(eval_metrics.episode_steps)
        metrics["eval/std_episode_length"] = np.std(eval_metrics.episode_steps)
        metrics["eval/epoch_eval_time"] = epoch_eval_time
        metrics["eval/sps"] = self._steps_per_unroll / epoch_eval_time
        self._eval_walltime = self._eval_walltime + epoch_eval_time
        metrics = {
            "eval/walltime": self._eval_walltime,
            **training_metrics,
            **metrics,
        }

        return metrics
