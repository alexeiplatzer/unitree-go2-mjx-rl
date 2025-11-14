"""Brax training acting functions."""

import time
from collections.abc import Callable, Sequence

import jax
import numpy as np

from quadruped_mjx_rl.environments.physics_pipeline import Env, EvalWrapper, State
from quadruped_mjx_rl.types import Metrics, Policy, PolicyParams, PRNGKey, Transition


def actor_step(
    env: Env,
    env_state: State,
    policy: Policy,
    key: PRNGKey,
    *,
    recurrent_state: jax.Array,
    extra_fields: Sequence[str] = (),
) -> tuple[State, jax.Array, Transition]:
    """Collect data."""
    actions, policy_extras, new_recurrent_state = policy(env_state.obs, key, recurrent_state)
    nstate = env.step(env_state, actions)
    state_extras = {x: nstate.info[x] for x in extra_fields}
    return nstate, new_recurrent_state, Transition(
        observation=env_state.obs,
        action=actions,
        reward=nstate.reward,
        discount=1 - nstate.done,
        next_observation=nstate.obs,
        recurrent_state=new_recurrent_state,
        extras={"policy_extras": policy_extras, "state_extras": state_extras},
    )


def generate_unroll(
    env: Env,
    env_state: State,
    policy: Policy,
    key: PRNGKey,
    first_recurrent_state: jax.Array,
    unroll_length: int,
    extra_fields: Sequence[str] = (),
) -> tuple[State, jax.Array, Transition]:
    """Collect trajectories of the given unroll_length."""

    @jax.jit
    def f(carry, unused_t):
        state, current_key, recurrent_state = carry
        current_key, next_key = jax.random.split(current_key)
        nstate, next_recurrent_state, transition = actor_step(
            env,
            state,
            policy,
            current_key,
            recurrent_state=recurrent_state,
            extra_fields=extra_fields,
        )
        return (nstate, next_key, next_recurrent_state), transition

    (final_state, _, final_recurrent_state), data = jax.lax.scan(
        f, (env_state, key, first_recurrent_state), (), length=unroll_length
    )
    return final_state, final_recurrent_state, data


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

        def generate_eval_unroll(policy_params: PolicyParams, key: PRNGKey) -> State:
            reset_keys = jax.random.split(key, num_eval_envs)
            eval_first_state = eval_env.reset(reset_keys)
            return generate_unroll(
                eval_env,
                eval_first_state,
                eval_policy_fn(policy_params),
                key,
                first_recurrent_state=None,  # TODO
                unroll_length=episode_length // action_repeat,
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
