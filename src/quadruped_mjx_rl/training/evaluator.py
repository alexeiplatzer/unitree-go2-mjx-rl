import time
from typing import Callable

import jax
import numpy as np

from quadruped_mjx_rl.environments import Env, State
from quadruped_mjx_rl.environments.wrappers import EvalWrapper
from quadruped_mjx_rl.models.acting import generate_unroll
from quadruped_mjx_rl.models.types import Policy, PolicyParams, RecurrentCarry, RecurrentPolicyMaker
from quadruped_mjx_rl.types import Metrics, PRNGKey


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
        init_carry_fn: Callable[[PRNGKey], RecurrentCarry] | None = None,
        vis_steps_per_rec_step: int = 1,
        recurrent_encoder: RecurrentPolicyMaker | None = None,
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
            key, init_carry_key = jax.random.split(key)
            reset_keys = jax.random.split(key, num_eval_envs)
            init_carry_key = jax.random.split(init_carry_key, num_eval_envs)
            eval_first_state = eval_env.reset(reset_keys)
            recurrent_carry = (
                init_carry_fn(init_carry_key) if init_carry_fn is not None else None
            )
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
