""" Teacher Student network """

from typing import Sequence, Tuple

from brax.training import distribution
from brax.training import networks
from brax.training import types
from brax.training.types import PRNGKey
from jax import numpy as jnp
import flax
from flax import linen
from brax.training.networks import MLP

from modules import TeacherStudentActorCritic


def make_inference_fn(network: TeacherStudentActorCritic):
    """Creates params and inference function for the PPO agent."""

    def make_policy(params: types.Params, deterministic: bool = False) -> types.Policy:
        def policy(
            observations: types.Observation,
            priveleged_observations: types.Observation,
            key_sample: PRNGKey,
        ) -> Tuple[types.Action, types.Extra]:
            logits = network.apply(params, observations, method=network.apply_teacher)
            if deterministic:
                return network.parametric_action_distribution.mode(logits), {}
            raw_actions = network.parametric_action_distribution.sample_no_postprocessing(
                logits, key_sample
            )
            log_prob = network.parametric_action_distribution.log_prob(logits, raw_actions)
            postprocessed_actions = network.parametric_action_distribution.postprocess(
                raw_actions
            )
            return postprocessed_actions, {
                "log_prob": log_prob,
                "raw_action": raw_actions,
            }

        return policy

    return make_policy
