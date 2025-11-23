from quadruped_mjx_rl.environments.physics_pipeline import State, Wrapper, Env
from quadruped_mjx_rl.types import RecurrentHiddenState, InitCarryFn
import jax
from jax import numpy as jnp
from flax.struct import dataclass as flax_dataclass


@flax_dataclass
class RecurrentEnvState(State):
    recurrent_hidden_state: RecurrentHiddenState


class RecurrentWrapper(Wrapper):
    """This wrapper lets the environment take care of the recurrent state resetting."""

    def __init__(self, env: Env, init_carry_fn: InitCarryFn):
        super().__init__(env)
        self._init_carry_fn = init_carry_fn

    def reset(self, rng: jax.Array) -> RecurrentEnvState:
        env_rng, carry_rng = jax.random.split(rng, 2)
        state = self.env.reset(env_rng)
        recurrent_state = self._init_carry_fn(carry_rng)
        state = RecurrentEnvState(
            pipeline_state=state.pipeline_state,
            obs=state.obs,
            reward=state.reward,
            done=state.done,
            info=state.info,
            metrics=state.metrics,
            recurrent_hidden_state=recurrent_state,
        )
        state.info["first_recurrent_state"] = recurrent_state
        return state

    def step(
        self, state: RecurrentEnvState, action: jax.Array
    ) -> State:

        state = self.env.step(state, action)
        assert isinstance(state, RecurrentEnvState)

        def where_done(x, y):
            done = state.done
            if done.shape:
                done = jnp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
            return jnp.where(done, x, y)

        recurrent_state = jax.tree.map(
            where_done, state.info["first_recurrent_state"], state.recurrent_hidden_state
        )
        return state.replace(recurrent_state=recurrent_state)
