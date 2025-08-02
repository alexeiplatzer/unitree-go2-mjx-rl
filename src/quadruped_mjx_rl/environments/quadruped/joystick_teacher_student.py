"""Quadruped Joytick environment adapted for PPO training."""

# Typing
from dataclasses import dataclass

# Supporting
from etils.epath import PathLike

# Math
import jax
import jax.numpy as jnp

# Sim
from quadruped_mjx_rl.environments import QuadrupedBaseEnv
from quadruped_mjx_rl.environments.physics_pipeline import PipelineState, EnvModel, EnvSpec
from quadruped_mjx_rl.environments.quadruped.base import register_environment_config_class

# Definitions
from quadruped_mjx_rl.robots import RobotConfig
from quadruped_mjx_rl.environments.quadruped.joystick_base import (
    JoystickBaseEnvConfig,
    QuadrupedJoystickBaseEnv,
)


@dataclass
class TeacherStudentEnvironmentConfig(JoystickBaseEnvConfig):
    @classmethod
    def environment_class_key(cls) -> str:
        return "TeacherStudent"

    @classmethod
    def get_environment_class(cls) -> type[QuadrupedBaseEnv]:
        return QuadrupedJoystickTeacherStudentEnv


register_environment_config_class(TeacherStudentEnvironmentConfig)


class QuadrupedJoystickTeacherStudentEnv(QuadrupedJoystickBaseEnv):

    def __init__(
        self,
        environment_config: TeacherStudentEnvironmentConfig,
        robot_config: RobotConfig,
        env_spec: EnvSpec | EnvModel,
    ):
        super().__init__(environment_config, robot_config, env_spec)

    def _init_obs(
        self,
        pipeline_state: PipelineState,
        state_info: dict[str, ...],
    ) -> jax.Array | dict[str, jax.Array]:
        # TODO: compare to init obs in superclass, state info must be updated
        state_obs = self._get_state_obs(pipeline_state, state_info)
        obs_history = jnp.zeros(state_obs.size * 15)
        obs_history = self._update_obs_history(obs_history, state_obs)
        privileged_obs = self._get_privileged_obs()
        obs = {
            "state": state_obs,
            "state_history": obs_history,
            "privileged_state": privileged_obs,
        }
        return obs

    def _get_obs(
        self,
        pipeline_state: PipelineState,
        state_info: dict[str, ...],
        previous_obs: jax.Array | dict[str, jax.Array],
    ) -> jax.Array | dict[str, jax.Array]:
        assert isinstance(previous_obs, dict)
        obs_history = previous_obs["state_history"]
        state_obs = self._get_state_obs(pipeline_state, state_info)
        obs_history = self._update_obs_history(obs_history, state_obs)
        privileged_obs = self._get_privileged_obs()
        obs = {
            "state": state_obs,
            "state_history": obs_history,
            "privileged_state": privileged_obs,
        }
        return obs

    def _get_privileged_obs(self) -> jax.Array:
        return jnp.concatenate(
            [
                self.sys.geom_friction.reshape(-1),
                # self.sys.dof_frictionloss,
                # self.sys.dof_damping,
                # self.sys.jnt_stiffness,
                # self.sys.actuator_forcerange,
                # self.sys.body_mass[0],
            ]
        )
