"""Quadruped Joytick environment adapted for PPO training."""

from dataclasses import dataclass, field

# Supporting
from etils.epath import PathLike

# Math
import jax
import jax.numpy as jnp

# Brax
from brax import math
from brax.base import Motion, System, Transform
from brax.base import State as PipelineState
from brax.envs.base import State


from quadruped_mjx_rl.robots import RobotConfig
from quadruped_mjx_rl.environments.joystick_base import (
    JoystickBaseEnvConfig,
    QuadrupedJoystickBaseEnv,
    environment_config_classes,
    configs_to_env_classes,
)


_ENVIRONMENT_CLASS = "TeacherStudent"


@dataclass
class TeacherStudentEnvironmentConfig(JoystickBaseEnvConfig):
    environment_class: str = _ENVIRONMENT_CLASS


environment_config_classes[_ENVIRONMENT_CLASS] = TeacherStudentEnvironmentConfig


class QuadrupedJoystickTeacherStudentEnv(QuadrupedJoystickBaseEnv):

    def __init__(
        self,
        environment_config: TeacherStudentEnvironmentConfig,
        robot_config: RobotConfig,
        init_scene_path: PathLike,
    ):
        super().__init__(environment_config, robot_config, init_scene_path)

    def _init_obs(
        self,
        pipeline_state: PipelineState,
        state_info: dict[str, ...],
    ) -> jax.Array | dict[str, jax.Array]:
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


configs_to_env_classes[TeacherStudentEnvironmentConfig] = QuadrupedJoystickTeacherStudentEnv
