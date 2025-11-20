"""A simple teacher student joystick environment."""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from quadruped_mjx_rl.environments.physics_pipeline import PipelineState
from quadruped_mjx_rl.environments.quadruped.base import QuadrupedBaseEnv
from quadruped_mjx_rl.environments.quadruped.base import register_environment_config_class
from quadruped_mjx_rl.environments.quadruped.joystick_base import (
    JoystickBaseEnvConfig,
    QuadrupedJoystickBaseEnv,
)


@dataclass
class TeacherStudentEnvironmentConfig(JoystickBaseEnvConfig):
    @classmethod
    def config_class_key(cls) -> str:
        return "TeacherStudent"

    @classmethod
    def get_environment_class(cls) -> type[QuadrupedBaseEnv]:
        return QuadrupedJoystickTeacherStudentEnv


register_environment_config_class(TeacherStudentEnvironmentConfig)


class QuadrupedJoystickTeacherStudentEnv(QuadrupedJoystickBaseEnv):
    """This environment implements observations as a dictionary of regular and privileged
    observations."""

    def _init_obs(
        self,
        pipeline_state: PipelineState,
        state_info: dict[str, Any],
    ) -> jax.Array | dict[str, jax.Array]:
        state_obs = self._get_proprioceptive_obs_vector(pipeline_state, state_info)
        privileged_obs = jnp.concatenate(self._get_privileged_obs_list())
        obs = {
            "state": state_obs,
            "privileged_state": privileged_obs,
        }
        if self._obs_config.history_length is not None:
            obs_history = jnp.zeros(state_obs.size * self._obs_config.history_length)
            obs["state_history"] = self._update_obs_history(
                obs_history=obs_history, current_obs=state_obs
            )
        return obs

    def _get_obs(
        self,
        pipeline_state: PipelineState,
        state_info: dict[str, Any],
        previous_obs: jax.Array | dict[str, jax.Array],
    ) -> jax.Array | dict[str, jax.Array]:
        state_obs = self._get_proprioceptive_obs_vector(pipeline_state, state_info)
        obs = {
            "proprioceptive": state_obs,
            "environment_privileged": previous_obs["environment_privileged"],
        }

        if self._obs_config.history_length is not None:
            assert isinstance(previous_obs, dict)
            obs["proprioceptive_history"] = self._update_obs_history(
                obs_history=previous_obs["proprioceptive_history"], current_obs=state_obs
            )
        return obs

    def _get_privileged_obs_list(self) -> list[jax.Array]:
        return [
            self._pipeline_model.model.geom_friction,
            self._pipeline_model.model.acturator_gainprm,
            self._pipeline_model.model.actuator_biasprm,
            # self._pipeline_model.model.geom_friction.reshape(-1),
            # self.sys.dof_frictionloss,
            # self.sys.dof_damping,
            # self.sys.jnt_stiffness,
            # self.sys.actuator_forcerange,
            # self.sys.body_mass[0],
        ]
