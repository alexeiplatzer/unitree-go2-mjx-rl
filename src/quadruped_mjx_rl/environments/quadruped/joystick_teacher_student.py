"""A simple teacher student joystick environment."""

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp

from quadruped_mjx_rl.environments import EnvModel
from quadruped_mjx_rl.environments.physics_pipeline import EnvSpec, PipelineState
from quadruped_mjx_rl.environments.quadruped.base import QuadrupedBaseEnv
from quadruped_mjx_rl.environments.quadruped.base import register_environment_config_class
from quadruped_mjx_rl.environments.quadruped.joystick_base import (
    JoystickBaseEnvConfig,
    QuadrupedJoystickBaseEnv,
)
from quadruped_mjx_rl.robots import RobotConfig
from quadruped_mjx_rl.types import Observation


@dataclass
class TeacherStudentEnvironmentConfig(JoystickBaseEnvConfig):
    @dataclass
    class ObservationConfig(JoystickBaseEnvConfig.ObservationConfig):
        extended_history_length: int = 45

    observation_noise: ObservationConfig = field(default_factory=ObservationConfig)

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

    def __init__(
        self,
        environment_config: TeacherStudentEnvironmentConfig,
        robot_config: RobotConfig,
        env_model: EnvModel | EnvSpec,
    ):
        super().__init__(environment_config, robot_config, env_model)
        self._extended_history_length = environment_config.observation_noise.extended_history_length

    def _init_obs(
        self,
        pipeline_state: PipelineState,
        state_info: dict[str, Any],
    ) -> Observation:

        short_history = self._init_proprioceptive_obs(pipeline_state, state_info)
        state_obs_length = short_history.size // self._obs_config.history_length
        long_history = jnp.zeros(state_obs_length * self._extended_history_length)
        long_history = self._update_obs_history(
            obs_history=long_history, current_obs=short_history[-state_obs_length:]
        )
        privileged_obs = jnp.concatenate(self._get_privileged_obs_list())
        obs = {
            "proprioceptive": short_history,
            "proprioceptive_history": long_history,
            "environment_privileged": privileged_obs,
        }
        return obs

    def _get_obs(
        self,
        pipeline_state: PipelineState,
        state_info: dict[str, Any],
        previous_obs: Observation,
    ) -> Observation:
        short_history = self._get_proprioceptive_obs(pipeline_state, state_info, previous_obs["proprioceptive"])
        state_obs_length = short_history.size // self._obs_config.history_length
        long_history = self._update_obs_history(
            obs_history=previous_obs["proprioceptive_history"],
            current_obs=short_history[-state_obs_length:],
        )
        obs = {
            "proprioceptive": short_history,
            "proprioceptive_history": long_history,
            "environment_privileged": previous_obs["environment_privileged"],
        }
        return obs

    def _get_privileged_obs_list(self) -> list[jax.Array]:
        return [
            jnp.reshape(self._pipeline_model.model.geom_friction, -1),
            jnp.reshape(self._pipeline_model.model.actuator_gainprm, -1),
            jnp.reshape(self._pipeline_model.model.actuator_biasprm, -1),
        ]
