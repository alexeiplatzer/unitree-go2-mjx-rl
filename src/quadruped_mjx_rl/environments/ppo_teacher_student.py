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
from quadruped_mjx_rl.environments.ppo_enhanced import EnhancedEnvironmentConfig
from quadruped_mjx_rl.environments.ppo_enhanced import QuadrupedJoystickEnhancedEnv
from quadruped_mjx_rl.environments.base import environment_config_classes
from quadruped_mjx_rl.environments.base import configs_to_env_classes


_ENVIRONMENT_CLASS = "TeacherStudent"


@dataclass
class TeacherStudentEnvironmentConfig(EnhancedEnvironmentConfig):
    environment_class: str = _ENVIRONMENT_CLASS


environment_config_classes[_ENVIRONMENT_CLASS] = TeacherStudentEnvironmentConfig


class QuadrupedJoystickTeacherStudentEnv(QuadrupedJoystickEnhancedEnv):

    def __init__(
        self,
        environment_config: TeacherStudentEnvironmentConfig,
        robot_config: RobotConfig,
        init_scene_path: PathLike,
    ):
        super().__init__(environment_config, robot_config, init_scene_path)

    def reset(self, rng: jax.Array) -> State:
        state = super().reset(rng)

        return state

    def step(self, state: State, action: jax.Array) -> State:
        state = super().step(state, action)

        return state

    def _get_obs(
        self,
        pipeline_state: PipelineState,
        state_info: dict[str, ...],
        last_obs: jax.Array | dict[str, jax.Array],
    ) -> jax.Array | dict[str, jax.Array]:
        if isinstance(last_obs, jax.Array):
            last_obs = {"state_history": last_obs}
        obs_history = super()._get_obs(pipeline_state, state_info, last_obs["state_history"])
        current_obs = obs_history[:31]

        privileged_obs = jnp.concatenate(
            [
                self.sys.geom_friction.reshape(-1),
                # self.sys.dof_frictionloss,
                # self.sys.dof_damping,
                # self.sys.jnt_stiffness,
                # self.sys.actuator_forcerange,
                # self.sys.body_mass[0],
            ]
        )

        obs = {
            "state": current_obs,
            "state_history": obs_history,
            "privileged_state": privileged_obs,
        }

        return obs


configs_to_env_classes[TeacherStudentEnvironmentConfig] = QuadrupedJoystickTeacherStudentEnv
